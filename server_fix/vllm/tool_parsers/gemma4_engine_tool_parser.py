# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import replace
from typing import Any

from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.parser.engine.registered_adapters import Gemma4ParserToolAdapter
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.gemma4_structural_tag import build_envelope_plus_tools_tag

logger = init_logger(__name__)

AnyRequest = ChatCompletionRequest | ResponsesRequest


class Gemma4EngineToolParser(Gemma4ParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False

    def adjust_request(self, request: AnyRequest) -> AnyRequest:
        """Choose the structured-output constraint for a Gemma4 request.

        The rule: whenever a caller constrains text output with a JSON schema *and* passes
        tools, the constraint must be a union grammar that leaves gemma4's native
        ``<|tool_call>call:...`` syntax reachable. A bare JSON grammar accepts only the schema
        object, so a tool call can never be *sampled* — tools silently never fire, no matter
        what the tool parser does downstream. See
        :mod:`vllm.tool_parsers.gemma4_structural_tag`.

        Cases:

        1. ``tool_choice`` required/named, no JSON schema — skip structured output entirely, as
           before. Gemma4 emits its native syntax and the parser extracts it directly; the base
           ``ToolParser.adjust_request`` would force JSON via guided decoding, conflicting with
           that syntax (it leaks as content and crashes EngineCore under speculative decoding).
           Mirrors the GLM4 parser.

        2. ``tool_choice`` required/named **with** a JSON schema — union grammar that *requires*
           a tool call (named choice: only that tool). Returning early here, as this parser did
           before, left the caller's schema in place and made the demanded tool call
           unreachable.

        3. ``tool_choice`` auto/unset **with** a JSON schema — union grammar where a tool call
           is permitted but not forced: schema object, tool call, or both.

        4. anything else — default behavior.
        """
        if not request.tools:
            return request

        tool_choice = request.tool_choice
        structured_outputs = getattr(request, "structured_outputs", None)
        if structured_outputs is not None and structured_outputs.structural_tag is not None:
            # A caller that sent its own structural tag knows what it wants.
            return request

        schema = _extract_json_schema(request)
        forced_name = _forced_tool_name(tool_choice)
        is_forced = forced_name is not None or tool_choice == "required"

        # Case 1.
        if is_forced and schema is None:
            request.skip_special_tokens = False
            return request

        # Cases 2 and 3.
        if schema is not None and (is_forced or tool_choice in (None, "auto")):
            return self._apply_union_grammar(
                request,
                schema,
                # forced choice: text alone is not an acceptable output
                allow_envelope_only=not is_forced,
                only_tool=forced_name,
            )

        # Case 4.
        return super().adjust_request(request)

    def _apply_union_grammar(
        self,
        request: AnyRequest,
        schema: Any,
        *,
        allow_envelope_only: bool = True,
        only_tool: str | None = None,
    ) -> AnyRequest:
        tool_names = _tool_names(request)
        if only_tool is not None:
            tool_names = [name for name in tool_names if name == only_tool]
        if not tool_names:
            return super().adjust_request(request)

        # The parser engine knows whether the reasoning channel is active for this
        # deployment; when in doubt allow it — an optional prefix the model never opens
        # costs nothing, whereas forbidding it breaks a channel the model was trained to use.
        reasoning = bool(getattr(self._parser_engine, "_has_reasoning", True))

        tag = build_envelope_plus_tools_tag(
            schema,
            tool_names,
            reasoning=reasoning,
            allow_envelope_only=allow_envelope_only,
        )
        tag_json = json.dumps(tag.model_dump(by_alias=True, exclude_none=True))

        # Clear the json_schema constraint first: both request types reject having a response
        # format and ``structured_outputs`` set at the same time.
        _clear_json_schema(request)
        existing = getattr(request, "structured_outputs", None)
        if existing is None:
            request.structured_outputs = StructuredOutputsParams(  # type: ignore[call-arg]
                structural_tag=tag_json
            )
        else:
            # Preserve unrelated knobs (whitespace handling, backend hints) and swap the
            # constraint. ``StructuredOutputsParams`` allows exactly one constraint.
            request.structured_outputs = replace(
                existing,
                json=None,
                json_object=None,
                regex=None,
                choice=None,
                grammar=None,
                structural_tag=tag_json,
            )

        logger.debug(
            "gemma4: replaced json_schema constraint with union grammar over tools %s "
            "(reasoning=%s, text_alone_allowed=%s)",
            tool_names,
            reasoning,
            allow_envelope_only,
        )

        # Also runs the parser engine's own adjustment (skip_special_tokens = False), which is
        # required: <|tool_call> / <tool_call|> must survive detokenization to be parsed.
        return self._parser_engine.adjust_request(request)


# ------------------------------------------------------------------------------------------
# Request-shape helpers. Chat Completions carries the schema on ``response_format``, the
# Responses API on ``text.format`` — production traffic uses the latter.
# ------------------------------------------------------------------------------------------


def _extract_json_schema(request: AnyRequest) -> Any | None:
    """Return the JSON schema constraining this request's text output, if any."""
    response_format = getattr(request, "response_format", None)
    if response_format is not None and getattr(response_format, "type", None) == "json_schema":
        json_schema = getattr(response_format, "json_schema", None)
        schema = getattr(json_schema, "json_schema", None)
        if schema is not None:
            return schema

    text = getattr(request, "text", None)
    text_format = getattr(text, "format", None)
    if text_format is not None and getattr(text_format, "type", None) == "json_schema":
        schema = getattr(text_format, "schema_", None)
        if schema is not None:
            return schema

    structured_outputs = getattr(request, "structured_outputs", None)
    if structured_outputs is not None and structured_outputs.json is not None:
        return structured_outputs.json

    return None


def _clear_json_schema(request: AnyRequest) -> None:
    if getattr(request, "response_format", None) is not None:
        request.response_format = None
    text = getattr(request, "text", None)
    if text is not None and getattr(text, "format", None) is not None:
        text.format = None


def _forced_tool_name(tool_choice: Any) -> str | None:
    """Return the tool name for a named tool_choice, else None."""
    if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
        return tool_choice.function.name
    if isinstance(tool_choice, ToolChoiceFunction):
        return tool_choice.name
    return None


def _tool_names(request: AnyRequest) -> list[str]:
    names = []
    for tool in request.tools or []:
        # Chat Completions: {"type": "function", "function": {"name": ...}}
        # Responses:        {"type": "function", "name": ...}
        name = getattr(getattr(tool, "function", None), "name", None) or getattr(
            tool, "name", None
        )
        if name:
            names.append(name)
    return names
