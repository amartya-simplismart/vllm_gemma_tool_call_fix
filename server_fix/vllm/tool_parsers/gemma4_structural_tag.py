"""Build an xgrammar structural tag that lets gemma4 emit a JSON envelope AND/OR tool calls.

Why this exists
---------------
When a request carries ``response_format={"type":"json_schema",...}``, vLLM turns it into
``StructuredOutputsParams(json=<schema>)`` — a grammar whose *only* accepting string is the
JSON object. Gemma4 signals a tool call with its native ``<|tool_call>call:name{...}<tool_call|>``
syntax. The ``<|tool_call>`` special token still gets emitted (special tokens are not masked),
but the grammar then forces the JSON object, so the tool call is **coerced into the envelope
string as escaped text** and the envelope never terminates — the request burns its whole token
budget and the client gets a malformed envelope with no usable tool call. Measured raw output
under the json_schema constraint::

    <|tool_call>{"assistant_reply": "<|tool_call>call:get_weather{city:\"Tokyo\"}<tool_call|>"
    \n  \n  \n  …   (to max_tokens)

This is independent of the ``--dyn-tool-call-parser gemma4`` parser, which only runs on
already-generated text.

The fix is one grammar that accepts all three shapes the SquadStack runtime handles:

    1. envelope only                      (normal turn)
    2. tool call(s) only                  (tool turn — the bug today: unreachable)
    3. envelope followed by tool call(s)  (dual output — allowed, not forced)

Tool-call *arguments* are left unconstrained inside the tag on purpose: gemma4 writes them in
its own dialect (``key:<|"|>value<|"|>``), not JSON, so a JSON grammar there would force the
model off its trained format. This matches ``strict: false`` tools, which is what SquadStack
sends — no argument-level schema enforcement is expected.

Self-test (requires xgrammar, which vLLM already depends on)::

    python gemma4_structural_tag.py
"""

from __future__ import annotations

from typing import Any, Iterable

from xgrammar.structural_tag import (
    AnyTextFormat,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    SequenceFormat,
    StructuralTag,
    TagFormat,
    TagsWithSeparatorFormat,
)

# Gemma4 channel markers, as they appear in decoded text.
# Kept in sync with vllm/tool_parsers/gemma4_utils.py and vllm/parser/gemma4.py.
TOOL_CALL_BEGIN_PREFIX = "<|tool_call>call:"
TOOL_CALL_END = "<tool_call|>"
THOUGHT_BEGIN = "<|channel>thought\n"
THOUGHT_END = "<channel|>"


def build_envelope_plus_tools_tag(
    envelope_schema: dict[str, Any] | bool,
    tool_names: Iterable[str],
    *,
    reasoning: bool = True,
    allow_envelope_only: bool = True,
    allow_tools_only: bool = True,
) -> StructuralTag:
    """Grammar: [thought] ( envelope [tool_calls] | tool_calls ).

    Args:
        envelope_schema: the JSON schema from ``response_format.json_schema.schema``.
        tool_names: names of the tools in the request (``tool_choice`` is "auto"/absent).
        reasoning: allow an optional leading thinking block. Must be True whenever the model
            may open the thought channel (``<|think|>`` in the system prompt with
            ``--dyn-reasoning-parser gemma4``), otherwise the grammar forbids a channel the
            model was trained to use. Both openings are accepted — see below.
        allow_envelope_only: keep shape 1 legal (turn off only to force a tool call).
        allow_tools_only: keep shape 2 legal — this is the branch that fixes the bug.
    """
    tool_names = list(tool_names)
    if not tool_names:
        raise ValueError("no tools in request — plain json_schema is already correct")

    tool_calls = TagsWithSeparatorFormat(
        tags=[
            TagFormat(
                begin=TOOL_CALL_BEGIN_PREFIX + name,
                content=AnyTextFormat(),  # native <|"|> arg dialect, unconstrained
                end=TOOL_CALL_END,
            )
            for name in tool_names
        ],
        separator="",
        at_least_one=True,
    )
    envelope = JSONSchemaFormat(json_schema=envelope_schema)

    branches: list[Any] = []
    if allow_envelope_only:
        # envelope, optionally followed by tool calls -> shapes 1 and 3
        branches.append(SequenceFormat(elements=[envelope, OptionalFormat(content=tool_calls)]))
    else:
        branches.append(SequenceFormat(elements=[envelope, tool_calls]))
    if allow_tools_only:
        branches.append(tool_calls)  # shape 2

    body = branches[0] if len(branches) == 1 else OrFormat(elements=branches)

    if not reasoning:
        return StructuralTag(format=body)

    # Two ways generation can begin inside the thought channel, and the grammar must accept
    # both because it cannot see the prompt:
    #   a) the model opens it itself -> "<|channel>thought\n … <channel|>"
    #   b) the chat template already left the prompt inside an open <|channel> block, so
    #      generation starts in the reasoning state and only ever emits the closer. This is the
    #      post-tool-response continuation case handled by
    #      Gemma4Parser.adjust_initial_state_from_prompt() (vllm/parser/gemma4.py, issue #45834)
    #      — i.e. exactly SquadStack's request 2 after a function_call_output.
    thought = OrFormat(
        elements=[
            TagFormat(begin=THOUGHT_BEGIN, content=AnyTextFormat(), end=THOUGHT_END),
            TagFormat(begin="", content=AnyTextFormat(), end=THOUGHT_END),
        ]
    )
    return StructuralTag(
        format=SequenceFormat(elements=[OptionalFormat(content=thought), body])
    )


# ----------------------------------------------------------------------------------------
# Self-test: compile the grammar and check every shape the runtime must support.
# ----------------------------------------------------------------------------------------

_ENVELOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "assistant_reply": {"type": "string"},
        "memory": {
            "type": ["object", "null"],
            "properties": {
                key: {"type": "string"}
                for key in ("language", "buyer_interested", "seller_details_shared", "callback_time")
            },
        },
        "next_stage": {
            "type": ["string", "null"],
            "enum": ["STAGE_INTRO", "STAGE_SELLER_DETAILS", "STAGE_CLOSING", None],
        },
    },
    "required": ["assistant_reply", "memory", "next_stage"],
}
_TOOLS = ["fetch_seller_details", "hangup_call_with_custom_delay"]


def _self_test() -> int:
    import xgrammar as xgr
    from xgrammar.testing import _is_grammar_accept_string as accepts

    tag = build_envelope_plus_tools_tag(_ENVELOPE_SCHEMA, _TOOLS)
    grammar = xgr.Grammar.from_structural_tag(tag)

    envelope = (
        '{"assistant_reply": "theek hai ji", "memory": {"language": "hindi"}, '
        '"next_stage": "STAGE_INTRO"}'
    )
    hangup = (
        '<|tool_call>call:hangup_call_with_custom_delay'
        '{message:<|"|>theek hai, dhanyavaad<|"|>,delay:6}<tool_call|>'
    )
    fetch = "<|tool_call>call:fetch_seller_details{}<tool_call|>"
    thought = "<|channel>thought\nbuyer is closing the call<channel|>"
    # prompt already opened the channel (post-tool-response turn): closer only
    thought_continued = "buyer asked for seller details<channel|>"

    expectations = [
        (True, "shape 1: envelope only", envelope),
        (True, "shape 2: tool call only (the bug today)", hangup),
        (True, "shape 2: zero-arg tool call only", fetch),
        (True, "shape 3: envelope + tool call", envelope + hangup),
        (True, "shape 3 + thinking channel", thought + envelope + hangup),
        (True, "shape 2 + thinking channel", thought + fetch),
        (True, "shape 1, prompt-opened thought channel (closer only)", thought_continued + envelope),
        (True, "shape 2, prompt-opened thought channel (closer only)", thought_continued + fetch),
        (False, "reject: unconstrained prose", "sorry, I can't help with that"),
        (False, "reject: envelope missing required keys", '{"assistant_reply": "hi"}'),
        (False, "reject: tool not in request", "<|tool_call>call:transfer_call{}<tool_call|>"),
    ]

    failures = 0
    for expected, label, text in expectations:
        got = accepts(grammar, text)
        ok = got is expected
        failures += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {'accepts' if got else 'rejects':>7}  {label}")
    print(f"\n  {len(expectations) - failures}/{len(expectations)} grammar assertions passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
