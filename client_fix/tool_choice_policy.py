"""Per-turn `tool_choice` policy — the grammar-enforced half of the scenario A fix.

Why this exists
---------------
Once the server-side fix unlocks the tool channel (see ``../server_fix`` and ``../dynamo_fix``),
the model *may* call a tool on any turn. On a voice call that is usually right, but on some turns
it is definitionally wrong — an intro turn should speak, and the turn right after a tool result
should deliver the envelope, not call another tool.

Sending `tool_choice: "none"` on those turns is enforced by the decoding grammar: the server emits
no tool branch at all, so it does not depend on the model's judgment. Measured on
gemma-4-31B-it through patched dynamo: 3/3 envelope, 0/3 tool calls, versus 1/3 and 2/3 with
`auto`.

This module is deliberately dependency-free so it can be dropped into the pipecat pipeline as-is.

Usage in the runtime::

    from tool_choice_policy import ToolPolicy, StageRules

    policy = ToolPolicy(StageRules(
        tools_by_stage={
            "STAGE_INTRO": (),                              # speak only
            "STAGE_SELLER_DETAILS": ("fetch_seller_details",),
            "STAGE_CLOSING": ("hangup_call_with_custom_delay",),
        },
        all_tools=("fetch_seller_details", "hangup_call_with_custom_delay"),
    ))

    request["tool_choice"] = policy.for_turn(
        stage=current_stage,
        after_tool_result=is_follow_up_request,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

# OpenAI tool_choice values this policy emits.
AUTO = "auto"
NONE = "none"


def named(tool_name: str) -> dict[str, Any]:
    """A forced single-tool choice, in Chat Completions shape."""
    return {"type": "function", "function": {"name": tool_name}}


@dataclass(frozen=True)
class StageRules:
    """Which tools each call stage may use.

    Args:
        tools_by_stage: stage name -> tool names permitted in that stage. An empty tuple means
            the stage is speech-only and the turn is sent with ``tool_choice="none"``.
        all_tools: every tool the campaign defines. Stages missing from ``tools_by_stage`` fall
            back to this (i.e. ``auto`` over everything) so a new stage never silently loses
            its tools.
        force_envelope_after_tool_result: send ``"none"`` on the follow-up request that carries a
            ``function_call_output``. That request exists precisely to turn the tool result into
            the envelope, so a second tool call there is never what the runtime wants.
    """

    tools_by_stage: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    all_tools: tuple[str, ...] = ()
    force_envelope_after_tool_result: bool = True


class ToolPolicy:
    """Decide `tool_choice` for one turn."""

    def __init__(self, rules: StageRules) -> None:
        self.rules = rules

    def for_turn(
        self,
        *,
        stage: str | None,
        after_tool_result: bool = False,
    ) -> str | dict[str, Any]:
        """Return the `tool_choice` value for this turn.

        - ``"none"`` — no tool branch is compiled into the grammar; the model must produce the
          schema-conforming envelope.
        - ``"auto"`` — the model chooses (the normal case on tool-capable stages).
        - a named choice — the stage permits exactly one tool, so name it: the grammar then
          admits only that tool, which also removes any chance of calling the wrong one.
        """
        if after_tool_result and self.rules.force_envelope_after_tool_result:
            return NONE

        allowed = self.allowed_tools(stage)
        if not allowed:
            return NONE
        if len(allowed) == 1:
            return named(allowed[0])
        return AUTO

    def allowed_tools(self, stage: str | None) -> tuple[str, ...]:
        """Tools permitted in `stage`; unknown stages keep every tool."""
        if stage is None:
            return self.rules.all_tools
        return tuple(self.rules.tools_by_stage.get(stage, self.rules.all_tools))

    def filter_tools(self, tools: Iterable[Mapping[str, Any]], stage: str | None) -> list[Any]:
        """Optionally narrow the `tools` array itself to what the stage permits.

        Sending fewer tools shortens the prompt and keeps the tool list and `tool_choice`
        consistent. Accepts both Chat Completions (``{"function": {"name": ...}}``) and
        Responses (``{"name": ...}``) tool shapes.
        """
        allowed = set(self.allowed_tools(stage))
        kept = []
        for tool in tools:
            name = tool.get("name") or tool.get("function", {}).get("name")
            if name in allowed:
                kept.append(tool)
        return kept


# ------------------------------------------------------------------------------------------
# The rules for the campaign used in the repro. Mirror this per campaign.
# ------------------------------------------------------------------------------------------

REPRO_RULES = StageRules(
    tools_by_stage={
        # intro: greet and understand the enquiry — speaking only
        "STAGE_INTRO": (),
        # seller stage: may look the seller up
        "STAGE_SELLER_DETAILS": ("fetch_seller_details",),
        # closing: may hang up
        "STAGE_CLOSING": ("hangup_call_with_custom_delay",),
    },
    all_tools=("fetch_seller_details", "hangup_call_with_custom_delay"),
)
