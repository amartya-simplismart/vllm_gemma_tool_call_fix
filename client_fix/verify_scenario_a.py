"""Measure the scenario A fixes: prompt gating and per-turn `tool_choice`.

Runs the repro's three scenarios in four configurations against any OpenAI-compatible endpoint,
so the two fixes can be compared on the same server and prompts:

    1. baseline          — repro prompt, tool_choice absent (auto)
    2. prompt only       — gated prompt, tool_choice absent (auto)
    3. tool_choice only  — repro prompt, per-turn policy from tool_choice_policy.py
    4. both              — gated prompt + per-turn policy

Scenario definitions, tools, and the JSON schema are imported from the repro script, which is
never modified — it stays the acceptance check.

    python verify_scenario_a.py --base-url http://127.0.0.1:8090/v1 --api-key dummy --runs 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

from tool_choice_policy import REPRO_RULES, ToolPolicy

REPRO = Path(__file__).resolve().parent.parent / "repro_simplismart_json_toolcall.py"


def load_repro():
    spec = importlib.util.spec_from_file_location("repro", REPRO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r = load_repro()
policy = ToolPolicy(REPRO_RULES)

# The negative rule from prompt_tool_gating.md, applied to the repro's stand-in prompt.
GATED_PROMPT = r.SYSTEM_PROMPT.replace(
    "- Call fetch_seller_details when the buyer asks about the seller/supplier.",
    "- Call fetch_seller_details ONLY when the buyer explicitly asks about the seller/supplier.\n"
    "  NEVER call it in STAGE_INTRO or to answer a question about the buyer's own enquiry.\n"
    "  If no tool is clearly required, do not call any tool — just reply with the JSON object.",
)

# The stage each scenario's turn happens in — in production this comes from the envelope's
# `next_stage`, which the runtime already tracks.
SCENARIO_STAGE = {
    "A": "STAGE_INTRO",
    "B": "STAGE_CLOSING",
    "C": "STAGE_SELLER_DETAILS",
}


def call(client, model, messages, tool_choice, max_tokens):
    kwargs = dict(
        model=model,
        messages=messages,
        tools=r.CHAT_TOOLS,
        temperature=0.7,
        max_tokens=max_tokens,
        response_format=r.RESPONSE_FORMAT,
    )
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    message = client.chat.completions.create(**kwargs).choices[0].message
    tool_calls = [
        {"id": t.id, "name": t.function.name, "arguments": t.function.arguments}
        for t in (message.tool_calls or [])
    ]
    return (message.content or ""), tool_calls


def envelope_ok(content: str) -> bool:
    return r.parse_envelope(content) is not None


def run_scenario(client, args, name, *, gated_prompt: bool, use_policy: bool) -> bool:
    """One scenario, one configuration. Returns True on the repro's own PASS criteria."""
    scenario = r.SCENARIOS[name]
    messages = r.conversation_input(scenario["user"])
    if gated_prompt:
        messages = [{"role": "system", "content": GATED_PROMPT}] + messages[1:]

    stage = SCENARIO_STAGE[name]
    choice = policy.for_turn(stage=stage, after_tool_result=False) if use_policy else None
    content, tool_calls = call(client, args.model, messages, choice, args.max_tokens)

    if name == "A":
        return envelope_ok(content) and not tool_calls

    if name == "B":
        return any(t["name"] == "hangup_call_with_custom_delay" for t in tool_calls)

    fetch = next((t for t in tool_calls if t["name"] == "fetch_seller_details"), None)
    if fetch is None:
        return False
    # Follow-up request: the policy sends "none" here, which is what guarantees an envelope
    # instead of a second tool call.
    followup = r.tool_roundtrip_input(messages, fetch, r.FETCH_SELLER_RESULT)
    choice2 = policy.for_turn(stage=stage, after_tool_result=True) if use_policy else None
    content2, _ = call(client, args.model, followup, choice2, args.max_tokens)
    return envelope_ok(content2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default="gemma4")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()

    client = OpenAI(
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
        base_url=args.base_url,
    )

    configs = [
        ("1 baseline         ", False, False),
        ("2 prompt only      ", True, False),
        ("3 tool_choice only ", False, True),
        ("4 both             ", True, True),
    ]

    print(f"Target: {args.base_url}   model: {args.model}   runs: {args.runs}\n")
    header = "config".ljust(20) + "".join(f"  {s}".ljust(10) for s in ("A", "B", "C")) + "  total"
    print(header)
    print("-" * len(header))

    overall_ok = True
    for label, gated, use_policy in configs:
        cells, total = [], 0
        for name in ("A", "B", "C"):
            passed = sum(
                run_scenario(client, args, name, gated_prompt=gated, use_policy=use_policy)
                for _ in range(args.runs)
            )
            total += passed
            cells.append(f"{passed}/{args.runs}")
        line = label.ljust(20) + "".join(f"  {c}".ljust(10) for c in cells)
        print(f"{line}  {total}/{args.runs * 3}")
        if use_policy and gated:
            overall_ok = total == args.runs * 3

    print("\n'both' is the recommended configuration: the prompt fixes the model's judgment, "
          "\nand the per-turn tool_choice makes speech-only turns grammar-enforced.")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
