"""Standalone repro: how SquadStack production calls GPT-4.1 with structured output + tools.

Replicates EXACTLY what SquadStack production sends to OpenAI, so the same behavior can be
reproduced and used as the reference when testing structured output (json_schema) and tool
calling together.

Endpoint request shape -> POST /v1/chat/completions:
  stream=true, system prompt as a system message,
  response_format={"type":"json_schema","json_schema":{...}},
  Chat Completions function tools with strict=false.
  tool_choice / parallel_tool_calls are NEVER sent (API defaults apply).

Scenarios (mirror real call turns):
  A  non-tool turn      -> expect the JSON envelope as content, no tool call
  B  hangup turn        -> expect hangup_call_with_custom_delay to FIRE
                           (envelope alongside it is fine; envelope-only = the bug)
  C  fetch turn, step 1 -> expect fetch_seller_details to FIRE
     fetch turn, step 2 -> tool result appended as assistant tool_calls + tool message,
                           then a
                           second request -> expect the envelope

Usage:
  # Reference behavior on OpenAI GPT-4.1:
  OPENAI_API_KEY=sk-... python repro_simplismart_json_toolcall.py

  # Any OpenAI-compatible /v1/chat/completions endpoint:
  python repro_simplismart_json_toolcall.py --base-url https://<host>/v1 --api-key <token> --model <model>

  # Endpoint that requires a deployment/routing id header:
  python repro_simplismart_json_toolcall.py --base-url https://<host>/v1 --endpoint-id <id> --api-key <token> --model <model>

  # Tool-turn output is stochastic — repeat scenarios to see the distribution:
  python repro_simplismart_json_toolcall.py --runs 5

Requires: pip install openai  (v1.x)
"""

import argparse
import json
import os
import sys

from openai import OpenAI

# --------------------------------------------------------------------------------------
# The JSON envelope schema — byte-identical to what production sends as text.format.
# Non-strict on purpose: memory keys are hints, the model outputs only keys with data.
# --------------------------------------------------------------------------------------

MEMORY_KEYS = ["language", "buyer_interested", "seller_details_shared", "callback_time"]
STAGE_NAMES = ["STAGE_INTRO", "STAGE_SELLER_DETAILS", "STAGE_CLOSING"]

ENVELOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "assistant_reply": {"type": "string"},
        "memory": {
            "type": ["object", "null"],
            "properties": {key: {"type": "string"} for key in MEMORY_KEYS},
        },
        "next_stage": {"type": ["string", "null"], "enum": [*STAGE_NAMES, None]},
    },
    "required": ["assistant_reply", "memory", "next_stage"],
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "call_response",
        "schema": ENVELOPE_SCHEMA,
        "strict": False,
    },
}

# --------------------------------------------------------------------------------------
# Tools — real production tool schemas, in the flat Responses API format production sends.
# fetch_seller_details: a 2-LLM-call tool (tool result feeds a follow-up request).
# hangup_call_with_custom_delay: a 1-LLM-call tool (no follow-up request in production).
# --------------------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "name": "fetch_seller_details",
        "description": "Fetch seller details for the available seller",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "strict": False,
    },
    {
        "type": "function",
        "name": "hangup_call_with_custom_delay",
        "description": "End the call with a custom delay. The delay is capped at 12 seconds.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The hang up message"},
                "delay": {
                    "type": "integer",
                    "description": "Delay in seconds before ending the call. Must be between 1 and 12.",
                },
            },
            "required": ["delay"],
        },
        "strict": False,
    },
]

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
            "strict": tool["strict"],
        },
    }
    for tool in TOOLS
]

# --------------------------------------------------------------------------------------
# System prompt — compact stand-in for a production campaign prompt (~15K tokens in prod).
# The envelope instructions below mirror how production prompts teach the format.
# --------------------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Meera, a SquadStack voice agent calling a buyer on behalf of IndiaMART.
You speak Hinglish (Hindi in Devanagari script with common English words). Replies must be
short (1-2 sentences), natural and phone-friendly.

# TOOLS
- Call fetch_seller_details when the buyer asks about the seller/supplier.
- Call hangup_call_with_custom_delay when the conversation is over (buyer says goodbye,
  is not interested, or asks to stop). Pass a short farewell as `message` and delay=6.

# RESPONSE FORMAT — every turn
Always respond with a single JSON object:
{"assistant_reply": "<what you say to the buyer>", "memory": {<only keys that changed>}, "next_stage": "<stage or null>"}
- memory keys you may set: language, buyer_interested, seller_details_shared, callback_time
- stages: STAGE_INTRO, STAGE_SELLER_DETAILS, STAGE_CLOSING
- assistant_reply is spoken aloud to the buyer over the phone.
"""

GREETING = (
    '{"assistant_reply": "नमस्ते! मैं मीरा बोल रही हूँ IndiaMART की तरफ से। '
    'आपने हाल में एक enquiry डाली थी, उसी के बारे में बात करनी थी।", '
    '"memory": {"language": "hindi"}, "next_stage": "STAGE_INTRO"}'
)

SCENARIOS = {
    "A": {
        "title": "non-tool turn -> envelope only",
        "user": "haan boliye, kya enquiry thi meri?",
        "expect": "envelope content, NO tool call",
    },
    "B": {
        "title": "hangup turn (1 LLM call in production)",
        "user": "theek hai thank you, mujhe abhi baat nahi karni, rakhti hoon. bye.",
        "expect": "hangup_call_with_custom_delay MUST fire (envelope alongside is fine)",
    },
    "C": {
        "title": "fetch turn (2 LLM calls in production)",
        "user": "accha pehle ये batao seller kaun hai? unke baare mein details do.",
        "expect": "call 1: fetch_seller_details fires; call 2 (after tool result): envelope",
    },
}

# What the tool returns in production (result_callback payload shape).
FETCH_SELLER_RESULT = {
    "status": "success",
    "message": "Available seller details fetched successfully",
    "response": {
        "seller_name": "Sharma Industrial Supplies",
        "location": "Ludhiana, Punjab",
        "products": ["hydraulic pumps", "industrial valves"],
        "rating": "4.2",
    },
}


def conversation_input(user_text):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": GREETING},
        {"role": "user", "content": user_text},
    ]


def tool_roundtrip_input(input_items, tool_call, result):
    """Append Chat Completions tool-call history and its tool result."""
    return input_items + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call["id"],
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": tool_call["arguments"] or "{}",
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": json.dumps(result),
        },
    ]


# --------------------------------------------------------------------------------------
# Responses API caller — request dict mirrors production's request builder. Streaming
# consumption mirrors production: text deltas and function_call items accumulate
# independently from the SAME stream, so a response carrying both is fully consumed.
# --------------------------------------------------------------------------------------


def run_chat_completions(client, model, input_items, args):
    params = {
        "model": model,
        "messages": input_items,
        "tools": CHAT_TOOLS,
        "temperature": 0.7,
        "max_tokens": args.max_tokens,
        "response_format": RESPONSE_FORMAT,
        "stream": not args.no_stream,
    }

    if args.no_stream:
        resp = client.chat.completions.create(**params)
        message = resp.choices[0].message
        content = message.content or ""
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in (message.tool_calls or [])
        ]
        return content, tool_calls, resp.usage

    content, usage = "", None
    tool_calls = {}
    stream = client.chat.completions.create(**params)
    for chunk in stream:
        if not chunk.choices:
            usage = chunk.usage or usage
            continue
        delta = chunk.choices[0].delta
        content += delta.content or ""
        for tc in delta.tool_calls or []:
            current = tool_calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                current["id"] = tc.id
            if tc.function:
                current["name"] += tc.function.name or ""
                current["arguments"] += tc.function.arguments or ""
        usage = chunk.usage or usage
    return content, [tool_calls[i] for i in sorted(tool_calls)], usage


# --------------------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------------------


def parse_envelope(content):
    if not content or not content.strip():
        return None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and {"assistant_reply", "memory", "next_stage"} <= set(data):
        return data
    return None


def describe(content, tool_calls, usage):
    envelope = parse_envelope(content)
    if content:
        kind = "valid envelope" if envelope else "NON-CONFORMING content"
        print(f"  content ({kind}): {content[:400]}")
    else:
        print("  content: <none>")
    for tc in tool_calls:
        print(f"  tool_call: {tc['name']}({tc['arguments']})  id={tc['id']}")
    if not tool_calls:
        print("  tool_call: <none>")
    if usage:
        print(f"  usage: input={usage.prompt_tokens} output={usage.completion_tokens}")
    return envelope


def verdict(ok, label):
    print(f"  => {'PASS' if ok else 'FAIL'}: {label}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemma4")
    ap.add_argument(
        "--base-url",
        default="https://http.pyo5ngrfem.ss-in.s9t.link/v1",
        help="OpenAI-compatible base URL",
    )
    ap.add_argument("--api-key", default=None, help="defaults to $OPENAI_API_KEY")
    ap.add_argument(
        "--endpoint-id",
        default="adbb461d-c516-48fb-b35b-450d73f716ab",
        help="send this value as the custom 'id' routing header",
    )
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="production uses up to 1000 on json_call_memory campaigns")
    ap.add_argument("--runs", type=int, default=1, help="repeat all scenarios N times (tool-turn output is stochastic)")
    ap.add_argument("--no-stream", action="store_true", help="production always streams; non-stream for debugging")
    ap.add_argument("--scenario", choices=["A", "B", "C"], default=None, help="run only one scenario")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("No API key: pass --api-key or set OPENAI_API_KEY")
    default_headers = {"id": args.endpoint_id} if args.endpoint_id else None
    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        default_headers=default_headers,
    )
    print(f"Target: {args.base_url}/chat/completions")
    print(f"Model: {args.model}")
    print(f"Routing id header: {args.endpoint_id or '<not set>'}")

    results = []
    scenarios = [args.scenario] if args.scenario else ["A", "B", "C"]
    for run in range(1, args.runs + 1):
        for name in scenarios:
            sc = SCENARIOS[name]
            print(f"\n=== run {run} | scenario {name}: {sc['title']} ===")
            print(f"  user: {sc['user']}")
            print(f"  expect: {sc['expect']}")

            input_items = conversation_input(sc["user"])
            content, tool_calls, usage = run_chat_completions(client, args.model, input_items, args)
            envelope = describe(content, tool_calls, usage)

            if name == "A":
                ok = verdict(envelope is not None and not tool_calls, "envelope only, no tool call")
            elif name == "B":
                fired = any(tc["name"] == "hangup_call_with_custom_delay" for tc in tool_calls)
                shape = "dual output (envelope + tool)" if (fired and envelope) else \
                        ("tool call only" if fired else "envelope only — THE BUG")
                ok = verdict(fired, f"hangup tool fired [{shape}]")
            else:  # C
                fetch = next((tc for tc in tool_calls if tc["name"] == "fetch_seller_details"), None)
                ok = verdict(fetch is not None, "call 1: fetch_seller_details fired")
                if fetch:
                    print("  --- appending tool result exactly like production, follow-up request ---")
                    input2 = tool_roundtrip_input(input_items, fetch, FETCH_SELLER_RESULT)
                    content2, tool_calls2, usage2 = run_chat_completions(client, args.model, input2, args)
                    envelope2 = describe(content2, tool_calls2, usage2)
                    ok = verdict(envelope2 is not None, "call 2: envelope after tool result") and ok
            results.append((run, name, ok))

    print("\n=== summary ===")
    for run, name, ok in results:
        print(f"  run {run} scenario {name}: {'PASS' if ok else 'FAIL'}")
    failed = [r for r in results if not r[2]]
    print(f"  {len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
