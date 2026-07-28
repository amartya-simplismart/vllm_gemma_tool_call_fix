"""Isolate model choice from grammar reachability.

A: no response_format, tool_choice auto      -> control: does this model call the tool at all?
B: json_schema, tool_choice auto             -> the production shape
C: json_schema, tool_choice required         -> is a tool call REACHABLE through the grammar?
D: no response_format, tool_choice required   -> control for C
"""
import importlib.util, json, sys

spec = importlib.util.spec_from_file_location("r", "/home/key/Amartya/squadstack/repro_simplismart_json_toolcall.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

from openai import OpenAI

client = OpenAI(api_key="dummy", base_url="http://127.0.0.1:8078/v1")
USER = "theek hai thank you, mujhe abhi baat nahi karni, rakhti hoon. bye."
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def call(response_format, tool_choice):
    kw = dict(
        model="gemma4",
        messages=r.conversation_input(USER),
        tools=r.CHAT_TOOLS,
        temperature=0.7,
        max_tokens=400,
        stream=True,
    )
    if response_format:
        kw["response_format"] = r.RESPONSE_FORMAT
    if tool_choice:
        kw["tool_choice"] = tool_choice
    content, tcs = "", {}
    for chunk in client.chat.completions.create(**kw):
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        content += d.content or ""
        for tc in d.tool_calls or []:
            cur = tcs.setdefault(tc.index, {"name": "", "arguments": ""})
            if tc.function:
                cur["name"] += tc.function.name or ""
                cur["arguments"] += tc.function.arguments or ""
    return content, [tcs[i] for i in sorted(tcs)]


CASES = [
    ("A  no schema   + auto    ", False, None),
    ("B  json_schema + auto    ", True, None),
    ("C  json_schema + required", True, "required"),
    ("D  no schema   + required", False, "required"),
]

for label, rf, tc in CASES:
    fired = 0
    detail = []
    for _ in range(RUNS):
        try:
            content, tool_calls = call(rf, tc)
        except Exception as e:
            detail.append(f"ERROR {type(e).__name__}: {str(e)[:150]}")
            continue
        fired += bool(tool_calls)
        detail.append(
            (f"tool={tool_calls[0]['name']}({tool_calls[0]['arguments']})" if tool_calls else "no tool")
            + f" | content={content[:90]!r}"
        )
    print(f"{label}  tool fired {fired}/{RUNS}")
    for d in detail:
        print(f"      {d}")
