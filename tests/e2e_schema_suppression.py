"""Decisive e2e: a tool the model provably calls, with and without a json_schema response_format.

The model calls get_weather 3/3 with no response_format (verified). So if adding a json_schema
response_format suppresses the tool call, that is the bug — and if the tool still fires with the
schema in place, the fix works. Same server, same model, same prompt; only response_format varies.
"""
import json
import sys

from openai import OpenAI

client = OpenAI(api_key="dummy", base_url="http://127.0.0.1:8077/v1")
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 5

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "strict": False,
        },
    }
]

ENVELOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "assistant_reply": {"type": "string"},
        "memory": {"type": ["object", "null"], "properties": {"language": {"type": "string"}}},
        "next_stage": {"type": ["string", "null"], "enum": ["STAGE_INTRO", "STAGE_CLOSING", None]},
    },
    "required": ["assistant_reply", "memory", "next_stage"],
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "call_response", "schema": ENVELOPE_SCHEMA, "strict": False},
}

MESSAGES = [{"role": "user", "content": "What's the weather in Tokyo right now? Use the tool."}]


def call(with_schema, stream):
    kw = dict(model="gemma4", messages=MESSAGES, tools=TOOLS, max_tokens=300, temperature=0.7)
    if with_schema:
        kw["response_format"] = RESPONSE_FORMAT
    if not stream:
        m = client.chat.completions.create(**kw).choices[0].message
        return (m.content or ""), [
            {"name": t.function.name, "arguments": t.function.arguments} for t in (m.tool_calls or [])
        ]
    content, tcs = "", {}
    for chunk in client.chat.completions.create(stream=True, **kw):
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


def envelope_ok(content):
    try:
        d = json.loads(content)
    except Exception:
        return False
    return isinstance(d, dict) and {"assistant_reply", "memory", "next_stage"} <= set(d)


for stream in (True, False):
    for with_schema in (False, True):
        label = f"{'stream ' if stream else 'nostream'} | {'json_schema' if with_schema else 'no schema  '}"
        fired = envelopes = errors = 0
        samples = []
        for _ in range(RUNS):
            try:
                content, tool_calls = call(with_schema, stream)
            except Exception as e:
                errors += 1
                samples.append(f"ERROR {type(e).__name__}: {str(e)[:120]}")
                continue
            fired += bool(tool_calls)
            envelopes += envelope_ok(content)
            samples.append(
                (f"tool={tool_calls[0]['name']}({tool_calls[0]['arguments']})" if tool_calls else "NO TOOL")
                + (f" content={content[:70]!r}" if content else "")
            )
        print(f"{label}  tool {fired}/{RUNS}  envelope-parses {envelopes}/{RUNS}  errors {errors}")
        for s in samples[:3]:
            print(f"      {s}")
