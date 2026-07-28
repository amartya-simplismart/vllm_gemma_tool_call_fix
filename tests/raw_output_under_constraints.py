"""Raw model output under each constraint, via /v1/completions (no tool parser in the way).

Renders the chat prompt exactly as the chat endpoint would, then generates with:
  1. no structured output
  2. the json_schema envelope constraint  (what production sends today)
  3. the union structural tag             (the fix)
and prints the undecorated text so we can see what the grammar actually permits.
"""
import json
import sys

import requests
from transformers import AutoTokenizer

sys.path.insert(0, "/home/key/Amartya/squadstack/server_fix/vllm/tool_parsers")
from gemma4_structural_tag import build_envelope_plus_tools_tag

M = "/home/key/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7"
URL = "http://127.0.0.1:8077/v1/completions"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
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

tok = AutoTokenizer.from_pretrained(M)
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "What's the weather in Tokyo right now? Use the tool."}],
    tools=TOOLS,
    tokenize=False,
    add_generation_prompt=True,
)

tag = build_envelope_plus_tools_tag(ENVELOPE_SCHEMA, ["get_weather"], reasoning=True)

CASES = [
    ("1  unconstrained", None),
    ("2  json_schema (production today)",
     {"type": "json_schema", "json_schema": {"name": "call_response", "schema": ENVELOPE_SCHEMA, "strict": False}}),
    ("3  union structural tag (the fix)", tag.model_dump(by_alias=True, exclude_none=True)),
]

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 2

for label, rf in CASES:
    print(f"\n=== {label} ===")
    for i in range(RUNS):
        body = {
            "model": "gemma4",
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.7,
            "skip_special_tokens": False,
        }
        if rf:
            body["response_format"] = rf
        r = requests.post(URL, json=body, timeout=180)
        if r.status_code != 200:
            print(f"  run {i+1}: HTTP {r.status_code} {r.text[:200]}")
            continue
        text = r.json()["choices"][0]["text"]
        print(f"  run {i+1} RAW: {text[:260]!r}")
