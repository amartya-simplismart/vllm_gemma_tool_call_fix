# Testing guide: structured output (json_schema) + tools, the way SquadStack production uses them

Companion to `repro_simplismart_json_toolcall.py`, which replicates the exact request shape
SquadStack production sends for GPT-4.1 and asserts the expected behavior per turn type.

## The production request shape

SquadStack's OpenAI traffic uses the Responses API exclusively. `POST /v1/responses`:

```json
{
  "model": "gpt-4.1",
  "stream": true,
  "store": false,
  "truncation": "auto",
  "instructions": "<system prompt, ~15K tokens in production>",
  "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "..."}]}],
  "tools": [{"type": "function", "name": "...", "description": "...", "parameters": {...}, "strict": false}],
  "temperature": 0.7,
  "max_output_tokens": 400,
  "text": {
    "format": {
      "type": "json_schema",
      "name": "call_response",
      "strict": false,
      "schema": {
        "type": "object",
        "properties": {
          "assistant_reply": {"type": "string"},
          "memory": {"type": ["object", "null"], "properties": {"<memory_key>": {"type": "string"}}},
          "next_stage": {"type": ["string", "null"], "enum": ["<stage>", null]}
        },
        "required": ["assistant_reply", "memory", "next_stage"]
      }
    }
  }
}
```

Notes: **non-strict** schema (memory keys are hints — the model outputs only keys with
data). `stream` is always true in production. `tool_choice` and `parallel_tool_calls` are
**never sent** (API defaults apply).

## What GPT-4.1 actually does with these requests (measured)

| Turn type | GPT-4.1 result |
|---|---|
| Non-tool turn | one `message` with schema-conforming JSON envelope, no tool call |
| Tool turn (small prompt, N=12) | `function_call` **only**, no envelope — 12/12 |
| Tool turn (production, ~15K-token prompts) | usually `function_call` only; **sometimes BOTH** the envelope message **and** the `function_call` in one response |

The "sometimes BOTH" is real: on 2026-07-24 a production GPT-4.1 response (108 completion
tokens) carried the full envelope **and** a `hangup_call` function_call in the same streamed
response; 59 distinct calls hit our dual-output handling in ~36h. But it is **stochastic** —
the same campaign also produced tool-call-only turns, and with minimal prompts we could not
elicit dual output at all.

## What the SquadStack runtime requires

Our pipeline consumes the stream with two independent accumulators — text deltas (parsed
token-by-token as the envelope) and function-call items — so **every one of these response
shapes works today**:

1. envelope message only (normal turn)
2. tool call only (tool turn — the envelope then comes from the follow-up request, see below)
3. envelope **and** tool call in one response (must not crash / drop either channel)

**The bug being fixed:** on gemma with structured output enforced, shape 2 never happens —
the constrained decoder always emits the envelope and silently drops the tool call, so tools
never fire. **Per-turn dual output (shape 3) is NOT a hard requirement** — if the model
emits a tool call *instead of* the envelope on tool turns, the runtime handles it exactly as
it already handles GPT-4.1's tool-only turns. If the model emits both, both must come
through intact.

## Two-call tools: the follow-up request

Important context: the model never "makes calls" — our client runtime (a pipecat-based
pipeline) orchestrates every request. When a response contains a `function_call`, the
runtime executes the actual tool and, for tools whose result feeds the dialogue, issues the
follow-up request itself. So "two LLM calls" is deterministic **client-side** behavior — not
something the model or the API decides:

1. Request 1 (schema + tools) → model returns the `fetch_seller_details` tool call.
2. The runtime executes the tool and appends to the request `input`:
   - `{"type": "function_call", "name": "...", "arguments": "{...}", "call_id": "..."}`
   - `{"type": "function_call_output", "call_id": "...", "output": "<json result>"}`
3. Request 2 — **same schema, same tools**, sent by the runtime → model must return a
   schema-conforming envelope that uses the tool result.

For tools like `hangup_call_with_custom_delay` the runtime hangs up and never issues a
follow-up — one request total.

The repro script re-implements this exact orchestration in plain code
(`tool_roundtrip_input()` + a second `run_responses()` call), so the flow can be tested
without any of our runtime. The model's only responsibilities are: emit the `function_call`
in response 1, and emit the envelope in response 2.

## Acceptance criteria (what the repro script asserts)

| Scenario | Request | PASS |
|---|---|---|
| A: non-tool turn | schema + tools | valid envelope content, no tool call |
| B: hangup turn | schema + tools | `hangup_call_with_custom_delay` fires (with or without envelope) |
| C1: fetch turn | schema + tools | `fetch_seller_details` fires |
| C2: fetch follow-up | schema + tools + tool-call history | valid envelope content |

Plus: streamed and non-streamed behavior must match, and if the model emits text and a tool
call in one response, both must survive to the client.

## Running

```bash
pip install openai

# Reference behavior on OpenAI GPT-4.1:
OPENAI_API_KEY=... python repro_simplismart_json_toolcall.py

# Any OpenAI-compatible /v1/responses endpoint:
python repro_simplismart_json_toolcall.py --base-url https://<host>/v1 --api-key <token> --model <model>

# Repeat runs (model behavior on tool turns is stochastic):
python repro_simplismart_json_toolcall.py --runs 5
# Non-streamed variant for debugging (production always streams):
python repro_simplismart_json_toolcall.py --no-stream
```

Verified 2026-07-25 against OpenAI `gpt-4.1`: all scenarios pass.
