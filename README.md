# vLLM gemma4: structured output (`json_schema`) + tool calling

Fix for a defect in vLLM 0.26's Gemma 4 path: when a request carries **both** a
`json_schema` response format and `tools`, the tool call is destroyed. Verified end to end on
`google/gemma-4-31B-it`.

## The defect

With `tool_choice` unset (defaults to `"auto"`), vLLM leaves the caller's JSON schema in place
and compiles a grammar whose only accepting strings are the schema object. Raw model output under
each constraint, same prompt, captured from `/v1/completions` with `skip_special_tokens=false`:

```
1  unconstrained
   <|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>            <- clean native call

2  json_schema  (the bug)
   <|tool_call>{"assistant_reply": "<|tool_call>call:get_weather{city:\"Tokyo\"}<tool_call|>"
   \n  \n  \n  \n  \n  …                                                     <- runs to max_tokens

3  union structural tag  (this fix)
   <|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>            <- identical to (1)
```

The `<|tool_call>` special token *is* emitted — special tokens are not masked by the grammar. But
the grammar then demands the JSON object, so the tool call is **coerced into the JSON string as
escaped text**, the object never terminates, and the request burns its entire token budget. The
client receives a malformed envelope and no usable tool call.

This is not a tool-parser problem: `--tool-call-parser gemma4` only ever sees text the grammar
already permitted.

## The fix

Translate `json_schema` + `tools` into a single xgrammar **structural tag** that accepts the union
of the legitimate output shapes, instead of a bare JSON grammar:

```
[ <|channel>thought … <channel|> ]     # optional thinking channel
(
    schema_json  [ tool_calls ]        # text-only, and text + tool call
  | tool_calls                         # tool call only
)
```

The schema keeps its hard guarantee on speaking turns; tool calls keep Gemma 4's native argument
dialect (`key:<|"|>value<|"|>`); dual output is permitted but never forced.

Tool *arguments* are intentionally left unconstrained inside the tag. xgrammar ships a `gemma_4`
structural-tag template but leaves it unregistered
(`xgrammar/builtin_structural_tag.py`): *"we are dropping Gemma support because its parameter
format is special and not supported yet: the string are wrapped by `<|"|>` instead of `"`"*.
Constraining arguments as JSON would push the model off its trained format, and for
`strict: false` tools no argument-level enforcement is expected anyway.

## What's in here

| Path | Role |
|---|---|
| `server_fix/vllm/tool_parsers/gemma4_structural_tag.py` | builds the union tag; `python <file>` self-tests it against xgrammar (11/11) |
| `server_fix/vllm/tool_parsers/gemma4_engine_tool_parser.py` | drop-in replacement for the shipped parser; picks the constraint per request |
| `server_fix/vllm/parser/gemma4_turn_end.patch` | separate fix: `<turn|>` leaking into streamed content |
| `server_fix/test_gemma4_adjust_request.py` | request-shaping + grammar test (no GPU needed) |
| `tests/probe_schema_vs_toolcall.py` | controlled 4-way probe: schema × tool_choice |
| `tests/e2e_schema_suppression.py` | does adding a schema suppress a tool the model provably calls? |
| `tests/raw_output_under_constraints.py` | raw generation under each constraint, via `/v1/completions` |
| `repro_simplismart_json_toolcall.py` | client-side acceptance check (3 scenarios, PASS/FAIL) |
| `dynamo_fix/` | Rust patches for NVIDIA Dynamo (`dynamo-parsers` + `dynamo-llm`) with build steps and results |
| `FIX_gemma4_json_plus_tools.md` | full write-up: root cause, patch, deployment, dynamo notes |

## Install

```bash
V=$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
cp $V/tool_parsers/gemma4_engine_tool_parser.py $V/tool_parsers/gemma4_engine_tool_parser.py.orig
cp server_fix/vllm/tool_parsers/gemma4_structural_tag.py      $V/tool_parsers/
cp server_fix/vllm/tool_parsers/gemma4_engine_tool_parser.py  $V/tool_parsers/
patch -p2 -d $V < server_fix/vllm/parser/gemma4_turn_end.patch

python3 server_fix/test_gemma4_adjust_request.py
```

`--tool-call-parser gemma4` resolves through vLLM's lazy registry to `Gemma4EngineToolParser`, so
no launch-flag changes are needed for `vllm serve`. **Under NVIDIA Dynamo this is not enough** —
see the dynamo section below.

## Verified on `gemma-4-31B-it`

2×H200, `--tensor-parallel-size 2 --max-model-len 64000 --max-num-seqs 8
--gpu-memory-utilization 0.95`, speculative decoding with the `gemma-4-31B-it-assistant` MTP
draft (`num_speculative_tokens 4`), `--tool-call-parser gemma4 --reasoning-parser gemma4`,
streaming. Only the parser file differs between before/after.

Controlled probe, 3 runs per cell. The `no schema` column is the control: it proves the model is
willing to call the tool, so the difference is attributable to the schema alone.

| | no `response_format` | `json_schema` |
|---|---|---|
| `tool_choice: auto` | 3/3 → 3/3 | **0/3 → 3/3** |
| `tool_choice: required` | 3/3 → 3/3 | **0/3 → 3/3** |

Acceptance script: **1/3 → 9/9** over 3 streamed runs, and 3/3 non-streamed. The two-call flow
works: request 1 returns `fetch_seller_details({})`, request 2 (with the tool result appended)
returns a schema-conforming object that uses it. Arguments arrive clean and correctly typed.

Notes:

- The `required` + `json_schema` row is a second defect the same fix covers: the shipped parser
  returns early for forced tool choice, leaving the caller's schema in place, so even a
  *demanded* tool call fired 0/3.
- The shipped parser's docstring warns that guided decoding "crashes EngineCore under speculative
  decoding". That did **not** reproduce: 9/9 clean with the MTP draft loaded. Not yet load-tested
  at production concurrency.
- `google/gemma-4-31B-it-assistant` is the MTP **draft** model — it belongs in
  `--speculative-config.model`, not `--model`.

## NVIDIA Dynamo: fixed separately in Rust (`dynamo_fix/`)

Tested with `ai-dynamo` 1.3.0.post1 (etcd + NATS, `dynamo.vllm` worker, `dynamo.frontend`),
gemma-4-31B-it, patch installed:

- **Default Rust frontend** — `--dyn-tool-call-parser` selects a *Rust* parser
  (`dynamo-parsers/src/tool_calling/gemma4/parser.rs`), the Rust ingress builds
  `guided_decoding` itself, and vLLM's Python `adjust_request` is never called. Result: **1/3,
  identical to unpatched.** `--dyn-enable-structural-tag --dyn-structural-tag-scope always` does
  not help (0/2 with a schema), and a client-supplied `response_format: {"type":"structural_tag"}`
  is rejected with HTTP 400 `unknown variant`. There is currently no configuration that makes it
  work — the union tag has to be emitted by dynamo's Rust constraint selection, which already
  plumbs `guided_decoding.structural_tag` to the worker.
- **`--dyn-chat-processor vllm`** (Python processor) — `prepost.py` *does* call
  `adjust_request()` on the class from `ToolParserManager`, so the patch runs. Non-streaming
  **passes** with clean arguments. Streaming generates the native tool call (the grammar fix
  works) but dynamo delivers it as *content* rather than `tool_calls`, because vLLM 0.26's gemma4
  parser is engine-based while `prepost.py` drives reasoning and tool parsing separately.

**Both gaps are now fixed in Rust — see `dynamo_fix/`.** `dynamo-parsers` gets a real structural
tag builder for gemma4 (it had none, which is why `--dyn-enable-structural-tag` did nothing) plus
the union composition; `dynamo-llm` stops bailing out of constraint selection when a
`response_format` is present alongside tools. Verified through `dynamo.frontend` + `dynamo.vllm`
with no extra launch flags: the two tool scenarios went **0/4 → 4/4** and the follow-up envelope
**4/4**. Details, build steps, and how to fix the remaining non-tool-turn behaviour (per-turn
`tool_choice: "none"` — 3/3 — or a stricter prompt — 3/3) are in `dynamo_fix/README.md`.

`vllm serve` remains verified end to end at 9/9.

## Third fix: `<turn|>` leaks into streamed content

`TURN_END` (`<turn|>`) is not one of the Gemma 4 parser's terminals, and the parser engine sets
`skip_special_tokens=False` so `<|tool_call>` survives detokenization. On the **streaming** path
the turn terminator therefore arrives as a content delta, so a streamed JSON object ends
`…"next_stage": "STAGE_INTRO"}<turn|>` and `json.loads()` fails. Non-streaming is unaffected, and
a client-side `stop: ["<turn|>"]` does not suppress it. Independent of the grammar fix; present
before and after.
