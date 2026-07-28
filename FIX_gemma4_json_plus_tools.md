# Server-side fix: gemma4 must be able to emit a tool call while `response_format` is a `json_schema`

The client contract does not change. SquadStack keeps sending exactly what they send to GPT-4.1
today — `text.format` / `response_format` of type `json_schema`, flat `tools` with
`strict: false`, no `tool_choice`. All of the change is in the serving stack.
`repro_simplismart_json_toolcall.py` is untouched; it is the acceptance check.

## Root cause — the envelope grammar swallows the tool call

With `tool_choice` unset it defaults to `"auto"`
(`entrypoints/openai/chat_completion/protocol.py:845-848`). At `"auto"`,
`get_json_schema_from_tools()` returns `None`, so `ToolParser.adjust_request()` leaves the
caller's JSON schema in place (`tool_parsers/abstract_tool_parser.py:118-166`). The request then
becomes `StructuredOutputsParams(json=<envelope schema>)`
(`entrypoints/openai/engine/protocol.py:190-193`), and the structured-output backend compiles a
grammar whose only accepting strings are the envelope object.

What that does to a Gemma4 tool call, captured raw from `/v1/completions` on a live server
(`skip_special_tokens=false`, weather tool, same prompt in all three cases):

```
1  unconstrained
   '<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>'          <- clean native call

2  json_schema (what production sends today)
   '<|tool_call>{"assistant_reply": "<|tool_call>call:get_weather{city:\\"Tokyo\\"}<tool_call|>"
    \n  \n  \n  \n  \n  \n  \n  …'                                            <- runs to max_tokens

3  union structural tag (the fix)
   '<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>'          <- identical to (1)
```

Read case 2 carefully, because it corrects an earlier theory of mine. The special token
`<|tool_call>` **is** emitted — special tokens are not masked by the grammar. But the very next
step the grammar demands the envelope object, so the model is forced into
`{"assistant_reply": "` and its tool call gets **coerced into the JSON string as escaped text**
(`\"Tokyo\"` instead of `<|"|>Tokyo<|"|>`). The envelope then never terminates and the request
burns its whole token budget on `\n  ` filler.

So the tool call is not "blocked at the logit mask" — it is captured by the envelope and
destroyed. What reaches the client is a malformed, unterminated envelope and no usable tool
call, which is exactly SquadStack's description: *"the JSON always wins and the tool call gets
dropped."* Downstream, the parser either finds nothing or (with a tool-eager prompt) recovers a
tool call with mangled arguments — measured `{"city": "\\\"Tokyo\\\""}`.

GPT-4.1 differs because it applies no such grammar to the tool channel.

## The fix — union grammar via an xgrammar structural tag

vLLM already supports xgrammar structural tags as a structured-output mode
(`StructuredOutputsParams.structural_tag`, `sampling_params.py:83`), and a structural tag can
express alternation. So one grammar can cover every shape the SquadStack runtime handles:

```
[ <|channel>thought … <channel|> ]     # optional, only when a reasoning parser is configured
(
    envelope_json  [ tool_calls ]      # shape 1 (envelope only) + shape 3 (both)
  | tool_calls                         # shape 2 — destroyed today, this is the bug
)
```

Properties this preserves, matching what SquadStack said they cannot give up:

- the envelope keeps its **hard schema guarantee** on every turn where the model speaks —
  no best-effort prompt-based JSON;
- **one generation** decides reply, memory, stage *and* tool — no second request, no split
  reasoning;
- shape 3 is **permitted but never forced**, and shape 2 is now reachable — exactly the
  "one-channel-per-turn is fine, dual output must not break" contract;
- tool arguments stay in gemma4's **native dialect** (`key:<|"|>value<|"|>`).

### Why tool arguments are not schema-constrained inside the tag

xgrammar ships a gemma_4 structural-tag template but leaves it **deliberately unregistered**
(`xgrammar/builtin_structural_tag.py:1699-1701`):

> `# TODO: We are dropping Gemma support because its parameter format is special and not
> supported yet: the string are wrapped by <|"|> instead of ".`

`JSONSchemaFormat` has no gemma4 style, so constraining arguments would force plain-JSON quoting
and push the model off its trained tool-call format. The tag constrains the wrapper and the tool
**name**, and leaves the argument body free (`AnyTextFormat`) terminated by `<tool_call|>`.
Since every SquadStack tool is `strict: false`, argument-level enforcement is not expected
anyway — vLLM's own `_get_function_parameters()` returns `True` (any value) for `strict: false`.

## Files

| File | Role |
|---|---|
| `server_fix/vllm/tool_parsers/gemma4_structural_tag.py` | builds the union tag; `python <file>` self-tests it against xgrammar (11/11) |
| `server_fix/vllm/tool_parsers/gemma4_engine_tool_parser.py` | drop-in replacement for the shipped file; decides the constraint per request |
| `server_fix/vllm/parser/gemma4_turn_end.patch` | **second fix** — stops `<turn|>` leaking into streamed content (see below) |
| `server_fix/test_gemma4_adjust_request.py` | request-shaping + grammar test, no GPU/model/server needed |

### Second bug found during the e2e: `<turn|>` leaks into streamed content

`TURN_END` (`<turn|>`) is not one of the Gemma4 parser's terminals, and the parser engine sets
`skip_special_tokens=False` so that `<|tool_call>` survives detokenization. Result: on the
**streaming** path the turn terminator arrives as a content delta, so the streamed envelope ends
`…"next_stage": "STAGE_INTRO"}<turn|>` and `json.loads()` fails. Non-streaming is unaffected, and
a client-side `stop: ["<turn|>"]` does **not** suppress it (it is a special token, not matched as
a stop string) — measured both ways.

This matters because SquadStack always streams and parses that envelope token-by-token to drive
TTS. It is present with **and** without the union-grammar fix — an independent, pre-existing bug.
The patch strips `TURN_END` from content deltas in `Gemma4Parser._events_to_delta`.

### What `adjust_request` now does

| `tool_choice` | JSON schema on the request | constraint applied |
|---|---|---|
| required / named | no | none — unchanged early return, native syntax + `skip_special_tokens=False` |
| required / named | **yes** | union grammar **requiring** a tool call (named → only that tool) |
| auto / unset | yes | union grammar — envelope, tool call, or both |
| auto / unset | no | unchanged (`super().adjust_request`) |
| any | caller sent its own `structural_tag` | left untouched |
| no tools | yes | unchanged — plain `json_schema` |

The required/named + schema row is a **third bug this fixes**, and it is measured, not theorised:
the shipped code returns early there, leaving the caller's `response_format` in place, so the
envelope grammar still swallows the call and `tool_choice: "required"` fired **0/3** on
gemma-4-31B-it. This is likely what the shipped docstring records as "leaks as content"
(`gemma4_engine_tool_parser.py:20-27`) — the model is forced to produce one thing while the
parser waits for another. (The same docstring's EngineCore crash under speculative decoding did
**not** reproduce in the e2e; see the verification section.)

## Deploy

`--dyn-tool-call-parser gemma4` resolves through vLLM's lazy registry to
`vllm.tool_parsers.gemma4_engine_tool_parser:Gemma4EngineToolParser`
(`tool_parsers/__init__.py:193-196`), so dropping both files into the vLLM install inside the
serving image is enough — no dynamo change, no new launch flags:

```bash
V=$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
cp $V/tool_parsers/gemma4_engine_tool_parser.py $V/tool_parsers/gemma4_engine_tool_parser.py.orig
cp server_fix/vllm/tool_parsers/gemma4_structural_tag.py      $V/tool_parsers/
cp server_fix/vllm/tool_parsers/gemma4_engine_tool_parser.py  $V/tool_parsers/

# second fix: stop <turn|> leaking into streamed content
patch -p2 -d $V < server_fix/vllm/parser/gemma4_turn_end.patch

python3 server_fix/test_gemma4_adjust_request.py    # grammar layer, no GPU needed
```

Then restart the deployment with the same command line and run the client acceptance check
against it:

```bash
python repro_simplismart_json_toolcall.py --base-url https://<host>/v1 --api-key <token> \
    --model gemma4 --runs 5
```

Expected: A envelope-only, B `hangup_call_with_custom_delay` fires, C1 `fetch_seller_details`
fires, C2 envelope after the tool result — 3/3 per run, matching gpt-4.1.

## End-to-end verification on `gemma-4-31B-it`

Run on 2×H200 with the production configuration: `--tensor-parallel-size 2 --max-model-len
64000 --max-num-seqs 8 --gpu-memory-utilization 0.95`, speculative decoding on with the
`gemma-4-31B-it-assistant` MTP draft at `num_speculative_tokens 4`, `--tool-call-parser gemma4
--reasoning-parser gemma4`, streaming. Plain `vllm serve` (dynamo is not installed on the test
box — see the dynamo section). Same server, same prompts; only the parser file changed between
columns.

**Controlled 4-way probe** (hangup prompt, 3 runs per cell — the `no schema` column is the
control proving the model itself is willing to call the tool):

| | no `response_format` | `json_schema` |
|---|---|---|
| `tool_choice: auto` (production) | 3/3 before · 3/3 after | **0/3 before · 3/3 after** |
| `tool_choice: required` | 3/3 before · 3/3 after | **0/3 before · 3/3 after** |

The control is what makes this conclusive: the model calls the tool 3/3 without a schema, and
0/3 with one, on the same server and prompt. The `required` + `json_schema` row is the second bug
described above — a *demanded* tool call could not fire either. Both are 3/3 after the fix.

**Acceptance matrix** — your unmodified `repro_simplismart_json_toolcall.py`:

| | before | after |
|---|---|---|
| streamed, 1 run | 1/3 (A only) | 3/3 |
| streamed, 3 runs | — | **9/9** |
| non-streamed, 1 run | — | 3/3 |

Scenario C's full two-call flow works: request 1 returns `fetch_seller_details({})`, the runtime
appends the tool result, request 2 returns a schema-conforming envelope that uses it —
*"seller का नाम Sharma Industrial Supplies है, जो Ludhiana, Punjab से…"* with
`seller_details_shared: true`. Tool arguments come through clean and correctly typed:
`{"delay": 6, "message": "ठीक है, कोई बात नहीं। आपका दिन शुभ हो, बाय!"}`.

Also measured on the way:

- **Speculative decoding did not crash.** I flagged the shipped docstring's "crashes EngineCore
  under speculative decoding" as the top rollout risk; with the MTP draft loaded and the union
  grammar active, all 9/9 runs completed clean. Caveat: `--max-num-seqs 8` with a short prompt,
  not production concurrency with ~15K-token campaign prompts — worth one load test, but this is
  no longer an open blocker.
- **Streamed and non-streamed agree** (3/3 each).
- **The `<turn|>` leak** was found here, not in the grammar tests — see above.
- Earlier runs on `gemma-4-E2B-it` (also verified) are not a substitute: E2B declines these tools
  even with no schema at all (0/3 control), so it can only validate plumbing, not tool firing.

Still-open items: grammar compile cost at production concurrency (the tag is compiled per
distinct schema + tool-set pair, so the first request of each campaign shape pays it), and
whether dynamo calls `adjust_request` at all — next section.

## Target config: gemma-4-31B-it + the `-assistant` MTP draft

`google/gemma-4-31B-it-assistant` is a **multi-token-prediction draft model for speculative
decoding**, not a servable target — it goes in `--speculative-config.model`, with the 31B IT
model as `--model`. Mapping that onto the current launch line (`/workspace/gemma4` = target,
`/workspace/draft_extra` = draft):

```bash
python3 -m dynamo.vllm \
  --model google/gemma-4-31B-it \
  --served-model-name=gemma4 \
  --dyn-tool-call-parser gemma4 \
  --dyn-reasoning-parser gemma4 \
  --tensor-parallel-size 2 --max-model-len 64000 --max-num-seqs 8 \
  --gpu-memory-utilization 0.95 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],"pass_config":{"fuse_allreduce_rms": false}}' \
  --speculative-config.model google/gemma-4-31B-it-assistant \
  --speculative-config.num_speculative_tokens 4 \
  --prefix-warmup-file '<warmup.json>' --prefix-warmup-parallel \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
```

Both parser names are already correct and are what the fix hangs off:

| flag | resolves to | registry |
|---|---|---|
| `--dyn-tool-call-parser gemma4` | `Gemma4EngineToolParser` (the patched class) | `vllm/tool_parsers/__init__.py:193` |
| `--dyn-reasoning-parser gemma4` | `Gemma4EngineReasoningParser` | `vllm/reasoning/__init__.py:51` |

Two model-card details that matter for this fix:

1. **Thinking is opt-in via `<|think|>` at the start of the system prompt.** For a
   latency-sensitive voice agent, leave it **out** — a thought block before the envelope is pure
   added time-to-first-audio. The reasoning parser is then harmless (it just never fires), and
   the grammar's thought prefix is an optional branch the model never takes. Either way the
   grammar is correct; this is a latency choice, not a correctness one.
2. **Previous turns' thoughts must not be replayed into the prompt** — only the final response.
   Worth checking the pipecat runtime doesn't echo a `reasoning` item back in the next request's
   `input`, since the Responses API surfaces it separately from `assistant_reply`.

One grammar case the 31B chat template forced, now handled: the template can leave the prompt
ending **inside an open `<|channel>` block** (`Gemma4Parser.adjust_initial_state_from_prompt`,
vLLM issue #45834) — the post-tool-response continuation, i.e. SquadStack's request 2. There the
model emits reasoning text and only the **closer** `<channel|>`, never the opener. The tag
accepts both openings; without that, request 2 would have been rejected by its own grammar
whenever thinking was enabled. Covered by the last two rows of the builder self-test (11/11).

## Does dynamo affect this?

Two parts of dynamo matter, one a lot:

**1. Whether the patch runs at all — verify this first.** The flags are `--dyn-tool-call-parser`
/ `--dyn-reasoning-parser`, i.e. **dynamo-owned**, not vLLM's own `--tool-call-parser`. So
dynamo is selecting the parser class through its own code path, and everything here depends on
one question: does that path call `ToolParser.adjust_request()`?

- If it does (directly, or by going through vLLM's `online_renderer`, which calls it at
  `renderers/online_renderer.py:388-419`), the patch takes effect with no dynamo change.
- If dynamo only uses the *parsing* side (`extract_tool_calls` / `extract_tool_calls_streaming`)
  and builds sampling params itself from the OpenAI request, then `adjust_request` is never
  called, the patch is dead code, and the same translation has to go where dynamo maps
  `response_format` → structured output.

I could not check this here — dynamo isn't installed on this machine, only vLLM 0.26.0. On the
serving box:

```bash
DYN=$(python3 -c 'import dynamo.vllm, os; print(os.path.dirname(dynamo.vllm.__file__))')
grep -rn "adjust_request" $DYN                       # is it called on the request path?
grep -rn "structured_outputs\|response_format\|guided" $DYN   # where the constraint is built
grep -rn "dyn_tool_call_parser\|tool_call_parser" $DYN        # how the parser is wired in
```

Runtime probe after deploying — the patch logs the swap, so this is definitive:

```bash
VLLM_LOGGING_LEVEL=DEBUG python3 -m dynamo.vllm … 2>&1 | grep "replaced json_schema constraint"
```

One line per affected request means the fix is live. Silence while scenario B still fails means
dynamo bypasses `adjust_request`, and the translation belongs in dynamo's request preprocessing
instead — the builder module is independent of vLLM's request classes, so it can be called from
there unchanged; only `_extract_json_schema` / `_clear_json_schema` / `_tool_names` need
re-pointing at whatever request object dynamo hands over.

**2. Speculative decoding, which dynamo is configuring here.** The `-assistant` draft makes
spec decoding load-bearing in this deployment, and it is the one dynamo-side setting that
genuinely interacts with the fix: the structured-output bitmask has to be applied to draft
tokens as well, and the shipped gemma4 parser explicitly avoids guided decoding for forced tool
choice because it *"crashes EngineCore under speculative decoding"*. Run the acceptance matrix
with the draft model loaded, at `--max-num-seqs 8`. If it destabilizes, drop
`--speculative-config.*` to confirm the grammar itself is fine, then fix the spec-decode +
structured-output interaction — do not conclude the grammar is wrong.

Everything else dynamo does here is orthogonal to the grammar: the KV router and
`--kv-events-config`, prefix caching and `--prefix-warmup-file`, `--compilation-config` /
cudagraph capture sizes. The grammar is per-request generation-time masking; it neither reads
nor invalidates the KV cache, and a prefix-cache hit does not carry a matcher state across
requests.

One product decision worth confirming with SquadStack: on the **follow-up** request (scenario
C2) the tools are sent again, so tool-only output is legal there too and the model could chain a
second tool call instead of returning the envelope. If they want a guaranteed envelope on
tool-result turns, pass `allow_tools_only=False` when the last input item is a
`function_call_output` — the builder already takes that flag.
