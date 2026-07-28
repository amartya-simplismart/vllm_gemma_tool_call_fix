# Dynamo-side fix: emit a union tag when `response_format` and `tools` are both present

The vLLM-side fix in `../server_fix` never runs under Dynamo's default (Rust) frontend — the Rust
ingress builds `sampling_options.guided_decoding` itself and never calls vLLM's Python
`ToolParser.adjust_request`. These two patches fix it where the decision is actually made.

Verified end to end on `gemma-4-31B-it` through `dynamo.frontend` + `dynamo.vllm` with **no extra
launch flags** (see "Results" below).

## What was wrong, in Dynamo

Two separate gaps:

1. **`dynamo-parsers`: gemma4 had no structural tag builder at all** —
   `ToolCallConfig::gemma4()` set `structural_tag_builder: None`, so
   `--dyn-enable-structural-tag` silently did nothing for this model family
   ("parser does not support it").

2. **`dynamo-llm`: the constraint selection bails out** — in
   `lib/llm/src/preprocessor/tool_choice.rs`:

   ```rust
   let has_assistant_constraint = has_explicit_guided_decoding || has_response_format_constraint;
   if !is_forced_tool_choice && has_assistant_constraint {
       return Ok(false);   // <-- leaves the schema as the only grammar
   }
   ```

   The comment reads *"response_format constrains assistant content, so tool-choice guided
   decoding stays inactive"* — but that is not neutral when tools are present. The schema grammar
   accepts nothing but the schema object, so a model that opens a native tool call is coerced into
   the JSON string and never terminates: the request burns its whole token budget and the client
   gets a malformed object and no tool call.

## The patches

| File | Change |
|---|---|
| `dynamo-parsers.patch` | adds `or` / `optional` format nodes; adds an `unconstrained_content` option to `TriggeredTagsConfig` (Gemma 4 argument syntax is not JSON); gives `gemma4()` a real structural tag builder; adds `build_tool_call_format_with_content_schema()` which composes the union |
| `dynamo-llm.patch` | in the non-forced + content-constrained branch, compose the union tag instead of returning early; clear `guided_decoding.json` (exactly one constraint is allowed) and set `structural_tag` |

The grammar emitted (verified by logging what reaches vLLM):

```
[ optional reasoning block ]
(
    <response_format schema>  [ tool_calls ]     # message, optionally + tool calls
  | tool_calls                                   # tool calls alone
)
```

Both openings of the reasoning block are accepted, because the tag cannot see the prompt: the
model may open the channel itself, or the chat template may leave the prompt inside an already
open block so only the closer is emitted.

Returns `Ok(None)` — leaving existing behaviour untouched — when there are no tools, no JSON
schema to preserve, no `--dyn-tool-call-parser`, or a parser without a tag builder. No new launch
flag: `--dyn-enable-structural-tag` is **not** required, because the combination being fixed is
broken by construction rather than merely suboptimal.

## Build

```bash
git clone https://github.com/ai-dynamo/dynamo.git && cd dynamo && git checkout v1.3.0
patch -p1 < /path/to/dynamo-llm.patch

# dynamo-parsers is a crates.io dependency: vendor and patch it, then redirect the build
cp -r ~/.cargo/registry/src/index.crates.io-*/dynamo-parsers-5.0.0 ../vendor/dynamo-parsers
chmod -R u+w ../vendor/dynamo-parsers
patch -d ../vendor/dynamo-parsers -p1 < /path/to/dynamo-parsers.patch
cat >> Cargo.toml <<'EOF'

[patch.crates-io]
dynamo-parsers = { path = "../vendor/dynamo-parsers" }
EOF
# the python bindings are their own workspace and need the same redirect
cat >> lib/bindings/python/Cargo.toml <<'EOF'

[patch.crates-io]
dynamo-parsers = { path = "../../../../vendor/dynamo-parsers" }
EOF

cd lib/bindings/python && cargo build --release
cp target/release/lib_core.so "$(python3 -c 'import dynamo._core as c; print(c.__file__)')"
```

Build notes for a host without NVIDIA NIXL or clang's builtin headers (both hit on the test box):

```bash
# dynamo-memory depends on nixl-sys unconditionally; the stub API needs no NIXL install
sed -i 's|^nixl-sys = { version = "=1.0.1" }|nixl-sys = { version = "=1.0.1", features = ["stub-api"] }|' lib/memory/Cargo.toml
# bindgen needs a stdbool.h
export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-linux-gnu/13/include"
```

Neither workaround belongs in a production image — build in the normal Dynamo container, where
NIXL and clang headers are present, and skip both.

## Results — `gemma-4-31B-it`, dynamo default Rust frontend, no extra flags

Launch used: `python -m dynamo.vllm --model <31B> --dyn-tool-call-parser gemma4
--dyn-reasoning-parser gemma4 --tensor-parallel-size 2 --max-model-len 32000 --max-num-seqs 8`
plus `python -m dynamo.frontend --http-port 8090`, streaming client.

`repro_simplismart_json_toolcall.py`, 4 runs:

| scenario | before (stock dynamo) | after |
|---|---|---|
| B: hangup turn → tool must fire | 0/4 | **4/4** |
| C1: fetch turn → tool must fire | 0/4 | **4/4** |
| C2: follow-up → schema-conforming envelope | n/a | **4/4** |
| A: non-tool turn → envelope only | 4/4 | 1/4 — see below |

B and C were previously **impossible**; they now pass every run, with clean typed arguments
(`{"delay":6,"message":"ठीक है, कोई बात नहीं। अपना ख्याल रखिये, बाय!"}`) and a follow-up envelope
that uses the tool result.

### Scenario A is model choice, not a grammar defect

With the tool channel unlocked, the model sometimes calls `fetch_seller_details` on the intro turn
("haan boliye, kya enquiry thi meri?") instead of answering — 3 of 4 runs. Before the fix it
always answered, because the schema made a tool call impossible; that "pass" was the bug masking
the model's actual preference.

The tag logged on the way to vLLM confirms the grammar offers both branches with no bias, and A
does still produce the envelope on some runs. Two things worth noting before treating this as a
regression:

- the repro's system prompt is a compact stand-in for a ~15K-token production campaign prompt with
  far more explicit stage/tool rules; this is a prompt-behaviour question, and the production
  prompt should be re-measured before drawing conclusions;
- SquadStack's runtime already handles a tool-only turn (the envelope then comes from the follow-up
  request), so it is not a protocol violation — only an assertion in the acceptance script.

If envelope-only turns must be guaranteed, that belongs in the prompt or in a per-turn
`tool_choice: "none"` from the client, not in the grammar — forcing it back into the grammar is
exactly the bug this patch removes.
