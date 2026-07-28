# Dynamo-side fix: emit a union tag when `response_format` and `tools` are both present

The vLLM-side fix in `../server_fix` never runs under Dynamo's default (Rust) frontend — the Rust
ingress builds `sampling_options.guided_decoding` itself and never calls vLLM's Python
`ToolParser.adjust_request`. These two patches fix it where the decision is actually made.

Verified end to end on `gemma-4-31B-it` through `dynamo.frontend` + `dynamo.vllm` with **no extra
launch flags** (see "Results" below).

## What was wrong, in Dynamo

Three separate gaps:

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

3. **`dynamo-llm`: forced tool choice falls back to a generic JSON tool-call shape.** With tag
   mode off (the default), `tool_choice: "required"` or a named tool routes to
   `get_json_schema_from_tools`, which constrains output to an OpenAI-style JSON tool-call object.
   Gemma 4 does not speak that — it emits `<|tool_call>call:name{...}<tool_call|>` — so the model
   is forced into a shape its own parser cannot read and the turn is lost. Measured before the fix:
   a named `tool_choice` returned `{\n  \n  \n  …` filler and zero tool calls. This is the same
   trap vLLM's gemma4 parser avoids by skipping guided decoding entirely for forced choice.

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

Under a forced tool choice (`required` or named) the message branch may still precede a call but
cannot stand alone, so the `[ tool_calls ]` above becomes mandatory.

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

Forced tool choice, same server (3 runs, `../client_fix/verify_scenario_a.py` config 3, which
sends a named `tool_choice` on the single-tool stage):

| | before | after |
|---|---|---|
| named `tool_choice` + `response_format` | 0/3 — `{\n  \n  …` filler | **3/3** native call |

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

### Fixing scenario A — two levers, both measured

Scenario A is a policy question ("should the model act on this turn?"), so the fix belongs in the
prompt or in the per-turn `tool_choice`, not in the grammar. Both were tested on
gemma-4-31B-it through patched dynamo, 3 runs each, same non-tool turn:

| approach | envelope | tool called |
|---|---|---|
| baseline: `auto` + the repro's compact prompt | 1/3 | 2/3 |
| **`tool_choice: "none"` on non-tool turns** | **3/3** | **0/3** |
| **stricter prompt + `auto`** | **3/3** | **0/3** |

**1. Per-turn `tool_choice: "none"` (hard guarantee).** The runtime already knows the call stage,
so it can send `none` on turns where no tool is permitted. The tag logged on the way to vLLM shows
no tool branch at all in that case, so this is enforced by the grammar rather than left to the
model's judgment. Tool turns keep `auto` and still fire 3/3.

This exposed a bug in the first version of this patch: the union was also being built for
`tool_choice: "none"`, so the tool branch was still offered and the model merely happened not to
take it. The patch now returns early for `None`, and the logged tag confirms `tool_branch=false`.

**2. Stricter prompt (better judgment, no client change).** The repro's stand-in prompt says only
*"Call fetch_seller_details when the buyer asks about the seller/supplier"*. Adding an explicit
negative rule fixed it 3/3 with `auto`:

```
- Call fetch_seller_details ONLY when the buyer explicitly asks about the seller/supplier.
  NEVER call it in STAGE_INTRO or to answer a question about the buyer's own enquiry.
  If no tool is clearly required, do not call any tool — just reply with the JSON object.
```

The production campaign prompt (~15K tokens) very likely already carries rules of this kind, which
is a further reason to re-measure scenario A against the real prompt rather than the stand-in.

Recommended: use both — the prompt for judgment, and `tool_choice: "none"` on stages where a tool
call is definitionally wrong, for a guarantee that does not depend on model behaviour.
