# Client-side: keeping speech-only turns speech-only

The server fixes (`../server_fix`, `../dynamo_fix`) make a tool call *possible* when a
`json_schema` response format is set. That is the bug fix. What remains is a policy question the
server cannot answer: **should the model act on this turn at all?**

Before the fix the schema made a tool call impossible, so every turn produced the envelope — the
acceptance script's scenario A passed for the wrong reason. With the channel unlocked, gemma-4-31B
sometimes calls `fetch_seller_details` on the intro turn instead of answering.

Two fixes, both measured, meant to be used together:

| | `prompt_tool_gating.md` | `tool_choice_policy.py` |
|---|---|---|
| mechanism | model judgment | decoding grammar — no tool branch compiled |
| guarantee | statistical | hard |
| client change | none (prompt only) | set `tool_choice` per turn |

## Files

| File | Role |
|---|---|
| `tool_choice_policy.py` | drop-in per-turn `tool_choice` policy: stage → allowed tools → `"none"` / named / `"auto"` |
| `prompt_tool_gating.md` | the prompt rule to add per tool, and why the positive-only phrasing over-calls |
| `verify_scenario_a.py` | runs the repro's three scenarios in all four configurations against any endpoint |

## Measured — `gemma-4-31B-it` via patched dynamo, 3 runs per cell

```
config                A         B         C         total
---------------------------------------------------------
1 baseline            0/3       3/3       3/3       6/9
2 prompt only         3/3       3/3       3/3       9/9
3 tool_choice only    3/3       3/3       3/3       9/9
4 both                3/3       3/3       3/3       9/9
```

Reproduce with:

```bash
python verify_scenario_a.py --base-url http://<host>/v1 --api-key <token> --model gemma4 --runs 3
```

Each fix alone gets to 9/9 here. Prefer both: the prompt makes the model's own judgment right on
turns you have not enumerated, and the policy makes the turns you *have* enumerated
grammar-enforced rather than a matter of model behaviour.

## What the policy does

```python
policy = ToolPolicy(REPRO_RULES)                       # stage -> permitted tools
request["tool_choice"] = policy.for_turn(
    stage=current_stage,                               # from the envelope's next_stage
    after_tool_result=is_follow_up_request,            # True on the 2nd of a 2-call tool
)
```

- **speech-only stage** (`STAGE_INTRO`) → `"none"`: the server compiles no tool branch, so the
  envelope is guaranteed. Verified by logging the tag on its way to vLLM (`tool_branch=false`).
- **single-tool stage** → a **named** `tool_choice`: the grammar admits only that tool, so the
  wrong tool cannot fire either. This path needed the third dynamo fix — before it, a named
  choice produced `{\n  \n  …` filler on gemma4.
- **multi-tool stage** → `"auto"`, the normal case.
- **follow-up request carrying a `function_call_output`** → `"none"`. That request exists only to
  turn the tool result into the envelope; a second tool call there is never wanted. This also
  removes the dual-output turns seen on scenario C2, where the follow-up returned an envelope
  *and* another `fetch_seller_details` call.

`filter_tools()` is provided to narrow the `tools` array to the same stage rules — shorter prompts
and no chance of `tool_choice` and `tools` disagreeing.

Stages default to *all* tools when absent from the rules, so adding a stage never silently
disables its tools.
