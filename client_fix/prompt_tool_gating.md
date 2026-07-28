# Prompt gating — the judgment half of the scenario A fix

Once the server-side fix unlocks the tool channel, the model decides for itself whether a turn
warrants a tool. With a permissive prompt gemma-4-31B over-calls: on the intro turn
("haan boliye, kya enquiry thi meri?") it called `fetch_seller_details` on 2 of 3 runs instead of
answering.

The repro's stand-in prompt only states the positive case:

```
- Call fetch_seller_details when the buyer asks about the seller/supplier.
```

That tells the model when calling is *allowed*, never when it is *wrong*. Adding the negative rule
fixed it — 3/3 envelope, 0/3 tool calls, with plain `tool_choice: "auto"` and no client change:

```
- Call fetch_seller_details ONLY when the buyer explicitly asks about the seller/supplier.
  NEVER call it in STAGE_INTRO or to answer a question about the buyer's own enquiry.
  If no tool is clearly required, do not call any tool — just reply with the JSON object.
```

## The general shape

For each tool in a campaign prompt, state three things rather than one:

1. **when to call it** — the trigger condition, as specifically as possible;
2. **when not to call it** — the stages or intents where it is wrong, named explicitly;
3. **the default** — that not calling any tool is the normal outcome, and a plain reply is a
   complete answer.

Point 3 matters most. Without it a tool list reads as a menu the model is expected to use.

## Where this sits relative to the `tool_choice` policy

They are complementary and it is worth having both:

| | prompt gating | `tool_choice: "none"` |
|---|---|---|
| mechanism | model judgment | decoding grammar — no tool branch is compiled |
| guarantee | statistical | hard |
| needs a client change | no | yes (stage is already known to the runtime) |
| handles "wrong tool for this stage" | partially | yes, via a named `tool_choice` |

Use the prompt so the model's own judgment is right, and `tool_choice: "none"` on the stages where
a tool call is definitionally wrong — the intro turn, and the follow-up request that carries a
`function_call_output` and exists only to turn the tool result into the envelope.

See `tool_choice_policy.py` for a drop-in implementation of the second half, and
`verify_scenario_a.py` for the measurements.
