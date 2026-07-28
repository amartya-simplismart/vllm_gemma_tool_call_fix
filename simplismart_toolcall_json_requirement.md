# Requirement: tool calling + structured JSON (`response_format`) in a single request

**Context:** On GPT-4.1, one request that sets a JSON `response_format` (our
`{assistant_reply, memory, next_stage}` schema) **and** passes `tools` returns *both* in the
same response — the schema-conforming JSON as the message content **and** a native
`tool_call`. On gemma-4 the same request returns only the JSON; the tool call never fires.
We need gemma to support the same single-request behaviour.

## Why this single-request structure is essential for us

Our voice agent produces a **structured JSON envelope on every turn** — it isn't just tool
arguments, it carries the **conversation memory and the call-stage state** that drive the
entire dialogue. At the same time, the model must be able to **invoke a tool** (hang up,
transfer, hold, data-lookup) as part of that same turn.

Both are decided together, in one model generation, and our runtime is built on that
single-request contract: the streamed JSON is parsed token-by-token to (a) speak
`assistant_reply`, (b) update memory, (c) advance the stage, and (d) dispatch any tool — all
from one coherent output. The state and the action come from the *same* reasoning step.

## Why Option 1 (two requests — tool pass, then JSON pass) does not work for us

Because we need the JSON envelope on **every** turn (for memory/stage), and the tool-deciding
pass must run **without** `response_format`, this forces **two requests on every turn — not
just tool turns**. On a latency-sensitive live phone call at high concurrency, that ~doubles
per-turn latency and cost across the whole conversation.

It also **splits "what to say / remember / next stage" from "which tool" into two independent
generations**, which can disagree — the model may record state or pick a next stage
inconsistent with the action it actually took. Our state integrity depends on these being one
atomic generation, which two requests cannot guarantee.

## Why Option 2 (prompt-based JSON + native tools) does not work for us

Dropping `response_format` and asking for the JSON shape in the prompt removes the **hard
schema guarantee** — the envelope becomes best-effort. Our pipeline streams and parses that
envelope in real time to drive TTS and state; a drifting or malformed envelope means
**dropped memory/stage updates and TTS failures (dead air or garbled speech) on a live
call**. The guaranteed grammar is the entire reason we use `response_format` — best-effort
JSON reintroduces exactly the failure mode we adopted structured output to eliminate.

## The ask

Support `response_format` (json_schema) **and** `tools` in the **same request**, returning
**both** a schema-conforming message **and** a native `tool_call` in one response — the
behaviour GPT-4.1's API already provides. That single-request contract is what our runtime is
built on; the two workarounds each break a property (latency/cost, state atomicity, or the
hard schema guarantee) that we can't give up.
