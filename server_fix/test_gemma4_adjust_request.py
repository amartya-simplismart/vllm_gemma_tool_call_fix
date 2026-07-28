"""Server-side test: does the patched gemma4 tool parser produce a grammar that permits tools?

Exercises Gemma4EngineToolParser.adjust_request() with the exact request bodies SquadStack
production sends (Responses API) and the repro script sends (Chat Completions), then compiles
the resulting grammar and checks which output shapes it accepts.

No GPU, no model, no server needed — this is the request-shaping and grammar layer only.

    python server_fix/test_gemma4_adjust_request.py
"""

import json

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string as accepts

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser

ENVELOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "assistant_reply": {"type": "string"},
        "memory": {
            "type": ["object", "null"],
            "properties": {
                key: {"type": "string"}
                for key in (
                    "language",
                    "buyer_interested",
                    "seller_details_shared",
                    "callback_time",
                )
            },
        },
        "next_stage": {
            "type": ["string", "null"],
            "enum": ["STAGE_INTRO", "STAGE_SELLER_DETAILS", "STAGE_CLOSING", None],
        },
    },
    "required": ["assistant_reply", "memory", "next_stage"],
}

RESPONSES_TOOLS = [
    {
        "type": "function",
        "name": "fetch_seller_details",
        "description": "Fetch seller details for the available seller",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "strict": False,
    },
    {
        "type": "function",
        "name": "hangup_call_with_custom_delay",
        "description": "End the call with a custom delay.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "delay": {"type": "integer"},
            },
            "required": ["delay"],
        },
        "strict": False,
    },
]

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
            "strict": False,
        },
    }
    for tool in RESPONSES_TOOLS
]


def responses_request(**overrides):
    """Production shape: /v1/responses with text.format json_schema + flat tools."""
    body = {
        "model": "gemma4",
        "stream": True,
        "store": False,
        "truncation": "auto",
        "instructions": "You are Meera, a SquadStack voice agent.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "bye, rakhti hoon"}],
            }
        ],
        "tools": RESPONSES_TOOLS,
        "temperature": 0.7,
        "max_output_tokens": 400,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "call_response",
                "strict": False,
                "schema": ENVELOPE_SCHEMA,
            }
        },
    }
    body.update(overrides)
    return ResponsesRequest(**body)


def chat_request(**overrides):
    """Repro-script shape: /v1/chat/completions with response_format json_schema."""
    body = {
        "model": "gemma4",
        "stream": True,
        "messages": [{"role": "user", "content": "bye, rakhti hoon"}],
        "tools": CHAT_TOOLS,
        "temperature": 0.7,
        "max_tokens": 400,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "call_response",
                "schema": ENVELOPE_SCHEMA,
                "strict": False,
            },
        },
    }
    body.update(overrides)
    return ChatCompletionRequest(**body)


class _FakeParserEngine:
    """Stand-in for the real Gemma4 parser engine (needs a tokenizer we don't have here)."""

    def __init__(self, has_reasoning=True):
        self._has_reasoning = has_reasoning

    def adjust_request(self, request):
        request.skip_special_tokens = False
        return request


def adjust(request, has_reasoning=True):
    """Call the patched adjust_request without constructing a tokenizer-backed engine."""
    parser = Gemma4EngineToolParser.__new__(Gemma4EngineToolParser)
    parser._parser_engine = _FakeParserEngine(has_reasoning)
    return Gemma4EngineToolParser.adjust_request(parser, request)


# Output shapes the SquadStack runtime handles (see simplismart_repro_testing_guide.md).
ENVELOPE = (
    '{"assistant_reply": "theek hai, dhanyavaad", "memory": {"language": "hindi"}, '
    '"next_stage": "STAGE_CLOSING"}'
)
HANGUP = (
    '<|tool_call>call:hangup_call_with_custom_delay'
    '{message:<|"|>theek hai, dhanyavaad<|"|>,delay:6}<tool_call|>'
)
FETCH = "<|tool_call>call:fetch_seller_details{}<tool_call|>"
THOUGHT = "<|channel>thought\nbuyer is closing the call<channel|>"

SHAPES = [
    (True, "shape 1  envelope only", ENVELOPE),
    (True, "shape 2  tool call only (scenario B/C1)", HANGUP),
    (True, "shape 2  zero-arg tool call only", FETCH),
    (True, "shape 3  envelope + tool call", ENVELOPE + HANGUP),
    (True, "shape 3  thinking channel + envelope + tool call", THOUGHT + ENVELOPE + HANGUP),
    (False, "reject   unconstrained prose", "sorry, I can't help with that"),
    (False, "reject   envelope missing required keys", '{"assistant_reply": "hi"}'),
    (False, "reject   tool not in request", "<|tool_call>call:transfer_call{}<tool_call|>"),
]


def sampling_params(request):
    """The two APIs expose different to_sampling_params signatures."""
    if isinstance(request, ResponsesRequest):
        return request.to_sampling_params(default_max_tokens=400)
    return request.to_sampling_params(max_tokens=400, default_sampling_params={})


def grammar_from(request):
    so = sampling_params(request).structured_outputs
    assert so is not None, "no structured output constraint on the request"
    if so.structural_tag is not None:
        return xgr.Grammar.from_structural_tag(json.loads(so.structural_tag)), "structural_tag"
    if so.json is not None:
        schema = so.json if isinstance(so.json, str) else json.dumps(so.json)
        return xgr.Grammar.from_json_schema(schema), "json (envelope-only)"
    raise AssertionError(f"unexpected constraint: {so}")


def check(label, request, expect_tools_reachable, reasoning=True):
    grammar, kind = grammar_from(request)
    print(f"\n{label}\n  constraint: {kind}")
    failures = 0
    for expected, name, text in SHAPES:
        want = expected
        # Before the patch, every tool-call shape is unreachable — that IS the bug, so when
        # checking baseline behavior the tool shapes are expected to be rejected.
        if not expect_tools_reachable and "tool call" in name:
            want = False
        # With no reasoning parser configured, the thinking channel is correctly forbidden.
        if not reasoning and "thinking channel" in name:
            want = False
        got = accepts(grammar, text)
        ok = got is want
        failures += not ok
        print(f"    {'PASS' if ok else 'FAIL'}  {'accepts' if got else 'rejects':>7}  {name}")
    return failures


def main():
    failures = 0

    print("=" * 88)
    print("BASELINE — unpatched behavior: json_schema constraint, tool calls unreachable")
    print("=" * 88)
    # Simulate the unpatched path by not calling adjust_request at all: the json_schema
    # response format becomes StructuredOutputsParams(json=...) on its own.
    failures += check("responses request, no adjust_request", responses_request(), False)

    print("\n" + "=" * 88)
    print("PATCHED — adjust_request installs the union grammar")
    print("=" * 88)
    failures += check(
        "responses request (production shape)", adjust(responses_request()), True
    )
    failures += check("chat completions request (repro shape)", adjust(chat_request()), True)
    failures += check(
        "responses request, reasoning parser disabled",
        adjust(responses_request(), has_reasoning=False),
        True,
        reasoning=False,
    )

    print("\n" + "=" * 88)
    print("FORCED TOOL CHOICE — a demanded tool call must also be reachable")
    print("=" * 88)

    named = adjust(chat_request(tool_choice={"type": "function",
                                             "function": {"name": "fetch_seller_details"}}))
    grammar, kind = grammar_from(named)
    checks = [
        (True, "the named tool", FETCH),
        (True, "envelope + the named tool", ENVELOPE + FETCH),
        (False, "envelope alone (tool is required)", ENVELOPE),
        (False, "a different tool", HANGUP),
    ]
    print(f"  constraint: {kind}, skip_special_tokens={named.skip_special_tokens}")
    for want, name, text in checks:
        got = accepts(grammar, text)
        ok = got is want
        failures += not ok
        print(f"    {'PASS' if ok else 'FAIL'}  {'accepts' if got else 'rejects':>7}  {name}")

    required = adjust(chat_request(tool_choice="required"))
    grammar, kind = grammar_from(required)
    checks = [
        (True, "either tool alone", HANGUP),
        (True, "envelope + a tool", ENVELOPE + HANGUP),
        (False, "envelope alone (tool is required)", ENVELOPE),
    ]
    print(f"  constraint: {kind}, skip_special_tokens={required.skip_special_tokens}")
    for want, name, text in checks:
        got = accepts(grammar, text)
        ok = got is want
        failures += not ok
        print(f"    {'PASS' if ok else 'FAIL'}  {'accepts' if got else 'rejects':>7}  {name}")

    print("\n" + "=" * 88)
    print("REGRESSIONS — everything else keeps its existing behavior")
    print("=" * 88)

    # Forced tool choice WITHOUT a json schema: unchanged early return, no grammar at all.
    bare = adjust(chat_request(response_format=None, tool_choice="required"))
    so = sampling_params(bare).structured_outputs
    ok = so is None and bare.skip_special_tokens is False
    failures += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  required tool_choice, no json_schema: no grammar "
          f"forced, skip_special_tokens={bare.skip_special_tokens}")

    no_tools = adjust(chat_request(tools=None, tool_choice=None))
    _, kind = grammar_from(no_tools)
    ok = kind.startswith("json")
    failures += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  no tools: plain json_schema untouched ({kind})")

    already = adjust(
        chat_request(
            response_format=None,
            structured_outputs={"structural_tag": json.dumps(
                {"type": "structural_tag", "format": {"type": "any_text", "excludes": []}}
            )},
        )
    )
    so = sampling_params(already).structured_outputs
    ok = so.structural_tag is not None and "any_text" in so.structural_tag
    failures += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  client-supplied structural_tag left untouched")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{failures} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
