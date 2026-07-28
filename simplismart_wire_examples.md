# Wire-level examples: tool turns with structured output (GPT-4.1, `POST /v1/responses`)

Captured live on 2026-07-25 by running `repro_simplismart_json_toolcall.py` against OpenAI
`gpt-4.1`. Requests are shown exactly as sent; responses exactly as returned (trimmed to
`output` + `usage` — the remaining response fields just echo the request configuration).
Captured with `"stream": false` for readability; production sends `"stream": true` and
receives the identical data as SSE events
(`response.output_item.added` / `response.output_text.delta` /
`response.function_call_arguments.delta` / `response.completed`).

## Common request skeleton

Every request below is this, plus a per-call `input` array. Note there is **no
`tool_choice`** and **no `parallel_tool_calls`** — production never sends them (the response
echoes back the API defaults `"auto"` / `true`).

```json
{
  "model": "gpt-4.1",
  "store": false,
  "instructions": "You are Meera, a SquadStack voice agent calling a buyer on behalf of IndiaMART.\nYou speak Hinglish (Hindi in Devanagari script with common English words). Replies must be\nshort (1-2 sentences), natural and phone-friendly.\n\n# TOOLS\n- Call fetch_seller_details when the buyer asks about the seller/supplier.\n- Call hangup_call_with_custom_delay when the conversation is over (buyer says goodbye,\n  is not interested, or asks to stop). Pass a short farewell as `message` and delay=6.\n\n# RESPONSE FORMAT — every turn\nAlways respond with a single JSON object:\n{\"assistant_reply\": \"<what you say to the buyer>\", \"memory\": {<only keys that changed>}, \"next_stage\": \"<stage or null>\"}\n- memory keys you may set: language, buyer_interested, seller_details_shared, callback_time\n- stages: STAGE_INTRO, STAGE_SELLER_DETAILS, STAGE_CLOSING\n- assistant_reply is spoken aloud to the buyer over the phone.\n",
  "tools": [
    {
      "type": "function",
      "name": "fetch_seller_details",
      "description": "Fetch seller details for the available seller",
      "parameters": {
        "type": "object",
        "properties": {},
        "required": []
      },
      "strict": false
    },
    {
      "type": "function",
      "name": "hangup_call_with_custom_delay",
      "description": "End the call with a custom delay. The delay is capped at 12 seconds.",
      "parameters": {
        "type": "object",
        "properties": {
          "message": {
            "type": "string",
            "description": "The hang up message"
          },
          "delay": {
            "type": "integer",
            "description": "Delay in seconds before ending the call. Must be between 1 and 12."
          }
        },
        "required": [
          "delay"
        ]
      },
      "strict": false
    }
  ],
  "temperature": 0.7,
  "max_output_tokens": 400,
  "truncation": "auto",
  "text": {
    "format": {
      "type": "json_schema",
      "name": "call_response",
      "schema": {
        "type": "object",
        "properties": {
          "assistant_reply": {
            "type": "string"
          },
          "memory": {
            "type": [
              "object",
              "null"
            ],
            "properties": {
              "language": {
                "type": "string"
              },
              "buyer_interested": {
                "type": "string"
              },
              "seller_details_shared": {
                "type": "string"
              },
              "callback_time": {
                "type": "string"
              }
            }
          },
          "next_stage": {
            "type": [
              "string",
              "null"
            ],
            "enum": [
              "STAGE_INTRO",
              "STAGE_SELLER_DETAILS",
              "STAGE_CLOSING",
              null
            ]
          }
        },
        "required": [
          "assistant_reply",
          "memory",
          "next_stage"
        ]
      },
      "strict": false
    }
  },
  "stream": false
}
```

---

## Case 1 — hangup turn: ONE LLM call

User says goodbye. The model returns a `function_call` output item **only** — no message, no
envelope. The runtime executes the hangup; there is no follow-up request.

### Request `input`

```json
[
  {
    "type": "message",
    "role": "assistant",
    "content": [
      {
        "type": "output_text",
        "text": "{\"assistant_reply\": \"नमस्ते! मैं मीरा बोल रही हूँ IndiaMART की तरफ से। आपने हाल में एक enquiry डाली थी, उसी के बारे में बात करनी थी।\", \"memory\": {\"language\": \"hindi\"}, \"next_stage\": \"STAGE_INTRO\"}"
      }
    ]
  },
  {
    "type": "message",
    "role": "user",
    "content": [
      {
        "type": "input_text",
        "text": "theek hai thank you, mujhe abhi baat nahi karni, rakhti hoon. bye."
      }
    ]
  }
]
```

### Response (verbatim)

```json
{
  "id": "resp_05d033d825ceb491016a63ccdf3d14819db4be036b37acde4c",
  "model": "gpt-4.1-2025-04-14",
  "status": "completed",
  "output": [
    {
      "arguments": "{\"message\":\"कोई बात नहीं, धन्यवाद! अच्छा दिन हो आपका. अलविदा!\",\"delay\":6}",
      "call_id": "call_utvR67UMN9S15riWC5XIrLFE",
      "name": "hangup_call_with_custom_delay",
      "type": "function_call",
      "id": "fc_05d033d825ceb491016a63ccdff644819db84fd391763ac531",
      "status": "completed"
    }
  ],
  "usage": {
    "input_tokens": 507,
    "output_tokens": 39,
    "total_tokens": 546
  }
}
```

---

## Case 2 — fetch turn: TWO LLM calls

User asks about the seller.

### Call 1 — request `input`

```json
[
  {
    "type": "message",
    "role": "assistant",
    "content": [
      {
        "type": "output_text",
        "text": "{\"assistant_reply\": \"नमस्ते! मैं मीरा बोल रही हूँ IndiaMART की तरफ से। आपने हाल में एक enquiry डाली थी, उसी के बारे में बात करनी थी।\", \"memory\": {\"language\": \"hindi\"}, \"next_stage\": \"STAGE_INTRO\"}"
      }
    ]
  },
  {
    "type": "message",
    "role": "user",
    "content": [
      {
        "type": "input_text",
        "text": "accha pehle ये batao seller kaun hai? unke baare mein details do."
      }
    ]
  }
]
```

### Call 1 — response (verbatim): `function_call` only

```json
{
  "id": "resp_09c7ee5fa263b0b0016a63cce0cf90819d917ece809664ce8f",
  "model": "gpt-4.1-2025-04-14",
  "status": "completed",
  "output": [
    {
      "arguments": "{}",
      "call_id": "call_W11Fn1oIHRuo4ubh9CH5vsuv",
      "name": "fetch_seller_details",
      "type": "function_call",
      "id": "fc_09c7ee5fa263b0b0016a63cce13bb0819daf79efa2fea50f08",
      "status": "completed"
    }
  ],
  "usage": {
    "input_tokens": 504,
    "output_tokens": 13,
    "total_tokens": 517
  }
}
```

### The runtime then executes the tool and appends two items to `input`

These two items (verbatim from call 2's request) are the tool round trip — the
`function_call` the model made, and the tool's result as `function_call_output`:

```json
[
  {
    "type": "function_call",
    "name": "fetch_seller_details",
    "arguments": "{}",
    "call_id": "call_W11Fn1oIHRuo4ubh9CH5vsuv"
  },
  {
    "type": "function_call_output",
    "call_id": "call_W11Fn1oIHRuo4ubh9CH5vsuv",
    "output": "{\"status\": \"success\", \"message\": \"Available seller details fetched successfully\", \"response\": {\"seller_name\": \"Sharma Industrial Supplies\", \"location\": \"Ludhiana, Punjab\", \"products\": [\"hydraulic pumps\", \"industrial valves\"], \"rating\": \"4.2\"}}"
  }
]
```

### Call 2 — request `input` (full)

Same skeleton — **same `tools`, same `text.format`** — with the grown `input`:

```json
[
  {
    "type": "message",
    "role": "assistant",
    "content": [
      {
        "type": "output_text",
        "text": "{\"assistant_reply\": \"नमस्ते! मैं मीरा बोल रही हूँ IndiaMART की तरफ से। आपने हाल में एक enquiry डाली थी, उसी के बारे में बात करनी थी।\", \"memory\": {\"language\": \"hindi\"}, \"next_stage\": \"STAGE_INTRO\"}"
      }
    ]
  },
  {
    "type": "message",
    "role": "user",
    "content": [
      {
        "type": "input_text",
        "text": "accha pehle ये batao seller kaun hai? unke baare mein details do."
      }
    ]
  },
  {
    "type": "function_call",
    "name": "fetch_seller_details",
    "arguments": "{}",
    "call_id": "call_W11Fn1oIHRuo4ubh9CH5vsuv"
  },
  {
    "type": "function_call_output",
    "call_id": "call_W11Fn1oIHRuo4ubh9CH5vsuv",
    "output": "{\"status\": \"success\", \"message\": \"Available seller details fetched successfully\", \"response\": {\"seller_name\": \"Sharma Industrial Supplies\", \"location\": \"Ludhiana, Punjab\", \"products\": [\"hydraulic pumps\", \"industrial valves\"], \"rating\": \"4.2\"}}"
  }
]
```

### Call 2 — response (verbatim): schema-conforming envelope, no tool call

```json
{
  "id": "resp_0c83f138d35c13f7016a63cce1a9cc81a3a912dda6a5df96b0",
  "model": "gpt-4.1-2025-04-14",
  "status": "completed",
  "output": [
    {
      "id": "msg_0c83f138d35c13f7016a63cce208dc81a3b8de40a9deea9c5a",
      "content": [
        {
          "annotations": [],
          "text": "{\"assistant_reply\":\"Seller ka naam Sharma Industrial Supplies hai, Ludhiana, Punjab se hain aur mainly hydraulic pumps aur industrial valves supply karte hain. Aur koi info chahiye?\",\"memory\":{\"seller_details_shared\":\"yes\"},\"next_stage\":\"STAGE_SELLER_DETAILS\"}",
          "type": "output_text",
          "logprobs": []
        }
      ],
      "role": "assistant",
      "status": "completed",
      "type": "message"
    }
  ],
  "usage": {
    "input_tokens": 583,
    "output_tokens": 64,
    "total_tokens": 647
  }
}
```

---

## Summary

| Call | `input` ends with | `output` contains |
|---|---|---|
| Case 1 (hangup) | user goodbye | one `function_call`, no message |
| Case 2, call 1 (fetch) | user question | one `function_call`, no message |
| Case 2, call 2 (fetch) | `function_call` + `function_call_output` | one `message` with the schema JSON envelope |

On tool turns GPT-4.1 usually returns the `function_call` only (as above). With large
production prompts it *sometimes* returns BOTH the envelope message and the `function_call`
in one response — the client consumes both channels independently, so either shape must
work.
