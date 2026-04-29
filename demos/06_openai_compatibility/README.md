# OpenAI Compatibility Demos

Llama Stack implements OpenAI-compatible APIs (`/v1/chat/completions`, `/v1/responses`). These demos prove that **existing OpenAI code works against a Llama Stack server** with only a `base_url` change — no SDK swap required.

## Key Differences from Native Llama Stack Client

| | Native `LlamaStackClient` | `openai.OpenAI` |
|---|---|---|
| **base_url** | `http://host:port` | `http://host:port/v1` (needs `/v1` suffix) |
| **api_key** | Not required | Required by SDK; use `"fake"` or set `LLAMA_STACK_API_KEY` |
| **Shared utils** | Can import `demos.shared.utils` | Uses `resolve_openai_model` from `demos.shared.utils` |

## Demos

### 01_chat_completion.py
Non-streaming and streaming chat completions using `openai.OpenAI`.

```bash
# Non-streaming
python -m demos.06_openai_compatibility.01_chat_completion localhost 8321

# Streaming
python -m demos.06_openai_compatibility.01_chat_completion localhost 8321 --stream
```

### 02_tool_calling.py
Standard OpenAI function-calling flow: define tools, parse `tool_calls`, execute locally, send results back.

```bash
python -m demos.06_openai_compatibility.02_tool_calling localhost 8321
```

### 03_responses_api.py
Uses `client.responses.create()` from the OpenAI SDK against Llama Stack, with both string and structured message-list inputs.

```bash
python -m demos.06_openai_compatibility.03_responses_api localhost 8321
```

### 04_responses_max_output_tokens.py
Demonstrates `max_output_tokens` to limit response length and inspect `incomplete_details` when output is truncated.

```bash
python -m demos.06_openai_compatibility.04_responses_max_output_tokens localhost 8321
```

### 05_responses_top_p.py
Uses `top_p` (nucleus sampling) to control sampling diversity, comparing low, medium, and high values.

```bash
python -m demos.06_openai_compatibility.05_responses_top_p localhost 8321
```

### 06_responses_truncation.py
Shows the `truncation` parameter (`"auto"` vs `"disabled"`) for handling long multi-turn contexts.

```bash
python -m demos.06_openai_compatibility.06_responses_truncation localhost 8321
```

### 07_responses_streaming.py
Streaming with the Responses API, including `stream_options` with `include_usage` to get token usage statistics.

```bash
python -m demos.06_openai_compatibility.07_responses_streaming localhost 8321
```

### 08_responses_parallel_tool_calls.py
Uses `parallel_tool_calls` to control whether the model issues multiple tool calls in a single turn.

```bash
python -m demos.06_openai_compatibility.08_responses_parallel_tool_calls localhost 8321
```

### 09_responses_service_tier.py
Sets the `service_tier` parameter (`"auto"`, `"default"`) to control request routing.

```bash
python -m demos.06_openai_compatibility.09_responses_service_tier localhost 8321
```

### 10_responses_logprobs.py
Requests `top_logprobs` to inspect token-level probability data for model interpretability.

```bash
python -m demos.06_openai_compatibility.10_responses_logprobs localhost 8321
```

### 11_responses_reasoning.py
Uses `reasoning.effort` (`"low"`, `"medium"`, `"high"`) to control how much reasoning the model applies.

```bash
python -m demos.06_openai_compatibility.11_responses_reasoning localhost 8321
```

### 12_responses_temperature.py
Controls output randomness with `temperature`, comparing deterministic (0.0) vs creative (1.5) outputs.

```bash
python -m demos.06_openai_compatibility.12_responses_temperature localhost 8321
```

### 13_responses_combined.py
Combines multiple parameters (`max_output_tokens`, `temperature`, `top_p`, `truncation`, `service_tier`, `stream`) in single requests.

```bash
python -m demos.06_openai_compatibility.13_responses_combined localhost 8321
```

## Prerequisites

- A running Llama Stack server (e.g. `llama stack run starter`)
- The `openai` Python package (`pip install openai>=1.75.0`)
