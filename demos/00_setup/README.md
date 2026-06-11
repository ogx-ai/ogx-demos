# 00 Setup: Server and Provider Configuration

Before running demos, you need an OGX server with at least one inference provider configured. OGX auto-detects providers based on environment variables.

## Quick Start (Ollama only)

```bash
ollama pull llama3.2:3b
OLLAMA_URL=http://localhost:11434/v1 uv run ogx run starter
```

## Adding Providers

OGX auto-detects providers when their API key is set in the environment. Set the keys in `.env` before starting the server:

| Provider | Environment Variable | Notes |
|----------|---------------------|-------|
| Ollama | `OLLAMA_URL` | Default: `http://localhost:11434/v1` |
| OpenAI | `OPENAI_API_KEY` | GPT-4o, GPT-4o-mini, etc. |
| Anthropic | `ANTHROPIC_API_KEY` | Claude models |
| Google Gemini | `GEMINI_API_KEY` | Gemini models |
| vLLM | `VLLM_URL` | Default: `http://localhost:8000/v1` |
| Together | `TOGETHER_API_KEY` | |
| Fireworks | `FIREWORKS_API_KEY` | |
| AWS Bedrock | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | |
| Azure OpenAI | `AZURE_API_KEY` + `AZURE_API_BASE` | |

### Example: Ollama + OpenAI

```bash
# In .env:
OLLAMA_URL=http://localhost:11434/v1
OPENAI_API_KEY=sk-...

# Start server (auto-detects both providers):
source .env
uv run ogx run starter
```

The server will show detected providers on startup:

```
Scanning for available providers...
  ✓ remote::ollama ... (2 models)
  ✓ remote::openai ... (10 models)
```

## Verify Your Setup

```bash
python -m demos.00_setup.01_list_providers localhost 8321
```

This lists all configured providers and available models.

## Which Provider Do I Need?

| Demo category | Minimum | Recommended |
|--------------|---------|-------------|
| 01 Foundations | Any LLM + embedding model | Ollama |
| 02 Responses | Any LLM | Ollama |
| 03 RAG | Any LLM + embedding model | Ollama |
| 04 Agents | Fast LLM with tool calling | OpenAI (gpt-4o-mini) |
| 06 OpenAI Compatibility | Any LLM | Ollama |

Agent demos (04) involve multi-turn tool calling and work best with faster cloud models. Local models via Ollama may time out on complex agent tasks.
