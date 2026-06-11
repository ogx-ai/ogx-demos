# OGX Demos

This directory contains demo examples for getting started with OGX.

## Setup

First, install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), a fast Python package manager.

```bash
# 0️⃣ Install Ollama (if using local inference)
#    - Download and install from https://ollama.com/download
#    - Or use your package manager (recommended for security):
#      - macOS: brew install ollama
#      - Linux: Follow instructions at https://ollama.com/download/linux
#      - Windows: Download installer from https://ollama.com/download/windows

#    - Pull a model (required for inference). Use smaller models for CPU-only systems:
ollama pull llama3.2:1b    # 1B model - fast on CPU
# OR
ollama pull llama3.2:3b    # 3B model - default, slower on CPU

#    - Verify Ollama is running (should return JSON with model list):
curl http://localhost:11434/api/tags

# 1️⃣ Create a virtual environment in the current directory (.venv)
#    - Use Python 3.12 explicitly
#    - --seed ensures pip and core packaging tools are installed in the venv
uv venv --python 3.12 --seed

# 2️⃣ Activate the virtual environment
#    - Updates PATH so `python` and `pip` now point to .venv/bin/
#    - Sets VIRTUAL_ENV for the current shell session
source .venv/bin/activate

# 3️⃣ Configure environment variables
#    - Copy the example env file to create your local config
cp .env.example .env
#    - Edit .env to configure your inference provider(s) and API keys.
#      Uncomment and fill in the providers you want to use:
#
#      Ollama (local, free):
#        OLLAMA_URL="http://localhost:11434/v1"
#
#      OpenAI (cloud, recommended for agent demos):
#        OPENAI_API_KEY="sk-..."
#
#      You can enable multiple providers at once — OGX auto-detects them.
#      See demos/00_setup/README.md for the full list of supported providers.
#
#    - For agent demos (04_agents), also set a search API key:
#        TAVILY_SEARCH_API_KEY="tvly-..."

# 4️⃣ Install all dependencies
#    - This installs ogx with starter extras (provider dependencies like
#      ollama, chromadb, faiss, sentence-transformers, etc.) and ogx-client
uv sync

# 5️⃣ Start the OGX server
#    - Load .env so the server can detect your configured providers
#    - The server auto-detects providers based on which API keys are set
#    - It starts on port 8321 by default
#    - IMPORTANT: Keep this terminal open - the server runs in foreground
set -a; source .env; set +a
uv run ogx run starter

# 6️⃣ Verify the server is running (in a NEW terminal - server must be running!)
#    - Open a SECOND terminal window
#    - Navigate to the repository directory and activate the virtual environment
cd <repo-root>  # Navigate to where you cloned the repo
source .venv/bin/activate

# 7️⃣ Test the connection
#    - Run the client setup demo to verify server is running
python -m demos.01_foundations.01_client_setup localhost 8321  # Note: port 8321 for local starter server
```

### Troubleshooting

**No inference providers detected:**
```bash
# Make sure your .env has at least one provider configured and exported
set -a; source .env; set +a
# Verify the key is in the environment
echo $OPENAI_API_KEY   # or $OLLAMA_URL, etc.
```

**Port already in use (8321):**
```bash
lsof -i :8321
kill <PID>
```

**Check which providers and models are available:**
```bash
python -m demos.00_setup.01_list_providers localhost 8321
```

## Available Demos

### 00_setup
Server setup and provider configuration. Verify your OGX server is running and inspect available providers and models.

### 01_foundations
Foundation examples demonstrating core OGX concepts: client setup, chat completions, vector DBs, tool registration, and MCP.

### 02_responses_basics
Higher-level response generation with tools, structured outputs, streaming, and multi-turn conversations.

### 03_rag
RAG (Retrieval-Augmented Generation) examples showing how to ground model responses in retrieved documents using OGX's vector stores and search capabilities.

### 04_agents
Agent examples demonstrating conversational agents with chat, tools, RAG, ReAct, and multi-agent routing.

### 05_observability
OpenTelemetry, Jaeger, Prometheus, and Grafana integration for monitoring OGX servers. *(Requires Docker for the telemetry stack.)*

### 06_openai_compatibility
Demos showing that existing OpenAI Python SDK code works against an OGX server with only a `base_url` change, covering chat completions, tool calling, and the Responses API.
