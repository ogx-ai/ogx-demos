# Llama Stack Demos

This directory contains demo examples for getting started with Llama Stack.

## Setup

### Prerequisites

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), a fast Python package manager.

### Step 0: Install Ollama (for local inference)

**Install Ollama:**
- Download from https://ollama.com/download
- Or use your package manager (recommended for security):
  - macOS: `brew install ollama`
  - Linux: Follow instructions at https://ollama.com/download/linux
  - Windows: Download installer from https://ollama.com/download/windows

**Pull a model:**

> **Model Selection:** Most demos work with `llama3.2:3b` on CPU. Some agent demos need larger models for tool calling. Multimodal demos (e.g., `02_chat_multimodal.py`) require `llama3.2-vision:latest` which needs GPU (NVIDIA/AMD) or a remote Llama Stack server (e.g., OpenShift AI).

```bash
ollama pull llama3.2:3b    # Start with this for most demos
```

**Verify Ollama is running:**
```bash
curl http://localhost:11434/api/tags
```
Should return JSON with your model list.

### Step 1: Create Virtual Environment

```bash
uv venv --python 3.12 --seed
```

This creates a virtual environment in `.venv/` with Python 3.12.

### Step 2: Activate Virtual Environment

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install Llama Stack

```bash
uv pip install -U llama_stack
```

This installs the `llama` CLI and core dependencies.

### Step 4: Install Llama Stack Client

```bash
uv pip install -U llama-stack-client
```

This is the Python client library for interacting with Llama Stack servers.

### Step 5: Install Demo Dependencies (REQUIRED)

```bash
llama stack list-deps starter | xargs -L1 uv pip install
```

**⚠️ IMPORTANT:** This step is required! It installs additional dependencies like `together`, `tavily`, etc. that the demos need. Skipping this will cause `ModuleNotFoundError`.

### Step 6: Start Llama Stack Server

```bash
OLLAMA_URL=http://localhost:11434/v1 uv run llama stack run starter
```

**Important notes:**
- This starts a local server on port 8321 (default for starter distribution)
- Keep this terminal open - the server runs in foreground
- The server must stay running for demos to work

### Step 7: Verify Server is Running

**In a NEW terminal:**

**Linux/macOS:**
```bash
cd <repo-root>  # Navigate to where you cloned the repo
source .venv/bin/activate
python -m demos.01_foundations.01_client_setup localhost 8321
```

**Windows:**
```cmd
cd <repo-root>
.venv\Scripts\activate.bat    # Or Activate.ps1 for PowerShell
python -m demos.01_foundations.01_client_setup localhost 8321
```

If successful, you should see the server responding with model information.

## Troubleshooting

### Port already in use (8321)

```bash
# Find and kill the process using port 8321
lsof -i :8321
kill <PID>
```

### Server not starting

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Check if a model is pulled
ollama list
```

### Version compatibility errors

```bash
# Reinstall all packages with matching versions
pip uninstall -y llama-stack llama-stack-api llama-stack-client
uv pip install -U llama-stack llama-stack-client
```

### ModuleNotFoundError (e.g., 'together', 'tavily')

You likely skipped Step 5. Run:
```bash
llama stack list-deps starter | xargs -L1 uv pip install
```

### Multimodal agent demos failing or timing out

Multimodal agent demos (e.g., `02_chat_multimodal.py`) require vision capabilities that most models don't have.

**Solution:**
- Pull the vision model: `ollama pull llama3.2-vision:latest` (requires NVIDIA/AMD GPU)
- Or connect to a remote Llama Stack server with GPU (e.g., OpenShift AI)

**Note:** Intel GPUs are not currently supported by Ollama. The vision model will timeout on CPU.

## Available Demos

### 01_foundations
Foundation examples demonstrating core Llama Stack concepts and basic usage patterns.

**Model requirements:** llama3.2:3b or any chat model

### 02_responses_basic
Basic examples showing how to work with responses in Llama Stack.

**Model requirements:** llama3.2:3b or any chat model

### 03_rag
RAG (Retrieval-Augmented Generation) examples showing how to ground model responses in retrieved documents using Llama Stack's vector stores and search capabilities.

**Model requirements:** llama3.2:3b or any chat model

### 04_agents
Agent examples demonstrating how to build conversational agents with various capabilities including chat, multimodal processing, document grounding, custom tools, and multi-agent coordination.

**Model requirements:**
- Most agent demos: llama3.2:3b or larger
- Multimodal demos (e.g., `02_chat_multimodal.py`): ollama/llama3.2-vision:latest (requires NVIDIA/AMD GPU)
- **Alternative for vision demos:** Use remote Llama Stack server (e.g., OpenShift AI) with GPU-accelerated models

### 06_openai_compatibility
Demos showing that existing OpenAI Python SDK code works against a Llama Stack server with only a `base_url` change, covering chat completions, tool calling, and the Responses API.

**Model requirements:** llama3.2:3b or any chat model
