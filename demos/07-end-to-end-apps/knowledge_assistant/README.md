# Knowledge Assistant

A RAG-based document Q&A application built on Llama Stack. Upload documents or web pages into named knowledge bases, then ask questions grounded in your own content.

When multiple knowledge bases are selected, the app queries each one independently and synthesizes the results into a single answer — useful for comparing or combining information across sources.

**Stack**: FastAPI backend · Vite + React frontend · Llama Stack for all AI operations.

## What this demonstrates

| Llama Stack API | Used for |
|---|---|
| **Files API** | Uploading and storing documents |
| **Vector Stores API** | Creating per-KB vector stores, ingesting chunks, similarity search |
| **Inference API** | Generating answers from retrieved context; synthesizing across KBs |
| **Safety API** | Screening user questions through a shield before RAG; blocked requests never reach the model |
| **Conversations API** | Maintaining per-KB session history so follow-up questions have context |

**RAG pattern**: retrieval and generation are kept explicit and separate. For each query, the app calls `vector_stores.search()` to retrieve relevant chunks, then constructs a prompt with that context before calling the Inference API. This avoids relying on model-side tool calling, which is unreliable with smaller models.

**Multi-source pattern**: when multiple KBs are selected, each is queried independently in parallel, and a final synthesis call combines the results. This is parallel retrieval with synthesis — not the Llama Stack Agents API, but the same coordinator + specialist decomposition in structure.

## Prerequisites

- Llama Stack server running and reachable (tested with the `starter` distribution)
- A chat-capable model served via Ollama, vLLM, or any supported provider
- The `inline::sentence-transformers` inference provider enabled, with `trust_remote_code: true` in its config (required for `nomic-ai/nomic-embed-text-v1.5`, the default embedding model)
- A `vector_io` provider (FAISS works out of the box with the starter distribution)
- Node.js ≥ 18 (for the frontend)

### Llama Stack server config note

The sentence-transformers provider must have `trust_remote_code: true` in `~/.llama/distributions/starter/config.yaml`:

```yaml
- provider_id: sentence-transformers
  provider_type: inline::sentence-transformers
  config:
    trust_remote_code: true
```

## Running in development

**Terminal 1 — Llama Stack server:**

```bash
llama stack run ~/.llama/distributions/starter/config.yaml
```

**Terminal 2 — backend:**

```bash
# From the repo root
uv sync
cd demos/07-end-to-end-apps/knowledge_assistant
uv run uvicorn app:app --reload --port 8000
```

**Terminal 3 — frontend:**

```bash
cd demos/07-end-to-end-apps/knowledge_assistant/frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api` requests to the FastAPI backend at port 8000.

## Running in production

Build the React app and serve everything from FastAPI:

```bash
cd demos/07-end-to-end-apps/knowledge_assistant/frontend
npm run build               # outputs to frontend/dist/

cd ..
uv run uvicorn app:app --port 8000  # serves API + built frontend
```

Open http://localhost:8000.

## Usage

1. **Connect** — enter your Llama Stack server host and port, then click Connect
2. **Create a knowledge base** — give it a name (e.g. "Research Papers", "Product Docs")
3. **Add content** — select a target knowledge base, then either:
   - Upload `.txt`, `.md`, or `.pdf` files
   - Paste a URL to ingest a web page (text is extracted automatically)
4. **Select knowledge bases** — use the checkboxes to choose which ones to query
5. **Ask questions**
   - **1 KB selected** → answer grounded in that knowledge base
   - **2+ KBs selected** → each KB queried independently, then results synthesized

Each KB's files can be expanded in the sidebar to see ingestion status and delete individual documents.

## Architecture

```
frontend/src/
│  App.jsx                  state, API calls, KB list refresh
│  components/
│    Sidebar.jsx             connection, KB management, file/URL upload, file delete
│    Chat.jsx                message thread, input
│    Message.jsx             bubble, mode badge, expandable sources
│  api.js                   fetch + SSE client
│
app.py                      FastAPI backend
│  POST   /api/connect                      detect models, init orchestrator
│  GET    /api/knowledge-bases              list KBs with doc counts
│  POST   /api/knowledge-bases             create KB (vector store)
│  GET    /api/knowledge-bases/{name}/files list files with status
│  POST   /api/knowledge-bases/{name}/files upload and ingest files
│  DELETE /api/knowledge-bases/{name}/files/{id}  remove a file
│  POST   /api/knowledge-bases/{name}/urls  fetch URL, extract text, ingest
│  POST   /api/chat                         SSE stream: status → answer
│
agents/
  knowledge_agent.py        manages one KB: vector store CRUD, search, answer
  orchestrator.py           routes to single KB or runs parallel retrieval + synthesis
```

## Deployment notes

- Vector stores persist on the Llama Stack server between restarts — existing knowledge bases are automatically reloaded on connect
- The namespace prefix `ka::` is applied to all vector store names to avoid conflicts with other projects sharing the same Llama Stack server
- For production, replace FAISS with a persistent vector provider (Milvus, pgvector, Qdrant) and use a vLLM-hosted model
- See `kubernetes/` in the repo root for deployment manifests
