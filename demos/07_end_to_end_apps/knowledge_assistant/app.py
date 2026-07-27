# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Knowledge Assistant — FastAPI backend

Exposes REST + SSE endpoints consumed by the React frontend.
The orchestrator is held in process state; one OGX server
connection per running backend process.

Run with:
    uv run uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

# This app runs via uvicorn from its own directory, not via python -m from the
# repo root, so we need the repo root on sys.path for demos.shared.utils.
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import httpx  # noqa: E402
import trafilatura  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from ogx_client import OgxClient  # noqa: E402
from pydantic import BaseModel  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from agents.orchestrator import KnowledgeOrchestrator  # noqa: E402

from demos.shared.utils import (  # noqa: E402
    _get_model_id,
    _is_llm_model,
    _list_models,
    get_embedding_dimension,
    resolve_embedding_model,
    resolve_model,
)

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Knowledge Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

_orchestrator: KnowledgeOrchestrator | None = None
_executor = ThreadPoolExecutor(max_workers=4)


# ── Request/response models ────────────────────────────────────────────────────


class ConnectRequest(BaseModel):
    host: str = "localhost"
    port: int = 8321
    model_id: str | None = None


class CreateKBRequest(BaseModel):
    name: str


class ChatRequest(BaseModel):
    question: str
    kb_names: list[str]


class AddURLRequest(BaseModel):
    url: str


# ── Helpers ────────────────────────────────────────────────────────────────────


def _require_orchestrator() -> KnowledgeOrchestrator:
    if _orchestrator is None:
        raise HTTPException(status_code=400, detail="Not connected to an OGX server.")
    return _orchestrator


def _kb_list(orc: KnowledgeOrchestrator) -> list[dict]:
    return [{"name": name, "doc_count": orc.agents[name].doc_count} for name in orc.list_knowledge_bases()]


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.post("/api/connect")
async def connect(req: ConnectRequest):
    global _orchestrator

    def _do_connect() -> tuple[KnowledgeOrchestrator, list[str]]:
        client = OgxClient(base_url=f"http://{req.host}:{req.port}")

        model_id = resolve_model(client, req.model_id)
        if not model_id:
            raise ValueError("No chat-capable model found on the server.")

        available_models = [
            _get_model_id(m)
            for m in _list_models(client)
            if _is_llm_model(m) and _get_model_id(m) and "guard" not in (_get_model_id(m) or "")
        ]

        embedding_model = resolve_embedding_model(client)
        if not embedding_model:
            raise ValueError("No embedding model found on the server.")

        embedding_dimension = get_embedding_dimension(client, embedding_model)
        if not embedding_dimension:
            raise ValueError("Could not determine embedding dimension.")

        providers = client.providers.list()
        vector_provider = next((p for p in providers if p.api == "vector_io"), None)
        if not vector_provider:
            raise ValueError("No vector_io provider found on the server.")

        orc = KnowledgeOrchestrator(
            client=client,
            model_id=model_id,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            provider_id=vector_provider.provider_id,
            namespace=os.getenv("KA_NAMESPACE", "ka"),
        )
        orc.load_existing_knowledge_bases()
        return orc, available_models

    loop = asyncio.get_running_loop()
    try:
        _orchestrator, available_models = await loop.run_in_executor(_executor, _do_connect)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {
        "model_id": _orchestrator.model_id,
        "embedding_model": _orchestrator.embedding_model,
        "knowledge_bases": _kb_list(_orchestrator),
        "available_models": available_models,
    }


@app.get("/api/knowledge-bases")
async def list_knowledge_bases():
    orc = _require_orchestrator()
    return _kb_list(orc)


@app.delete("/api/knowledge-bases/{kb_name}")
async def delete_knowledge_base(kb_name: str):
    orc = _require_orchestrator()
    if kb_name not in orc.agents:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_name}' not found.")
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(_executor, lambda: orc.delete_knowledge_base(kb_name))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"deleted": kb_name}


@app.get("/api/knowledge-bases/{kb_name}/files")
async def list_kb_files(kb_name: str):
    orc = _require_orchestrator()
    if kb_name not in orc.agents:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_name}' not found.")
    loop = asyncio.get_running_loop()
    files = await loop.run_in_executor(_executor, lambda: orc.agents[kb_name].list_files())
    return files


@app.post("/api/knowledge-bases")
async def create_knowledge_base(req: CreateKBRequest):
    orc = _require_orchestrator()
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Knowledge base name cannot be empty.")
    if "/" in name:
        raise HTTPException(status_code=422, detail="Knowledge base name cannot contain '/'.")
    if "::" in name:
        raise HTTPException(status_code=422, detail="Knowledge base name cannot contain '::'.")

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(_executor, lambda: orc.create_knowledge_base(req.name.strip()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {"name": req.name.strip(), "doc_count": 0}


@app.post("/api/knowledge-bases/{kb_name}/files")
async def upload_files(kb_name: str, files: list[UploadFile] = File(...)):  # noqa: B008 — FastAPI requires File() in signature
    orc = _require_orchestrator()
    if kb_name not in orc.agents:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_name}' not found.")

    loop = asyncio.get_running_loop()

    async def _ingest(f: UploadFile) -> None:
        content = await f.read()
        await loop.run_in_executor(
            _executor,
            lambda: orc.ingest_file(kb_name, f.filename, content),
        )

    try:
        await asyncio.gather(*[_ingest(f) for f in files])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {"ingested": len(files), "kb_name": kb_name}


@app.delete("/api/knowledge-bases/{kb_name}/files/{file_id}")
async def delete_file(kb_name: str, file_id: str):
    orc = _require_orchestrator()
    if kb_name not in orc.agents:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_name}' not found.")
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(_executor, lambda: orc.delete_file(kb_name, file_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"deleted": file_id, "kb_name": kb_name}


@app.post("/api/knowledge-bases/{kb_name}/urls")
async def add_url(kb_name: str, req: AddURLRequest):
    orc = _require_orchestrator()
    if kb_name not in orc.agents:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_name}' not found.")

    parsed = urlparse(req.url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=422, detail="Only http and https URLs are allowed.")

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            r = await client.get(req.url, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            html = r.text
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to fetch URL: {e}") from e

    text = trafilatura.extract(html, include_comments=False, include_tables=True)
    if not text:
        raise HTTPException(status_code=422, detail="Could not extract readable text from URL.")

    slug = parsed.path.rstrip("/").split("/")[-1] or parsed.netloc
    filename = f"{slug}.txt"

    content = text.encode("utf-8")
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(_executor, lambda: orc.ingest_file(kb_name, filename, content))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {"ingested": 1, "kb_name": kb_name, "filename": filename}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    orc = _require_orchestrator()
    loop = asyncio.get_running_loop()

    async def event_stream():
        count = len(req.kb_names)
        label = f"Searching {count} knowledge base{'s' if count > 1 else ''}…"
        yield f"data: {json.dumps({'type': 'status', 'message': label})}\n\n"

        try:
            result = await loop.run_in_executor(
                _executor,
                lambda: orc.query(req.question, req.kb_names),
            )
            payload = {
                "type": "answer",
                "content": result["answer"],
                "mode": result["mode"],
                "sources": result["sources"],
            }
            yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Serve built React app in production ────────────────────────────────────────

_dist = Path(__file__).parent / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
