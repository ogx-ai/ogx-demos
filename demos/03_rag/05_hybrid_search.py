"""
Demo: Hybrid Search

Description:
This demo teaches how to combine file_search (local vector store) and web_search (real-time web data) for comprehensive answers.

Learning Objectives:
- Combine multiple search tools in a single workflow
- Retrieve context from both local documents and the web
- Synthesize information from diverse sources
- Build hybrid RAG systems with local and external knowledge
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from io import BytesIO
import os
from uuid import uuid4

import fire
from ogx_client import OgxClient
from termcolor import colored

from demos.shared.utils import (
    build_context_from_dicts,
    can_model_chat,
    check_model_is_available,
    get_any_available_chat_model,
    resolve_embedding_model,
    get_embedding_dimension,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


def _maybe_load_dotenv() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _attach_text(
    client: OgxClient,
    vector_store_id: str,
    text: str,
    filename: str,
) -> None:
    # Upload inline text as a file and attach it to the vector store.
    file_buffer = BytesIO(text.encode("utf-8"))
    file_buffer.name = filename
    uploaded_file = client.files.create(file=file_buffer, purpose="assistants")
    client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=uploaded_file.id,
        chunking_strategy={
            "type": "static",
            "static": {"max_chunk_size_tokens": 256, "chunk_overlap_tokens": 32},
        },
    )


def _to_dict(value) -> dict | None:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return None


def _print_file_search_results(results: list) -> None:
    if not results:
        print("[sources] file_search returned no results")
        return
    print(f"[sources] file_search results: {len(results)}")
    for result in results[:3]:
        result_dict = _to_dict(result) or {}
        filename = result_dict.get("filename", "unknown")
        score = result_dict.get("score", "n/a")
        content = result_dict.get("content") or []
        snippet_parts = []
        for chunk in content:
            chunk_dict = _to_dict(chunk) or {}
            text = chunk_dict.get("text")
            if text:
                snippet_parts.append(text.strip())
        snippet = " ".join(snippet_parts).strip()
        if snippet:
            snippet = snippet[:200] + ("..." if len(snippet) > 200 else "")
        print(f"- {filename} (score={score}): {snippet or 'no text'}")


def _print_web_search_sources(sources: list) -> None:
    if not sources:
        print("[sources] web_search returned no sources")
        return
    print(f"[sources] web_search sources: {len(sources)}")
    for source in sources[:3]:
        source_dict = _to_dict(source) or {}
        title = source_dict.get("title") or "untitled"
        url = source_dict.get("url") or source_dict.get("link") or "unknown"
        print(f"- {title} ({url})")


def _extract_file_search_results(response) -> list:
    output_items = getattr(response, "output", None) or []
    if isinstance(output_items, dict):
        output_items = [output_items]
    if not isinstance(output_items, list):
        return []
    for item in output_items:
        item_dict = _to_dict(item) or {}
        if item_dict.get("type") == "file_search_call":
            return item_dict.get("results") or item_dict.get("file_search_call", {}).get("results") or []
    return []


def _extract_web_search_sources(response) -> list:
    output_items = getattr(response, "output", None) or []
    if isinstance(output_items, dict):
        output_items = [output_items]
    if not isinstance(output_items, list):
        return []
    for item in output_items:
        item_dict = _to_dict(item) or {}
        if item_dict.get("type") == "web_search_call":
            action = item_dict.get("action") or item_dict.get("web_search_call", {}).get("action") or {}
            return action.get("sources") or []
    return []


def _summarize_output_item(item: dict) -> str:
    item_type = item.get("type", "unknown")
    keys = ", ".join(sorted(item.keys()))
    return f"{item_type} (keys: {keys})"


def _print_tool_outputs(response) -> bool:
    output_items = getattr(response, "output", None) or []
    if isinstance(output_items, dict):
        output_items = [output_items]
    if not isinstance(output_items, list):
        return False
    found = False
    for item in output_items:
        item_dict = _to_dict(item) or {}
        item_type = item_dict.get("type")
        if item_type == "file_search_call":
            results = item_dict.get("results") or item_dict.get("file_search_call", {}).get("results") or []
            _print_file_search_results(results)
            found = True
        elif item_type == "web_search_call":
            action = item_dict.get("action") or item_dict.get("web_search_call", {}).get("action") or {}
            sources = action.get("sources") or []
            _print_web_search_sources(sources)
            found = True
        elif item_type == "message":
            content = item_dict.get("content") or []
            text_parts = []
            for part in content:
                part_dict = _to_dict(part) or {}
                text = part_dict.get("text")
                if text:
                    text_parts.append(text.strip())
            if text_parts:
                print("[sources] no tool calls returned; model responded with text only")
    if not found:
        summaries = []
        for item in output_items:
            item_dict = _to_dict(item) or {}
            summaries.append(_summarize_output_item(item_dict))
        if summaries:
            print(f"[sources] no tool outputs found; output items: {summaries}")
    return found


def main(
    host: str,
    port: int,
    model_id: str | None = None,
    embedding_model_id: str | None = None,
    question: str = "What does OGX provide and what are recent updates?",
) -> None:
    _maybe_load_dotenv()

    client = OgxClient(base_url=f"http://{host}:{port}")
    resolved_model = model_id or os.getenv("OGX_MODEL") or None
    if resolved_model is None:
        resolved_model = get_any_available_chat_model(client)
        if resolved_model is None:
            return
    else:
        if not check_model_is_available(client, resolved_model):
            return
        if not can_model_chat(client, resolved_model):
            print(
                colored(
                    f"Model `{resolved_model}` does not support chat. Choose a chat-capable model.",
                    "red",
                )
            )
            return

    embedding_model = resolve_embedding_model(client, embedding_model_id)
    if embedding_model is None:
        return

    embedding_dimension = get_embedding_dimension(client, embedding_model)
    if embedding_dimension is None:
        print(colored("Unable to determine embedding dimension.", "red"))
        return

    vector_providers = [provider for provider in client.providers.list() if provider.api == "vector_io"]
    if not vector_providers:
        print(colored("No available vector_io providers. Exiting.", "red"))
        return
    provider_id = vector_providers[0].provider_id

    # Create a temporary vector store so file_search has local context to retrieve.
    vector_store = None
    try:
        vector_store = client.vector_stores.create(
            name=f"hybrid_search_{uuid4()}",
            extra_body={
                "provider_id": provider_id,
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
            },
        )

        _attach_text(
            client,
            vector_store.id,
            "OGX provides a unified API for models, tools, and vector stores.",
            "hybrid_doc.txt",
        )

        # Step 1: force a file_search call to get local vector-store context.
        file_search_response = client.responses.create(
            model=resolved_model,
            instructions="Use file_search and return results for the question.",
            input=[{"role": "user", "content": question}],
            tools=[{"type": "file_search", "vector_store_ids": [vector_store.id]}],
            tool_choice={"type": "file_search"},
            # Include tool outputs so you can inspect retrieved chunks and web sources.
            include=["file_search_call.results"],
            stream=False,
        )

        # Step 2: force a web_search call to fetch recent sources.
        web_search_response = client.responses.create(
            model=resolved_model,
            instructions="Use web_search and return sources for the question.",
            input=[{"role": "user", "content": question}],
            tools=[{"type": "web_search"}],
            tool_choice={"type": "web_search"},
            include=["web_search_call.action.sources"],
            stream=False,
        )

        file_results = _extract_file_search_results(file_search_response)
        web_sources = _extract_web_search_sources(web_search_response)
        local_context = build_context_from_dicts(file_results)
        web_context_lines = []
        for source in web_sources[:5]:
            source_dict = _to_dict(source) or {}
            title = source_dict.get("title") or "untitled"
            url = source_dict.get("url") or source_dict.get("link") or "unknown"
            web_context_lines.append(f"- {title} ({url})")
        web_context = "\n".join(web_context_lines)

        # Step 3: synthesize one answer using both contexts.
        final_response = client.responses.create(
            model=resolved_model,
            instructions=(
                "Answer the question using the local context and web sources. "
                "If a source is missing, say so briefly."
            ),
            input=[
                {
                    "role": "user",
                    "content": (
                        f"{local_context}\n\nWeb sources:\n{web_context}\n\nQuestion: {question}"
                    ),
                }
            ],
            stream=False,
        )

        print(f"[context] question: {question}")
        print(f"[context] vector_store: {vector_store.id}")
        print("[note] file_search uses the local vector store; web_search uses the server tool_runtime.")
        _print_tool_outputs(file_search_response)
        _print_tool_outputs(web_search_response)
        print(f"[response] {final_response.output_text or final_response.output}")
    finally:
        if vector_store is not None:
            try:
                client.vector_stores.delete(vector_store_id=vector_store.id)
            except Exception as e:
                print(colored(f"Warning: Failed to delete vector store: {e}", "yellow"))


if __name__ == "__main__":
    fire.Fire(main)
