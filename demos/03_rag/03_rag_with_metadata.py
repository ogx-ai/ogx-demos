"""
Demo: RAG with Metadata

Description:
This demo teaches how to use metadata filtering to retrieve documents from specific sources or with specific attributes.

Learning Objectives:
- Attach metadata attributes to documents in vector stores
- Filter search results using metadata filters
- Control which documents are retrieved based on attributes
- Build source-aware RAG systems
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
    can_model_chat,
    check_model_is_available,
    get_any_available_chat_model,
    resolve_embedding_model,
    get_embedding_dimension,
)

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


def _maybe_load_dotenv() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _attach_text(
    client: OgxClient,
    vector_store_id: str,
    text: str,
    filename: str,
    attributes: dict[str, str],
) -> None:
    # Upload inline text as a file and attach it to the vector store with attributes.
    file_buffer = BytesIO(text.encode("utf-8"))
    file_buffer.name = filename
    uploaded_file = client.files.create(file=file_buffer, purpose="assistants")
    client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=uploaded_file.id,
        attributes=attributes,
        chunking_strategy={
            "type": "static",
            "static": {"max_chunk_size_tokens": 256, "chunk_overlap_tokens": 32},
        },
    )


def main(
    host: str,
    port: int,
    model_id: str | None = None,
    embedding_model_id: str | None = None,
    question: str = "What does OGX provide?",
    source: str = "doc_a",
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

    vector_store = None
    try:
        vector_store = client.vector_stores.create(
            name=f"rag_metadata_{uuid4()}",
            extra_body={
                "provider_id": provider_id,
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
            },
        )

        # Attach two documents with different metadata attributes.
        _attach_text(
            client,
            vector_store.id,
            "OGX provides a unified API for models, tools, and vector stores.",
            "doc_a.txt",
            {"source": "doc_a"},
        )
        _attach_text(
            client,
            vector_store.id,
            "OGX supports serving models and building agentic workflows.",
            "doc_b.txt",
            {"source": "doc_b"},
        )

        response = client.responses.create(
            model=resolved_model,
            instructions="Use file_search with metadata filters to answer the question.",
            input=[{"role": "user", "content": question}],
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store.id],
                    "filters": {"type": "eq", "key": "source", "value": source},
                }
            ],
            tool_choice={"type": "file_search"},
            include=["file_search_call.results"],
            stream=False,
        )

        print(f"[context] filter source: {source}")
        print(f"[context] question: {question}")
        print(f"[response] {response.output_text or response.output}")
    finally:
        if vector_store is not None:
            try:
                client.vector_stores.delete(vector_store_id=vector_store.id)
            except Exception as e:
                print(colored(f"Warning: Failed to delete vector store: {e}", "yellow"))


if __name__ == "__main__":
    fire.Fire(main)
