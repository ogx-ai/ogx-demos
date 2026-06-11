"""
Demo: Multi-Source RAG

Description:
This demo teaches how to query across multiple vector stores to combine information from different document sources.

Learning Objectives:
- Create and manage multiple vector stores
- Attach different documents to separate vector stores
- Query across multiple vector stores simultaneously with file_search
- Combine information from diverse document sources
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
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


def _maybe_load_dotenv() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _create_vector_store(
    client: OgxClient, provider_id: str, embedding_model: str, embedding_dimension: int
):
    # Each vector store can hold its own document set.
    return client.vector_stores.create(
        name=f"multi_source_{uuid4()}",
        extra_body={
            "provider_id": provider_id,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
        },
    )


def _attach_text(
    client: OgxClient, vector_store_id: str, text: str, filename: str
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


def main(
    host: str,
    port: int,
    model_id: str | None = None,
    embedding_model_id: str | None = None,
    question: str = "What does OGX provide?",
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

    # Create two vector stores to simulate multiple sources.
    store_a = None
    store_b = None
    try:
        store_a = _create_vector_store(client, provider_id, embedding_model, embedding_dimension)
        store_b = _create_vector_store(client, provider_id, embedding_model, embedding_dimension)

        # Attach different content to each store.
        _attach_text(
            client,
            store_a.id,
            "OGX provides a unified API for models, tools, and vector stores.",
            "doc_a.txt",
        )
        _attach_text(
            client,
            store_b.id,
            "OGX supports serving models and building agentic workflows.",
            "doc_b.txt",
        )

        # Query across both vector stores with file_search.
        response = client.responses.create(
            model=resolved_model,
            instructions="Use file_search across all provided vector stores.",
            input=[{"role": "user", "content": question}],
            tools=[{"type": "file_search", "vector_store_ids": [store_a.id, store_b.id]}],
            tool_choice={"type": "file_search"},
            include=["file_search_call.results"],
            stream=False,
        )

        print(f"[context] question: {question}")
        print(f"[context] stores: {store_a.id}, {store_b.id}")
        print(f"[response] {response.output_text or response.output}")
    finally:
        if store_a is not None:
            try:
                client.vector_stores.delete(vector_store_id=store_a.id)
            except Exception as e:
                print(colored(f"Warning: Failed to delete vector store A: {e}", "yellow"))
        if store_b is not None:
            try:
                client.vector_stores.delete(vector_store_id=store_b.id)
            except Exception as e:
                print(colored(f"Warning: Failed to delete vector store B: {e}", "yellow"))


if __name__ == "__main__":
    fire.Fire(main)
