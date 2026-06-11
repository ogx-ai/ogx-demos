"""
Demo: Vector Database Basics

Description:
This demo teaches the fundamentals of creating vector stores, uploading documents, and performing semantic search.

Learning Objectives:
- Create and configure a vector store with embeddings
- Upload text files to a vector store
- Perform semantic search with natural language queries
- Understand embedding models and vector providers
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


def _get_vector_provider(client: OgxClient, provider_id: str | None):
    vector_providers = [
        provider for provider in client.providers.list() if provider.api == "vector_io"
    ]
    if not vector_providers:
        print(colored("No available vector_io providers. Exiting.", "red"))
        return None
    if provider_id is None:
        print(f"Using default vector provider: {vector_providers[0].provider_id}")
        return vector_providers[0]
    for provider in vector_providers:
        if provider.provider_id == provider_id:
            print(f"Using specified vector provider: {provider.provider_id}")
            return provider
    available = [provider.provider_id for provider in vector_providers]
    print(colored(f"Vector provider `{provider_id}` not found. Available: {available}", "red"))
    return None


def _print_top_hit(search_response) -> None:
    if not search_response.data:
        print("No results found.")
        return
    top = search_response.data[0]
    text = top.content[0].text if top.content else ""
    print("Top match:")
    print(f"- score: {top.score:.4f}")
    print(f"- file: {top.filename}")
    print(f"- text: {text}")


def main(
    host: str,
    port: int,
    text: str = "OGX provides a unified API for building AI applications.",
    query: str = "What does OGX provide?",
    embedding_model_id: str | None = None,
    provider_id: str | None = None,
) -> None:
    _maybe_load_dotenv()

    client = OgxClient(base_url=f"http://{host}:{port}")
    embedding_model = resolve_embedding_model(client, embedding_model_id)
    if embedding_model is None:
        return

    embedding_dimension = get_embedding_dimension(client, embedding_model)
    if embedding_dimension is None:
        print(colored("Unable to determine embedding dimension.", "red"))
        return

    selected_provider = _get_vector_provider(client, provider_id)
    if selected_provider is None:
        return

    vector_store = client.vector_stores.create(
        name=f"vector-db-basics-{uuid4()}",
        extra_body={
            "provider_id": selected_provider.provider_id,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
        },
    )
    print(f"Vector store created: {vector_store.id}")

    file_buffer = BytesIO(text.encode("utf-8"))
    file_buffer.name = "vector-db-basics.txt"
    uploaded_file = client.files.create(file=file_buffer, purpose="assistants")
    print(f"File uploaded: {uploaded_file.id}")

    client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=uploaded_file.id,
        attributes={"source": "inline-text"},
        chunking_strategy={
            "type": "static",
            "static": {"max_chunk_size_tokens": 256, "chunk_overlap_tokens": 32},
        },
    )
    print(f"File attached to vector store: {vector_store.id}")

    search_response = client.vector_stores.search(
        vector_store_id=vector_store.id,
        query=query,
        max_num_results=3,
    )
    _print_top_hit(search_response)


if __name__ == "__main__":
    fire.Fire(main)
