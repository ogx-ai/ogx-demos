"""
Demo: Chunking Strategies

Description:
This demo teaches how different chunking strategies affect retrieval quality and response generation in RAG systems.

Learning Objectives:
- Configure static chunking strategies with different parameters
- Compare small vs. large chunk sizes for retrieval
- Understand the impact of chunk overlap on retrieval quality
- Choose appropriate chunking strategies for different use cases
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


def _attach_with_chunking(
    client: OgxClient,
    vector_store_id: str,
    text: str,
    filename: str,
    chunking_strategy: dict,
) -> None:
    # Upload inline text as a file and attach it with the chosen chunking strategy.
    file_buffer = BytesIO(text.encode("utf-8"))
    file_buffer.name = filename
    uploaded_file = client.files.create(file=file_buffer, purpose="assistants")
    client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=uploaded_file.id,
        chunking_strategy=chunking_strategy,
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
    # Select a chat-capable model for the RAG response.
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

    # Create two vector stores with the same document but different chunk sizes.
    store_small = None
    store_large = None
    try:
        store_small = client.vector_stores.create(
            name=f"chunk_small_{uuid4()}",
            extra_body={
                "provider_id": provider_id,
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
            },
        )
        store_large = client.vector_stores.create(
            name=f"chunk_large_{uuid4()}",
            extra_body={
                "provider_id": provider_id,
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
            },
        )

        # Same document content for both stores to isolate chunking differences.
        # Using a longer document to make chunking differences more apparent.
        doc_text = (
            "OGX provides a unified API for models, tools, and vector stores. "
            "It supports serving, evaluation, and agentic workflows. "
            "You can build RAG systems and multi-tool agents with it. "
            "The framework enables developers to create sophisticated AI applications "
            "by combining language models with external tools and knowledge bases. "
            "Vector stores allow efficient semantic search over large document collections, "
            "while the tool integration system lets agents interact with external services. "
            "RAG (Retrieval-Augmented Generation) systems can ground model responses in factual data. "
            "The unified API design makes it easy to switch between different model providers "
            "and deployment configurations without changing application code. "
            "Agentic workflows enable autonomous task completion through iterative reasoning and tool use."
        )

        _attach_with_chunking(
            client,
            store_small.id,
            doc_text,
            "chunk_small.txt",
            {"type": "static", "static": {"max_chunk_size_tokens": 128, "chunk_overlap_tokens": 16}},
        )
        _attach_with_chunking(
            client,
            store_large.id,
            doc_text,
            "chunk_large.txt",
            {"type": "static", "static": {"max_chunk_size_tokens": 512, "chunk_overlap_tokens": 64}},
        )

        # Query the store with smaller chunks.
        response_small = client.responses.create(
            model=resolved_model,
            instructions="Use file_search and answer briefly.",
            input=[{"role": "user", "content": question}],
            tools=[{"type": "file_search", "vector_store_ids": [store_small.id]}],
            tool_choice={"type": "file_search"},
            include=["file_search_call.results"],
            stream=False,
        )
        # Query the store with larger chunks.
        response_large = client.responses.create(
            model=resolved_model,
            instructions="Use file_search and answer briefly.",
            input=[{"role": "user", "content": question}],
            tools=[{"type": "file_search", "vector_store_ids": [store_large.id]}],
            tool_choice={"type": "file_search"},
            include=["file_search_call.results"],
            stream=False,
        )

        print(f"[context] question: {question}")
        print(f"[small chunks] store={store_small.id}")
        print(f"[small chunks] response: {response_small.output_text or response_small.output}")
        print(f"[large chunks] store={store_large.id}")
        print(f"[large chunks] response: {response_large.output_text or response_large.output}")
    finally:
        if store_small is not None:
            try:
                client.vector_stores.delete(vector_store_id=store_small.id)
            except Exception as e:
                print(colored(f"Warning: Failed to delete small chunks vector store: {e}", "yellow"))
        if store_large is not None:
            try:
                client.vector_stores.delete(vector_store_id=store_large.id)
            except Exception as e:
                print(colored(f"Warning: Failed to delete large chunks vector store: {e}", "yellow"))


if __name__ == "__main__":
    fire.Fire(main)
