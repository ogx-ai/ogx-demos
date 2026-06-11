# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Demo: RAG Agent

Description:
This demo teaches how to build agents that use the file_search tool for retrieval-augmented generation (RAG).

Learning Objectives:
- Create agents with file_search tool capabilities
- Upload and index documents for agent-driven retrieval
- Enable agents to automatically search documents for answers
- Build knowledge-grounded conversational agents
"""

from io import BytesIO
from urllib.parse import urlparse
from urllib.request import urlopen
from uuid import uuid4

import fire

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None
import time
from ogx_client import Agent, AgentEventLogger, OgxClient
from termcolor import colored

from demos.shared.utils import (
    can_model_chat,
    check_model_is_available,
    get_any_available_chat_model,
    resolve_embedding_model,
    get_embedding_dimension,
)


def main(
    host: str,
    port: int,
    model_id: str | None = None,
    embedding_model_id: str | None = None,
):
    if load_dotenv is not None:
        load_dotenv()
    urls = [
        "memory_optimizations.rst",
        "chat.rst",
        "llama3.rst",
        "qat_finetune.rst",
        "lora_finetune.rst",
    ]
    document_urls = [
        f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}"
        for url in urls
    ]

    client = OgxClient(base_url=f"http://{host}:{port}")

    if model_id is None:
        model_id = get_any_available_chat_model(client)
        if model_id is None:
            return
    else:
        if not check_model_is_available(client, model_id):
            return
        if not can_model_chat(client, model_id):
            print(
                colored(
                    f"Model `{model_id}` does not support chat. Choose a chat-capable model.",
                    "red",
                )
            )
            return

    print(f"Using model: {model_id}")

    embedding_model = resolve_embedding_model(client, embedding_model_id)
    if embedding_model is None:
        return

    embedding_dimension = get_embedding_dimension(client, embedding_model)
    if embedding_dimension is None:
        print(colored("Unable to determine embedding dimension.", "red"))
        return

    vector_providers = [
        provider for provider in client.providers.list() if provider.api == "vector_io"
    ]
    if not vector_providers:
        print(colored("No available vector_io providers. Exiting.", "red"))
        return

    selected_vector_provider = vector_providers[0]

    # Create a vector store
    vector_store = None
    file_ids = []
    try:
        vector_store = client.vector_stores.create(
            name=f"test_vector_store_{uuid4()}",
            extra_body={
                "provider_id": selected_vector_provider.provider_id,
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
            },
        )

        # Upload and attach documents to the vector store
        start_time = time.time()
        for i, url in enumerate(document_urls):
            try:
                with urlopen(url, timeout=30) as response:
                    file_buffer = BytesIO(response.read())
            except Exception as e:
                print(colored(f"Failed to download {url}: {e}", "red"))
                continue
            filename = urlparse(url).path.rsplit("/", 1)[-1] or f"document-{i}.txt"
            file_buffer.name = filename
            uploaded_file = client.files.create(file=file_buffer, purpose="assistants")
            file_ids.append(uploaded_file.id)
            client.vector_stores.files.create(
                vector_store_id=vector_store.id,
                file_id=uploaded_file.id,
                attributes={"document_id": f"num-{i}", "source": url},
                chunking_strategy={
                    "type": "static",
                    "static": {"max_chunk_size_tokens": 512, "chunk_overlap_tokens": 128},
                },
            )
        end_time = time.time()
        print(colored(f"Inserted documents in {end_time - start_time:.2f}s", "cyan"))

        agent = Agent(
            client,
            model=model_id,
            instructions="You are a helpful assistant. Use file_search tool to gather information needed to answer questions. Answer succinctly.",
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store.id],
                }
            ],
        )
        session_id = agent.create_session("test-session")
        print(f"Created session_id={session_id}")

        user_prompts = [
            "Is anything related to 'Llama3' mentioned, if so what?",
            "Tell me how to use LoRA",
            "What about Quantization?",
        ]

        for prompt in user_prompts:
            response = agent.create_turn(
                messages=[{"role": "user", "content": prompt}],
                session_id=session_id,
            )
            print(colored(f"User> {prompt}", "blue"))
            for printable in AgentEventLogger().log(response):
                print(printable, end="", flush=True)
    finally:
        if vector_store is not None:
            try:
                client.vector_stores.delete(vector_store_id=vector_store.id)
            except Exception as e:
                print(colored(f"Warning: Failed to delete vector store: {e}", "yellow"))
        for file_id in file_ids:
            try:
                client.files.delete(file_id=file_id)
            except Exception as e:
                print(colored(f"Warning: Failed to delete file {file_id}: {e}", "yellow"))


if __name__ == "__main__":
    fire.Fire(main)
