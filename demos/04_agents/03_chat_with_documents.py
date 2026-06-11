# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Demo: Chat with Documents

Description:
This demo teaches how to build conversational agents that can answer questions about uploaded documents using vector search.

Learning Objectives:
- Upload and index documents in vector stores for agent access
- Perform vector search to retrieve relevant context for queries
- Inject retrieved context into agent instructions
- Build document-aware conversational agents
"""

import tempfile
import time
from pathlib import Path

import fire

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None
from ogx_client import OgxClient
from termcolor import colored

from demos.shared.utils import (
    build_context,
    download_documents,
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
        "https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/memory_optimizations.rst",
        "https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/chat.rst",
        "https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/llama3.rst",
        "https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/qat_finetune.rst",
        "https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/lora_finetune.rst",
    ]

    client = OgxClient(
        base_url=f"http://{host}:{port}",
    )

    if model_id is None:
        model_id = get_any_available_chat_model(client)
        if model_id is None:
            return
    else:
        if not check_model_is_available(client, model_id):
            return

    print(f"Using model: {model_id}")

    embedding_model = resolve_embedding_model(client, embedding_model_id)
    if embedding_model is None:
        return

    embedding_dimension = get_embedding_dimension(client, embedding_model)
    if embedding_dimension is None:
        print(colored("Unable to determine embedding dimension.", "red"))
        return

    vector_store = None
    file_ids: list[str] = []
    attached_file_ids: list[str] = []
    try:
        vector_store = client.vector_stores.create(
            name="chat-with-documents",
            extra_body={"embedding_model": embedding_model, "embedding_dimension": embedding_dimension},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_paths = download_documents(urls, Path(tmpdir))
            if not doc_paths:
                print(colored("No documents downloaded. Exiting.", "red"))
                return
            for doc_path in doc_paths:
                with open(doc_path, "rb") as doc_file:
                    file_response = client.files.create(file=doc_file, purpose="assistants")
                file_ids.append(file_response.id)

                file_attach_response = client.vector_stores.files.create(
                    vector_store_id=vector_store.id,
                    file_id=file_response.id,
                    chunking_strategy={
                        "type": "static",
                        "static": {
                            "max_chunk_size_tokens": 512,
                            "chunk_overlap_tokens": 64,
                        },
                    },
                )
                attach_start = time.time()
                while file_attach_response.status == "in_progress":
                    time.sleep(0.2)
                    if time.time() - attach_start > 30:
                        break
                    file_attach_response = client.vector_stores.files.retrieve(
                        vector_store_id=vector_store.id,
                        file_id=file_response.id,
                    )
                if file_attach_response.status != "completed":
                    error_message = None
                    if file_attach_response.last_error:
                        error_message = file_attach_response.last_error.message
                    if time.time() - attach_start > 30 and file_attach_response.status == "in_progress":
                        print(colored(f"Timed out attaching {doc_path.name}", "red"))
                    else:
                        print(
                            colored(
                                f"Failed to attach {doc_path.name}"
                                + (f": {error_message}" if error_message else ""),
                                "red",
                            )
                        )
                    continue
                attached_file_ids.append(file_response.id)

            if not attached_file_ids:
                print(colored("No documents attached to the vector store. Exiting.", "red"))
                return

        conversation = client.conversations.create(metadata={"name": "test-session"})
        session_id = conversation.id
        print(f"Created session_id={session_id}")

        user_prompts = [
            "I uploaded Torchtune documentation for reference.",
            "What are the top 5 topics that were explained? Only list succinct bullet points.",
            "Was anything related to 'Llama3' discussed, if so what?",
            "Tell me how to use LoRA",
            "What about Quantization?",
        ]

        for prompt in user_prompts:
            print(colored(f"User> {prompt}", "blue"))
            search_response = client.vector_stores.search(
                vector_store_id=vector_store.id,
                query=prompt,
                max_num_results=5,
            )
            context = build_context(search_response.data)
            instructions = "You are a helpful assistant."
            if context:
                instructions = f"{instructions}\n\n{context}"

            response = client.responses.create(
                model=model_id,
                conversation=session_id,
                instructions=instructions,
                input=[{"role": "user", "content": prompt}],
                stream=False,
            )
            print(response.output_text)
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
