"""
Demo: Insert Documents

Description:
This demo teaches how to insert multiple documents into a vector store from local files or URLs, with chunking strategies.

Learning Objectives:
- Upload documents from local directories or remote URLs
- Configure chunking strategies for document processing
- Handle asynchronous document attachment with status polling
- Manage vector store file attachments with error handling
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from uuid import uuid4

import fire
from ogx_client import OgxClient
from termcolor import colored

from demos.shared.utils import (
    download_documents,
    resolve_embedding_model,
    get_embedding_dimension,
)

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


DEFAULT_URLS = [
    "https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/memory_optimizations.rst",
    "https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/chat.rst",
    "https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/llama3.rst",
    "https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/qat_finetune.rst",
    "https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/lora_finetune.rst",
]


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


def _collect_local_files(file_dir: str) -> list[Path]:
    path = Path(file_dir).expanduser()
    if not path.exists() or not path.is_dir():
        print(colored(f"Directory not found: {file_dir}", "red"))
        return []
    files = [p for p in path.rglob("*") if p.is_file()]
    if not files:
        print(colored(f"No files found under: {file_dir}", "red"))
        return []
    print(f"Found {len(files)} files under: {file_dir}")
    return files


def _attach_file(
    client: OgxClient,
    vector_store_id: str,
    file_id: str,
    filename: str,
    timeout_seconds: float = 30,
) -> bool:
    response = client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=file_id,
        chunking_strategy={
            "type": "static",
            "static": {"max_chunk_size_tokens": 512, "chunk_overlap_tokens": 64},
        },
    )
    start_time = time.time()
    while response.status == "in_progress":
        time.sleep(0.2)
        if time.time() - start_time > timeout_seconds:
            break
        response = client.vector_stores.files.retrieve(
            vector_store_id=vector_store_id,
            file_id=file_id,
        )
    if response.status == "completed":
        return True
    error_message = response.last_error.message if response.last_error else None
    if time.time() - start_time > timeout_seconds and response.status == "in_progress":
        print(colored(f"Timed out attaching {filename}", "red"))
    else:
        print(
            colored(
                f"Failed to attach {filename}" + (f": {error_message}" if error_message else ""),
                "red",
            )
        )
    return False


def main(
    host: str,
    port: int,
    vector_store_id: str | None = None,
    embedding_model_id: str | None = None,
    provider_id: str | None = None,
    file_dir: str | None = None,
    urls: str | None = None,
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

    if vector_store_id is None:
        provider = _get_vector_provider(client, provider_id)
        if provider is None:
            return
        vector_store = client.vector_stores.create(
            name=f"insert-documents-{uuid4()}",
            extra_body={
                "provider_id": provider.provider_id,
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
            },
        )
        vector_store_id = vector_store.id
        print(f"Created vector store: {vector_store_id}")
    else:
        print(f"Using existing vector store: {vector_store_id}")

    file_paths: list[Path] = []
    if file_dir:
        file_paths = _collect_local_files(file_dir)
    if not file_paths:
        url_list = [u.strip() for u in (urls.split(",") if urls else DEFAULT_URLS) if u.strip()]
        with tempfile.TemporaryDirectory() as tmpdir:
            file_paths = download_documents(url_list, Path(tmpdir))
            if not file_paths:
                print(colored("No documents available to upload.", "red"))
                return
            for file_path in file_paths:
                with open(file_path, "rb") as doc_file:
                    file_response = client.files.create(file=doc_file, purpose="assistants")
                if _attach_file(client, vector_store_id, file_response.id, file_path.name):
                    print(f"Attached: {file_path.name} -> {file_response.id}")
            print(f"Done. Vector store ready: {vector_store_id}")
            return

    for file_path in file_paths:
        with open(file_path, "rb") as doc_file:
            file_response = client.files.create(file=doc_file, purpose="assistants")
        if _attach_file(client, vector_store_id, file_response.id, file_path.name):
            print(f"Attached: {file_path.name} -> {file_response.id}")

    print(f"Done. Vector store ready: {vector_store_id}")


if __name__ == "__main__":
    fire.Fire(main)
