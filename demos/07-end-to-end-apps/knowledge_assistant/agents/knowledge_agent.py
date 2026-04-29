# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from io import BytesIO

from llama_stack_client import LlamaStackClient


class KnowledgeAgent:
    """Manages a single knowledge base and answers questions from it using RAG."""

    def __init__(
        self,
        client: LlamaStackClient,
        model_id: str,
        embedding_model: str,
        embedding_dimension: int,
        provider_id: str,
        kb_name: str,
        store_name: str,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.provider_id = provider_id
        self.kb_name = kb_name        # display name shown to the user
        self._store_name = store_name  # actual vector store name (namespaced)
        self.vector_store_id: str | None = None
        self.conversation_id: str | None = None
        self._file_count: int = 0

    @property
    def is_ready(self) -> bool:
        return self.vector_store_id is not None

    @property
    def doc_count(self) -> int:
        return self._file_count

    def initialize(self) -> None:
        """Find an existing vector store for this KB, or create a new one."""
        try:
            stores = self.client.vector_stores.list()
            for store in stores.data:
                if store.name == self._store_name:
                    self.vector_store_id = store.id
                    break
        except Exception:
            pass

        if self.vector_store_id is None:
            store = self.client.vector_stores.create(
                name=self._store_name,
                extra_body={
                    "provider_id": self.provider_id,
                    "embedding_model": self.embedding_model,
                    "embedding_dimension": self.embedding_dimension,
                },
            )
            self.vector_store_id = store.id
        else:
            # Count existing files in the vector store
            try:
                files = list(
                    self.client.vector_stores.files.list(vector_store_id=self.vector_store_id)
                )
                self._file_count = len(files)
            except Exception:
                pass

        conversation = self.client.conversations.create(metadata={"kb_name": self.kb_name})
        self.conversation_id = conversation.id

    def ingest_file(self, file_name: str, file_content: bytes) -> None:
        """Add a document to this knowledge base."""
        if not self.is_ready:
            self.initialize()

        file_buffer = BytesIO(file_content)
        file_buffer.name = file_name

        uploaded = self.client.files.create(file=file_buffer, purpose="assistants")
        self.client.vector_stores.files.create(
            vector_store_id=self.vector_store_id,
            file_id=uploaded.id,
            chunking_strategy={
                "type": "static",
                "static": {"max_chunk_size_tokens": 512, "chunk_overlap_tokens": 64},
            },
        )
        self._file_count += 1

    def list_files(self) -> list[dict]:
        """Return metadata for all files in this knowledge base."""
        if not self.is_ready:
            return []
        try:
            vs_files = list(
                self.client.vector_stores.files.list(vector_store_id=self.vector_store_id)
            )
            result = []
            for vsf in vs_files:
                try:
                    info = self.client.files.retrieve(file_id=vsf.id)
                    result.append({
                        "id": vsf.id,
                        "name": info.filename,
                        "status": vsf.status,
                        "bytes": info.bytes,
                    })
                except Exception:
                    result.append({"id": vsf.id, "name": vsf.id, "status": vsf.status, "bytes": 0})
            return result
        except Exception:
            return []

    def delete_file(self, file_id: str) -> None:
        """Remove a file from this knowledge base and delete it from the files store."""
        if not self.is_ready:
            return
        self.client.vector_stores.files.delete(
            vector_store_id=self.vector_store_id,
            file_id=file_id,
        )
        try:
            self.client.files.delete(file_id=file_id)
        except Exception:
            pass
        self._file_count = max(0, self._file_count - 1)

    def query(self, question: str) -> dict:
        """Answer a question using documents in this knowledge base.

        Returns a dict with 'answer' and 'kb_name'.
        """
        if not self.is_ready:
            self.initialize()

        # Step 1: retrieve relevant chunks directly from the vector store.
        # Avoids relying on the model's tool-calling ability (unreliable with small models).
        try:
            search_resp = self.client.vector_stores.search(
                vector_store_id=self.vector_store_id,
                query=question,
                max_num_results=5,
            )
            chunks = search_resp.data or []
        except Exception:
            chunks = []

        if chunks:
            parts = []
            for chunk in chunks:
                text = " ".join(c.text for c in chunk.content if c.text)
                if text:
                    parts.append(f"[{chunk.filename}]\n{text}")
            context = "\n\n---\n\n".join(parts)
        else:
            context = ""

        # Step 2: ask the LLM with context embedded in the prompt.
        if context:
            user_content = f"Context:\n{context}\n\nQuestion: {question}"
            instructions = (
                f"You are a helpful assistant for the '{self.kb_name}' knowledge base. "
                "Answer questions based ONLY on the provided context. "
                "Be concise. If the answer is not in the context, say so clearly."
            )
        else:
            user_content = question
            instructions = (
                f"You are a helpful assistant for the '{self.kb_name}' knowledge base. "
                "No relevant documents were found for this question. "
                "Tell the user clearly that no relevant information was found in the knowledge base."
            )

        response = self.client.responses.create(
            model=self.model_id,
            instructions=instructions,
            input=[{"role": "user", "content": user_content}],
            conversation=self.conversation_id,
            stream=False,
        )

        return {
            "answer": response.output_text or "",
            "kb_name": self.kb_name,
        }
