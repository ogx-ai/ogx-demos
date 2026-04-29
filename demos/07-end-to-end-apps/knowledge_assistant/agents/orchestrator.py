# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from llama_stack_client import LlamaStackClient

from .knowledge_agent import KnowledgeAgent


class KnowledgeOrchestrator:
    """Coordinates one or more KnowledgeAgents to answer questions.

    Single KB  → delegates directly to that agent.
    Multiple KBs → queries each agent in parallel, then synthesizes the results
                   with a dedicated synthesis call, making this a multi-agent workflow.
    """

    def __init__(
        self,
        client: LlamaStackClient,
        model_id: str,
        embedding_model: str,
        embedding_dimension: int,
        provider_id: str,
        namespace: str = "ka",
        shield_id: str | None = None,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.provider_id = provider_id
        self.namespace = namespace
        self.shield_id = shield_id
        self.agents: dict[str, KnowledgeAgent] = {}

    def _store_name(self, kb_name: str) -> str:
        return f"{self.namespace}::{kb_name}"

    def _make_agent(self, kb_name: str) -> KnowledgeAgent:
        return KnowledgeAgent(
            client=self.client,
            model_id=self.model_id,
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
            provider_id=self.provider_id,
            kb_name=kb_name,
            store_name=self._store_name(kb_name),
        )

    def load_existing_knowledge_bases(self) -> list[str]:
        """Discover this app's vector stores on the server and register them as agents."""
        prefix = f"{self.namespace}::"
        try:
            stores = self.client.vector_stores.list()
            for store in stores.data:
                if not store.name.startswith(prefix):
                    continue
                kb_name = store.name[len(prefix):]
                if kb_name in self.agents:
                    continue
                agent = self._make_agent(kb_name)
                agent.vector_store_id = store.id
                try:
                    files = list(
                        self.client.vector_stores.files.list(vector_store_id=store.id)
                    )
                    agent._file_count = len(files)
                except Exception:
                    pass
                conversation = self.client.conversations.create(metadata={"kb_name": kb_name})
                agent.conversation_id = conversation.id
                self.agents[kb_name] = agent
        except Exception:
            pass
        return list(self.agents.keys())

    def create_knowledge_base(self, kb_name: str) -> KnowledgeAgent:
        """Create a new knowledge base and register its agent."""
        if kb_name in self.agents:
            return self.agents[kb_name]
        agent = self._make_agent(kb_name)
        agent.initialize()
        self.agents[kb_name] = agent
        return agent

    def ingest_file(self, kb_name: str, file_name: str, file_content: bytes) -> None:
        """Add a document to the specified knowledge base, creating it if needed."""
        if kb_name not in self.agents:
            self.create_knowledge_base(kb_name)
        self.agents[kb_name].ingest_file(file_name, file_content)

    def delete_file(self, kb_name: str, file_id: str) -> None:
        """Remove a file from the specified knowledge base."""
        if kb_name not in self.agents:
            raise KeyError(kb_name)
        self.agents[kb_name].delete_file(file_id)

    def list_knowledge_bases(self) -> list[str]:
        return list(self.agents.keys())

    def query(self, question: str, kb_names: list[str]) -> dict:
        """Answer a question from the specified knowledge bases.

        Returns a dict with:
          - 'answer': the final answer string
          - 'mode': 'single' or 'multi'
          - 'sources': list of per-KB results (each has 'kb_name' and 'answer')
        """
        violation = self._check_safety(question)
        if violation:
            return {"answer": violation, "mode": "single", "sources": [], "blocked": True}

        active = [name for name in kb_names if name in self.agents]
        if not active:
            return {"answer": "No knowledge bases selected.", "mode": "single", "sources": []}

        if len(active) == 1:
            result = self.agents[active[0]].query(question)
            return {"answer": result["answer"], "mode": "single", "sources": [result]}

        # Multi-agent: query each KB independently, then synthesize
        results = [self.agents[name].query(question) for name in active]
        synthesized = self._synthesize(question, results)
        return {"answer": synthesized, "mode": "multi", "sources": results}

    def _synthesize(self, question: str, results: list[dict]) -> str:
        """Combine answers from multiple knowledge bases into a single response."""
        parts = "\n\n".join(
            f"From '{r['kb_name']}':\n{r['answer']}" for r in results
        )
        synthesis_input = (
            f"Question: {question}\n\n"
            f"Answers retrieved from multiple knowledge bases:\n{parts}\n\n"
            "Synthesize these into a single comprehensive answer. "
            "Clearly attribute each piece of information to its source knowledge base."
        )
        response = self.client.responses.create(
            model=self.model_id,
            instructions=(
                "You are a synthesis expert. Combine answers from multiple knowledge bases "
                "into a clear, well-structured response. Mention which knowledge base each "
                "piece of information comes from. Avoid repetition."
            ),
            input=[{"role": "user", "content": synthesis_input}],
            stream=False,
        )
        return response.output_text or ""

    def _check_safety(self, text: str) -> str | None:
        """Return a refusal message if the text violates the shield policy, else None."""
        if not self.shield_id:
            return None
        try:
            resp = self.client.safety.run_shield(
                shield_id=self.shield_id,
                messages=[{"role": "user", "content": text}],
            )
            if resp.violation:
                return resp.violation.user_message or "I can't help with that request."
        except Exception:
            pass
        return None
