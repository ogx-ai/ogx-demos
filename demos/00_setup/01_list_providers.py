"""
Demo: List Providers and Models

Description:
This demo shows how to inspect the providers and models available on your OGX server.
Use it to verify your server is configured correctly before running other demos.

Learning Objectives:
- List all configured inference providers
- List all available models and their types (LLM, embedding, rerank)
- Understand how providers map to models
"""

from __future__ import annotations

import fire
from ogx_client import OgxClient
from termcolor import colored

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def main(host: str = "localhost", port: int = 8321) -> None:
    if load_dotenv is not None:
        load_dotenv()

    client = OgxClient(base_url=f"http://{host}:{port}")

    try:
        client.inspect.health()
    except Exception as exc:
        print(colored(f"Cannot connect to OGX server at {host}:{port}: {exc}", "red"))
        return

    print("=== Providers ===")
    providers = client.providers.list()
    inference_providers = [p for p in providers if p.api == "inference"]
    vector_providers = [p for p in providers if p.api == "vector_io"]
    other_providers = [p for p in providers if p.api not in ("inference", "vector_io")]

    if inference_providers:
        print("\nInference:")
        for p in inference_providers:
            print(f"  - {p.provider_id} ({p.provider_type})")
    if vector_providers:
        print("\nVector I/O:")
        for p in vector_providers:
            print(f"  - {p.provider_id} ({p.provider_type})")
    if other_providers:
        print("\nOther:")
        for p in other_providers:
            print(f"  - {p.provider_id} ({p.provider_type}) [{p.api}]")

    print("\n=== Models ===")
    resp = client.models.list()
    models = resp.data if hasattr(resp, "data") else list(resp)

    llm_models = []
    embedding_models = []
    other_models = []

    for m in models:
        model_id = getattr(m, "identifier", None) or getattr(m, "id", None) or "unknown"
        meta = getattr(m, "custom_metadata", None) or {}
        model_type = meta.get("model_type", "unknown") if isinstance(meta, dict) else "unknown"
        provider = meta.get("provider_id", "?") if isinstance(meta, dict) else "?"

        entry = f"  - {model_id}  (provider: {provider})"
        if model_type == "llm":
            llm_models.append(entry)
        elif model_type == "embedding":
            embedding_models.append(entry)
        else:
            other_models.append(entry)

    if llm_models:
        print(f"\nLLM ({len(llm_models)}):")
        for e in llm_models:
            print(e)
    if embedding_models:
        print(f"\nEmbedding ({len(embedding_models)}):")
        for e in embedding_models:
            print(e)
    if other_models:
        print(f"\nOther ({len(other_models)}):")
        for e in other_models:
            print(e)

    if not models:
        print(colored("  No models available.", "red"))

    total = len(llm_models) + len(embedding_models) + len(other_models)
    print(f"\nTotal: {total} models across {len(providers)} providers")


if __name__ == "__main__":
    fire.Fire(main)
