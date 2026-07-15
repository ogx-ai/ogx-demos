# demo-requires: EXAMPLE_ENV_VAR
"""
Demo: [Name]

Description:
[What this demo teaches — 1-2 sentences.]

Learning Objectives:
- [Objective 1]
- [Objective 2]
- [Objective 3]
"""

from __future__ import annotations

import os

import fire
from ogx_client import OgxClient
from termcolor import colored

from demos.shared.utils import (
    can_model_chat,
    check_model_is_available,
    resolve_model,
)

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


def _maybe_load_dotenv() -> None:
    if load_dotenv is not None:
        load_dotenv()


def main(
    host: str,
    port: int,
    model_id: str | None = None,
) -> None:
    _maybe_load_dotenv()

    client = OgxClient(base_url=f"http://{host}:{port}")

    resolved_model = model_id or os.getenv("OGX_CHAT_MODEL") or None
    if resolved_model is None:
        resolved_model = resolve_model(client, None)
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

    # --- Your demo logic here ---
    #
    # If you create resources (vector stores, files, etc.), clean them up:
    #
    # resource = None
    # try:
    #     resource = client.vector_stores.create(...)
    #     # ... demo logic ...
    # finally:
    #     if resource is not None:
    #         try:
    #             client.vector_stores.delete(vector_store_id=resource.id)
    #         except Exception as e:
    #             print(colored(f"Warning: cleanup failed: {e}", "yellow"))


if __name__ == "__main__":
    fire.Fire(main)
