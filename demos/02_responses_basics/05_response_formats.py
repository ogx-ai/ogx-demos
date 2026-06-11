"""
Demo: Response Formats

Description:
This demo teaches how to generate structured outputs using JSON mode and JSON schema validation.

Learning Objectives:
- Use JSON mode to ensure valid JSON output
- Define and enforce JSON schemas for structured data
- Understand the difference between json_object and json_schema formats
- Generate machine-readable structured responses
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import os

import fire
from ogx_client import OgxClient
from termcolor import colored

from demos.shared.utils import can_model_chat, check_model_is_available, get_any_available_chat_model

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

    conversation = client.conversations.create(metadata={"name": "responses-json"})
    conversation_id = conversation.id
    print(f"Created conversation={conversation_id}")

    # JSON mode: only guarantees valid JSON output, no schema enforcement.
    json_object = client.responses.create(
        model=resolved_model,
        conversation=conversation_id,
        input=[{"role": "user", "content": "Return JSON with keys name and role for Ada Lovelace."}],
        text={"format": {"type": "json_object"}},
        stream=False,
    )
    print("[json_object]")
    print(json_object.output_text)

    # JSON schema mode: output must conform to the provided schema.
    json_schema = client.responses.create(
        model=resolved_model,
        conversation=conversation_id,
        input=[
            {
                "role": "user",
                "content": "Return a short summary of OGX.",
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "summary",
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["summary", "tags"],
                },
            }
        },
        stream=False,
    )
    print("[json_schema]")
    print(json_schema.output_text)


if __name__ == "__main__":
    fire.Fire(main)
