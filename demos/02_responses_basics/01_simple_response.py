"""
Demo: Simple Response

Description:
This demo teaches how to use the Responses API for simple text generation with instructions and user input.

Learning Objectives:
- Use the Responses API to generate text completions
- Configure response instructions to control assistant behavior
- Understand the difference between Responses API and Chat Completions API
- Handle model selection and validation for response generation
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
    prompt: str = "Give me a short summary of OGX.",
    instructions: str = "You are a helpful assistant.",
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

    response = client.responses.create(
        model=resolved_model,
        instructions=instructions,
        input=[{"role": "user", "content": prompt}],
        stream=False,
    )
    print(response.output_text)


if __name__ == "__main__":
    fire.Fire(main)
