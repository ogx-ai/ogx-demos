"""
Demo: System Prompts

Description:
This demo teaches how to use system prompts to control the behavior and persona of the AI assistant.

Learning Objectives:
- Add system messages to chat completions
- Control assistant behavior through system prompts
- Understand the role of system messages in conversation context
- Combine system prompts with user messages
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

from demos.shared.utils import (
    can_model_chat,
    check_model_is_available,
    get_any_available_chat_model,
)

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


def _maybe_load_dotenv() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _print_stream(chunks) -> None:
    for chunk in chunks:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
    print()


def main(
    host: str,
    port: int,
    model_id: str | None = None,
    prompt: str = "Give me a short summary of OGX.",
    system_prompt: str = "You are a helpful assistant.",
    stream: bool = False,
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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if stream:
        response_stream = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            stream=True,
        )
        _print_stream(response_stream)
        response_stream.close()
        return

    completion = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
    )
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    fire.Fire(main)
