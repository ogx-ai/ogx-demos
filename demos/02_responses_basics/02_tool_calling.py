"""
Demo: Tool Calling

Description:
This demo teaches how to use built-in tools like web search with the Responses API to enhance model capabilities.

Learning Objectives:
- Enable tool calling in the Responses API
- Use the web_search tool for retrieving real-time information
- Control tool choice to ensure specific tools are used
- Include tool execution metadata in responses
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


def _resolve_model(client: OgxClient, model_id: str | None) -> str | None:
    resolved_model = model_id or os.getenv("OGX_MODEL") or None
    if resolved_model is None:
        return get_any_available_chat_model(client)
    if not check_model_is_available(client, resolved_model):
        return None
    return resolved_model


def main(
    host: str,
    port: int,
    model_id: str | None = None,
    prompt: str = (
        "Search the web for who was the 42nd president of the United States "
        "and answer with the name only."
    ),
) -> None:
    _maybe_load_dotenv()

    client = OgxClient(base_url=f"http://{host}:{port}")

    resolved_model = _resolve_model(client, model_id)
    if resolved_model is None:
        return
    print(f"Using model: {resolved_model}")
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
        instructions="Use web_search and reply with a plain-text answer only.",
        input=[{"role": "user", "content": prompt}],
        tools=[{"type": "web_search"}],
        tool_choice="auto",
        stream=False,
    )
    print(response.output_text or response.output)


if __name__ == "__main__":
    fire.Fire(main)
