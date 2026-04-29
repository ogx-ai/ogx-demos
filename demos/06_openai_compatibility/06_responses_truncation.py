"""
Demo: Responses API - truncation

Description:
This demo shows how to use the truncation parameter with the OpenAI Responses
API against a Llama Stack server. The truncation parameter controls how the
model handles context that exceeds the context window, allowing automatic
truncation of older messages.

Learning Objectives:
- Use the truncation parameter to handle long contexts
- Understand the "auto" vs "disabled" truncation strategies
- See how truncation affects multi-turn conversations
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import os
import sys

import fire
from openai import OpenAI
from termcolor import colored

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import resolve_openai_model

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


def _maybe_load_dotenv() -> None:
    if load_dotenv is not None:
        load_dotenv()


def main(
    host: str,
    port: int,
    model_id: str | None = None,
    scheme: str = "http",
) -> None:
    _maybe_load_dotenv()

    if scheme not in {"http", "https"}:
        raise ValueError("scheme must be 'http' or 'https'")
    if host not in {"localhost", "127.0.0.1", "::1"} and scheme != "https":
        print(colored("Warning: using HTTP for a non-local host. Consider --scheme https.", "yellow"))

    client = OpenAI(
        base_url=f"{scheme}://{host}:{port}/v1",
        api_key=os.getenv("LLAMA_STACK_API_KEY", "fake"),
    )

    resolved_model = resolve_openai_model(client, model_id)
    if resolved_model is None:
        return
    print(f"Using model: {resolved_model}")

    # --- Example 1: Auto truncation (truncate older messages if context is too long) ---
    # Note: truncation='auto' is not yet supported by Llama Stack.
    print(colored("\n--- truncation='auto' ---", "cyan"))
    try:
        response = client.responses.create(
            model=resolved_model,
            input=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! How can I help you today?"},
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 equals 4."},
                {"role": "user", "content": "What is my name?"},
            ],
            truncation="auto",
        )
        print(f"Output: {response.output_text}")
    except Exception as e:
        print(colored(f"Not supported: {e}", "yellow"))

    # --- Example 2: Disabled truncation (fail if context exceeds window) ---
    print(colored("\n--- truncation='disabled' ---", "cyan"))
    response = client.responses.create(
        model=resolved_model,
        input=[
            {"role": "user", "content": "My name is Bob."},
            {"role": "assistant", "content": "Nice to meet you, Bob!"},
            {"role": "user", "content": "What is my name?"},
        ],
        truncation="disabled",
    )
    print(f"Output: {response.output_text}")


if __name__ == "__main__":
    fire.Fire(main)
