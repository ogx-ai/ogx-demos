"""
Demo: Responses API - Streaming with stream_options

Description:
This demo shows how to use streaming and stream_options with the OpenAI
Responses API against a Llama Stack server. Streaming returns response
tokens incrementally, and stream_options can include usage statistics.

Learning Objectives:
- Stream responses from the Responses API
- Use stream_options to include usage data in the stream
- Process streaming events and extract output
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

    # --- Example 1: Basic streaming ---
    print(colored("\n--- Basic streaming ---", "cyan"))
    stream = client.responses.create(
        model=resolved_model,
        input="Explain what Llama Stack is in two sentences.",
        stream=True,
    )
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
        elif event.type == "response.completed":
            print()
            print(colored(f"\nStatus: {event.response.status}", "green"))

    # --- Example 2: Streaming with include_usage in stream_options ---
    print(colored("\n--- Streaming with stream_options (include_usage) ---", "cyan"))
    stream = client.responses.create(
        model=resolved_model,
        input="List three benefits of open-source AI.",
        stream=True,
        stream_options={"include_usage": True},
    )
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
        elif event.type == "response.completed":
            print()
            resp = event.response
            print(colored(f"\nStatus: {resp.status}", "green"))
            if hasattr(resp, "usage") and resp.usage:
                print(colored(f"Usage - Input tokens: {resp.usage.input_tokens}", "green"))
                print(colored(f"Usage - Output tokens: {resp.usage.output_tokens}", "green"))
                if hasattr(resp.usage, "total_tokens"):
                    print(colored(f"Usage - Total tokens: {resp.usage.total_tokens}", "green"))


if __name__ == "__main__":
    fire.Fire(main)
