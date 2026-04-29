"""
Demo: Responses API - Combined Parameters

Description:
This demo shows how to combine multiple Responses API parameters in a
single request against a Llama Stack server. It demonstrates using
max_output_tokens, temperature, top_p, truncation, and service_tier
together.

Learning Objectives:
- Combine multiple parameters in a single Responses API call
- Understand how parameters interact with each other
- Build production-ready API calls with fine-tuned control
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

    # --- Example 1: Controlled creative writing ---
    print(colored("\n--- Creative writing with combined params ---", "cyan"))
    response = client.responses.create(
        model=resolved_model,
        input=[
            {"role": "user", "content": "Write a short poem about open-source software."},
        ],
        max_output_tokens=150,
        temperature=0.8,
        top_p=0.9,
        truncation="disabled",
    )
    print(f"Output: {response.output_text}")
    print(f"Status: {response.status}")
    if hasattr(response, "usage") and response.usage:
        print(colored(f"Usage - Input: {response.usage.input_tokens}, "
                      f"Output: {response.usage.output_tokens}", "green"))

    # --- Example 2: Precise factual response ---
    print(colored("\n--- Factual Q&A with combined params ---", "cyan"))
    response = client.responses.create(
        model=resolved_model,
        input=[
            {"role": "system", "content": "You are a concise technical assistant."},
            {"role": "user", "content": "What are the key components of a transformer architecture?"},
        ],
        max_output_tokens=200,
        temperature=0.0,
        top_p=1.0,
        truncation="disabled",
    )
    print(f"Output: {response.output_text}")
    print(f"Status: {response.status}")

    # --- Example 3: Streaming with combined params ---
    print(colored("\n--- Streaming with combined params ---", "cyan"))
    stream = client.responses.create(
        model=resolved_model,
        input="Explain gradient descent in simple terms.",
        max_output_tokens=100,
        temperature=0.5,
        top_p=0.85,
        stream=True,
    )
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
        elif event.type == "response.completed":
            print()
            resp = event.response
            print(colored(f"\nStatus: {resp.status}", "green"))


if __name__ == "__main__":
    fire.Fire(main)
