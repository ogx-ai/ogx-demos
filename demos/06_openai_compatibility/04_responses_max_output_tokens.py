"""
Demo: Responses API - max_output_tokens

Description:
This demo shows how to use the max_output_tokens parameter with the
OpenAI Responses API against a Llama Stack server. The parameter limits
the number of tokens the model can generate in its response.

Learning Objectives:
- Control response length with max_output_tokens
- Observe how the model truncates output when the limit is reached
- Check the response status for length-related truncation
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

    # --- Example 1: Small token limit to force truncation ---
    print(colored("\n--- max_output_tokens=50 (short response) ---", "cyan"))
    response = client.responses.create(
        model=resolved_model,
        input="Write a detailed essay about the history of artificial intelligence.",
        max_output_tokens=50,
    )
    print(f"Output: {response.output_text}")
    print(f"Status: {response.status}")
    if hasattr(response, "incomplete_details") and response.incomplete_details:
        print(f"Incomplete details: {response.incomplete_details}")

    # --- Example 2: Larger token limit for a fuller response ---
    print(colored("\n--- max_output_tokens=200 (longer response) ---", "cyan"))
    response = client.responses.create(
        model=resolved_model,
        input="Write a detailed essay about the history of artificial intelligence.",
        max_output_tokens=200,
    )
    print(f"Output: {response.output_text}")
    print(f"Status: {response.status}")
    if hasattr(response, "incomplete_details") and response.incomplete_details:
        print(f"Incomplete details: {response.incomplete_details}")


if __name__ == "__main__":
    fire.Fire(main)
