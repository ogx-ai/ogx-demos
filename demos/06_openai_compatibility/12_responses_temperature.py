"""
Demo: Responses API - temperature

Description:
This demo shows how to use the temperature parameter with the OpenAI
Responses API against a Llama Stack server. Temperature controls the
randomness of token sampling - lower values make output more deterministic,
higher values make it more creative.

Learning Objectives:
- Control output randomness with the temperature parameter
- Compare deterministic vs creative responses
- Understand the temperature scale (0.0 to 2.0)
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

    prompt = "Write a one-sentence tagline for a coffee shop."

    # --- Example 1: Low temperature (deterministic) ---
    print(colored("\n--- temperature=0.0 (deterministic) ---", "cyan"))
    for i in range(2):
        response = client.responses.create(
            model=resolved_model,
            input=prompt,
            temperature=0.0,
        )
        print(f"  Run {i+1}: {response.output_text}")

    # --- Example 2: Medium temperature ---
    print(colored("\n--- temperature=0.7 (balanced) ---", "cyan"))
    for i in range(2):
        response = client.responses.create(
            model=resolved_model,
            input=prompt,
            temperature=0.7,
        )
        print(f"  Run {i+1}: {response.output_text}")

    # --- Example 3: High temperature (creative) ---
    print(colored("\n--- temperature=1.5 (creative) ---", "cyan"))
    for i in range(2):
        response = client.responses.create(
            model=resolved_model,
            input=prompt,
            temperature=1.5,
        )
        print(f"  Run {i+1}: {response.output_text}")


if __name__ == "__main__":
    fire.Fire(main)
