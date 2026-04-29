"""
Demo: Responses API - top_p (Nucleus Sampling)

Description:
This demo shows how to use the top_p parameter with the OpenAI Responses API
against a Llama Stack server. The top_p parameter controls nucleus sampling,
where the model considers only tokens whose cumulative probability mass
reaches the specified threshold.

Learning Objectives:
- Use top_p to control sampling diversity
- Compare outputs with different top_p values
- Understand the relationship between top_p and response randomness
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

    prompt = "Suggest a creative name for a new AI startup."

    # --- Example 1: Low top_p (more focused/deterministic) ---
    print(colored("\n--- top_p=0.1 (focused sampling) ---", "cyan"))
    response = client.responses.create(
        model=resolved_model,
        input=prompt,
        top_p=0.1,
        temperature=1.0,
    )
    print(f"Output: {response.output_text}")

    # --- Example 2: Medium top_p ---
    print(colored("\n--- top_p=0.5 (moderate sampling) ---", "cyan"))
    response = client.responses.create(
        model=resolved_model,
        input=prompt,
        top_p=0.5,
        temperature=1.0,
    )
    print(f"Output: {response.output_text}")

    # --- Example 3: High top_p (more diverse/creative) ---
    print(colored("\n--- top_p=0.95 (diverse sampling) ---", "cyan"))
    response = client.responses.create(
        model=resolved_model,
        input=prompt,
        top_p=0.95,
        temperature=1.0,
    )
    print(f"Output: {response.output_text}")


if __name__ == "__main__":
    fire.Fire(main)
