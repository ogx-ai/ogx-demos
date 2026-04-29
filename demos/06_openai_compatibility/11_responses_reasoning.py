"""
Demo: Responses API - reasoning.effort

Description:
This demo shows how to use the reasoning parameter with the OpenAI
Responses API against a Llama Stack server. The reasoning.effort parameter
controls how much reasoning effort the model puts into its response
(low, medium, or high).

Learning Objectives:
- Use the reasoning parameter to control reasoning effort
- Compare outputs at different reasoning effort levels
- Understand trade-offs between effort and response speed
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

    prompt = "Solve step by step: If a train travels 120 km in 2 hours, what is its speed in m/s?"

    # --- Example 1: Low reasoning effort ---
    # Note: reasoning requires a model that supports thinking (e.g., QwQ, DeepSeek-R1).
    for effort in ("low", "medium", "high"):
        print(colored(f"\n--- reasoning.effort='{effort}' ---", "cyan"))
        try:
            response = client.responses.create(
                model=resolved_model,
                input=prompt,
                reasoning={"effort": effort},
            )
            print(f"Output: {response.output_text}")
        except Exception as e:
            print(colored(f"Not supported: {e}", "yellow"))
            break


if __name__ == "__main__":
    fire.Fire(main)
