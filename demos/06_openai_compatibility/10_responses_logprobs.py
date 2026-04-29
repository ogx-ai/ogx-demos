"""
Demo: Responses API - top_logprobs

Description:
This demo shows how to use the top_logprobs parameter with the OpenAI
Responses API against a Llama Stack server. The parameter returns the
log probabilities of the top N most likely tokens at each position.

Learning Objectives:
- Request log probabilities with top_logprobs
- Inspect token-level probability data in the response
- Understand how to use logprobs for model interpretability
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

    # --- Example: Request top 3 logprobs per token ---
    print(colored("\n--- top_logprobs=3 ---", "cyan"))
    response = client.responses.create(
        model=resolved_model,
        input="What is 2 + 2?",
        max_output_tokens=30,
        top_logprobs=3,
    )
    print(f"Output: {response.output_text}")

    # Inspect logprobs in the response output items
    for item in response.output:
        if item.type == "message" and hasattr(item, "content"):
            for content_part in item.content:
                if hasattr(content_part, "logprobs") and content_part.logprobs:
                    print(colored("\nToken logprobs (first 5 positions):", "cyan"))
                    for i, token_logprob in enumerate(content_part.logprobs[:5]):
                        print(f"  Position {i}: token='{token_logprob.token}' "
                              f"logprob={token_logprob.logprob:.4f}")
                        if hasattr(token_logprob, "top_logprobs") and token_logprob.top_logprobs:
                            for alt in token_logprob.top_logprobs:
                                print(f"    alt: '{alt.token}' logprob={alt.logprob:.4f}")

    # --- Example 2: top_logprobs=5 for more alternatives ---
    print(colored("\n--- top_logprobs=5 ---", "cyan"))
    response = client.responses.create(
        model=resolved_model,
        input="The capital of France is",
        max_output_tokens=16,
        top_logprobs=5,
    )
    print(f"Output: {response.output_text}")


if __name__ == "__main__":
    fire.Fire(main)
