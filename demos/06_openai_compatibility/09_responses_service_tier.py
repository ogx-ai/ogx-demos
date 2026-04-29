"""
Demo: Responses API - service_tier

Description:
This demo shows how to use the service_tier parameter with the OpenAI
Responses API against a Llama Stack server. The service_tier parameter
allows you to specify the processing tier for the request (e.g., "auto",
"default").

Learning Objectives:
- Set the service_tier for a Responses API request
- Understand how service_tier affects request routing
- Inspect the service_tier in the response
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

    # --- Example 1: service_tier="auto" ---
    print(colored("\n--- service_tier='auto' ---", "cyan"))
    response = client.responses.create(
        model=resolved_model,
        input="What is Llama Stack?",
        service_tier="auto",
    )
    print(f"Output: {response.output_text}")
    if hasattr(response, "service_tier") and response.service_tier:
        print(colored(f"Service tier used: {response.service_tier}", "green"))

    # --- Example 2: service_tier="default" ---
    print(colored("\n--- service_tier='default' ---", "cyan"))
    response = client.responses.create(
        model=resolved_model,
        input="What are the benefits of open-source LLMs?",
        service_tier="default",
    )
    print(f"Output: {response.output_text}")
    if hasattr(response, "service_tier") and response.service_tier:
        print(colored(f"Service tier used: {response.service_tier}", "green"))


if __name__ == "__main__":
    fire.Fire(main)
