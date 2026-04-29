"""
Demo: Responses API - parallel_tool_calls

Description:
This demo shows how to use the parallel_tool_calls parameter with the
OpenAI Responses API against a Llama Stack server. This parameter controls
whether the model can issue multiple tool calls in a single response turn.

Learning Objectives:
- Use function tools with the Responses API
- Control parallel vs sequential tool calling behavior
- Handle multiple tool call results in a single turn
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import json
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


# -- Simulated local tools ---------------------------------------------------

WEATHER_DATA = {
    "san francisco": {"temperature": "62F", "condition": "Foggy"},
    "new york": {"temperature": "45F", "condition": "Cloudy"},
    "london": {"temperature": "50F", "condition": "Rainy"},
    "tokyo": {"temperature": "55F", "condition": "Sunny"},
}


def get_weather(location: str) -> str:
    data = WEATHER_DATA.get(location.lower())
    if data is None:
        return json.dumps({"error": f"No weather data for {location}"})
    return json.dumps(data)


TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'San Francisco'",
                },
            },
            "required": ["location"],
        },
    },
]

TOOL_MAP = {"get_weather": get_weather}


def _handle_tool_calls(response) -> list[dict]:
    """Extract and execute tool calls from a response, returning results."""
    results = []
    for item in response.output:
        if item.type == "function_call":
            fn_name = item.name
            try:
                fn_args = json.loads(item.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = {}
            print(colored(f"  Tool call: {fn_name}({fn_args})", "yellow"))

            fn = TOOL_MAP.get(fn_name)
            if fn is None:
                result = json.dumps({"error": f"Unknown function: {fn_name}"})
            else:
                result = fn(**fn_args)
            print(colored(f"  Tool result: {result}", "yellow"))
            results.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": result,
            })
    return results


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

    prompt = "What is the weather in San Francisco and Tokyo?"

    # --- Example 1: parallel_tool_calls=True (default) ---
    print(colored("\n--- parallel_tool_calls=True ---", "cyan"))
    print(colored(f"User> {prompt}", "blue"))
    response = client.responses.create(
        model=resolved_model,
        input=prompt,
        tools=TOOLS,
        parallel_tool_calls=True,
    )

    tool_results = _handle_tool_calls(response)
    num_calls = len(tool_results)
    print(f"  Number of tool calls in single turn: {num_calls}")

    if tool_results:
        final = client.responses.create(
            model=resolved_model,
            input=[
                {"role": "user", "content": prompt},
                *[{"type": "function_call", "name": item.name, "call_id": item.call_id, "arguments": item.arguments} for item in response.output if item.type == "function_call"],
                *tool_results,
            ],
            tools=TOOLS,
        )
        print(colored(f"Assistant> {final.output_text}", "green"))

    # --- Example 2: parallel_tool_calls=False ---
    print(colored("\n--- parallel_tool_calls=False ---", "cyan"))
    print(colored(f"User> {prompt}", "blue"))
    response = client.responses.create(
        model=resolved_model,
        input=prompt,
        tools=TOOLS,
        parallel_tool_calls=False,
    )

    tool_results = _handle_tool_calls(response)
    num_calls = len(tool_results)
    print(f"  Number of tool calls in single turn: {num_calls}")

    if tool_results:
        final = client.responses.create(
            model=resolved_model,
            input=[
                {"role": "user", "content": prompt},
                *[{"type": "function_call", "name": item.name, "call_id": item.call_id, "arguments": item.arguments} for item in response.output if item.type == "function_call"],
                *tool_results,
            ],
            tools=TOOLS,
        )
        print(colored(f"Assistant> {final.output_text}", "green"))


if __name__ == "__main__":
    fire.Fire(main)
