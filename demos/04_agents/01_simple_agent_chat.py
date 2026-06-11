# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Demo: Simple Agent Chat

Description:
This demo teaches how to create a basic conversational agent with web search capabilities and safety shields.

Learning Objectives:
- Create and configure an Agent with the Agent API
- Set up agent sessions for stateful conversations
- Enable web search tools for real-time information retrieval
- Configure safety shields for input and output filtering
"""

import inspect
import os

import fire

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None
from ogx_client import OgxClient, Agent, AgentEventLogger
from termcolor import colored

from demos.shared.utils import check_model_is_available, get_any_available_chat_model


def main(host: str, port: int, model_id: str | None = None):
    if load_dotenv is not None:
        load_dotenv()
    if "TAVILY_SEARCH_API_KEY" not in os.environ:
        print(
            colored(
                "Warning: TAVILY_SEARCH_API_KEY is not set; please set it for this script.",
                "yellow",
            )
        )
        return

    client = OgxClient(
        base_url=f"http://{host}:{port}",
        provider_data={"tavily_search_api_key": os.getenv("TAVILY_SEARCH_API_KEY")},
    )

    if model_id is None:
        model_id = get_any_available_chat_model(client)
        if model_id is None:
            return
    else:
        if not check_model_is_available(client, model_id):
            return

    print(f"Using model: {model_id}")

    agent_kwargs = {
        "model": model_id,
        "instructions": "",
        # OpenAI Responses tool schema requires a type discriminator.
        "tools": [{"type": "web_search"}],
        "enable_session_persistence": False,
    }
    allowed_params = set(inspect.signature(Agent.__init__).parameters)
    filtered_kwargs = {k: v for k, v in agent_kwargs.items() if k in allowed_params}
    try:
        agent = Agent(client, **filtered_kwargs)
    except TypeError as exc:
        # Fallback for older clients that only accept string tool names.
        if "Unsupported tool type" not in str(exc):
            raise
        filtered_kwargs["tools"] = ["builtin::websearch"]
        agent = Agent(client, **filtered_kwargs)
    user_prompts = [
        "Hello",
        "Search web for which players played in the winning team of the NBA western conference semifinals of 2024",
    ]

    session_id = agent.create_session("test-session")
    for prompt in user_prompts:
        print(f"User> {prompt}")
        response = agent.create_turn(
            messages=[{"role": "user", "content": prompt}],
            session_id=session_id,
        )

        for printable in AgentEventLogger().log(response):
            print(printable, end="", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
