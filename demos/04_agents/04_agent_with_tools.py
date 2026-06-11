# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Demo: Agent with Tools

Description:
This demo teaches how to create agents with multiple custom client-side tools for specialized tasks.

Learning Objectives:
- Register multiple client-side tools with an agent
- Enable agents to automatically select and execute appropriate tools
- Combine calculator, web search, and stock ticker tools in one agent
- Build multi-capability agents with tool orchestration
"""

import os
import fire

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None
from termcolor import colored

from demos.client_tools.ticker_data import get_ticker_data
from demos.client_tools.web_search import WebSearchTool
from demos.client_tools.calculator import calculator

from ogx_client import OgxClient, Agent, AgentEventLogger

from demos.shared.utils import can_model_chat, check_model_is_available, get_any_available_chat_model


def main(host: str, port: int, model_id: str | None = None):
    if load_dotenv is not None:
        load_dotenv()
    client = OgxClient(base_url=f"http://{host}:{port}")

    api_key = ""
    engine = "tavily"
    if "TAVILY_SEARCH_API_KEY" in os.environ:
        api_key = os.getenv("TAVILY_SEARCH_API_KEY")
    elif "BRAVE_SEARCH_API_KEY" in os.environ:
        api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        engine = "brave"
    else:
        print(
            colored(
                "Warning: TAVILY_SEARCH_API_KEY or BRAVE_SEARCH_API_KEY is not set; Web search will not work",
                "yellow",
            )
        )

    if model_id is None:
        model_id = get_any_available_chat_model(client)
        if model_id is None:
            return
    else:
        if not check_model_is_available(client, model_id):
            return
        if not can_model_chat(client, model_id):
            print(
                colored(
                    f"Model `{model_id}` does not support chat. Choose a chat-capable model.",
                    "red",
                )
            )
            return

    tools = [calculator, get_ticker_data]
    if api_key:
        # Note: While you can also use "builtin::websearch" as a tool,
        # this example shows how to use a client side custom web search tool.
        tools.append(WebSearchTool(engine, api_key))

    agent = Agent(
        client,
        model=model_id,
        instructions="You are a helpful assistant. Use the tools you have access to for providing relevant answers.",
        tools=tools,
    )

    session_id = agent.create_session("test-session")
    print(f"Created session_id={session_id}")

    user_prompts = [
        "What was the closing price of Google stock (ticker symbol GOOG) for 2023 ?",
        "Who was the 42nd president of the United States?",
        "What is 40+30?",
    ]
    for prompt in user_prompts:
        print(colored(f"User> {prompt}", "cyan"))
        response = agent.create_turn(
            messages=[{"role": "user", "content": prompt}],
            session_id=session_id,
        )

        for printable in AgentEventLogger().log(response):
            print(printable, end="", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
