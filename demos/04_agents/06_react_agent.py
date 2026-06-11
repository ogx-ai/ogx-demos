# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Demo: ReAct Agent

Description:
This demo teaches how to use the ReAct (Reasoning and Acting) pattern for iterative tool use and problem solving.

Learning Objectives:
- Create ReAct agents for multi-step reasoning
- Combine web search with custom tools in ReAct workflow
- Understand the ReAct pattern for tool orchestration
- Build agents that reason through complex tasks step-by-step
"""

import os
import uuid

import fire

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None
from ogx_client import AgentEventLogger, OgxClient
from ogx_client.lib.agents.react.agent import ReActAgent
from termcolor import colored

from demos.shared.utils import can_model_chat, check_model_is_available, get_any_available_chat_model


def torchtune(query: str = "torchtune"):  # noqa: ARG001
    """
    Answer information about torchtune.

    Note: This is a demo tool that returns static information.
    The query parameter is required by the tool interface but is not used.

    :param query: Unused query parameter (required for tool signature)
    :returns: Information about torchtune
    """
    dummy_response = """
    torchtune is a PyTorch library for easily authoring, finetuning and experimenting with LLMs.

    torchtune provides:

    PyTorch implementations of popular LLMs from Llama, Gemma, Mistral, Phi, and Qwen model families
    Hackable training recipes for full finetuning, LoRA, QLoRA, DPO, PPO, QAT, knowledge distillation, and more
    Out-of-the-box memory efficiency, performance improvements, and scaling with the latest PyTorch APIs
    YAML configs for easily configuring training, evaluation, quantization or inference recipes
    Built-in support for many popular dataset formats and prompt templates
    """
    return dummy_response


def main(host: str, port: int, model_id: str | None = None):
    if load_dotenv is not None:
        load_dotenv()
    tavily_api_key = os.getenv("TAVILY_SEARCH_API_KEY")
    if not tavily_api_key:
        print(
            colored(
                "Error: TAVILY_SEARCH_API_KEY environment variable is not set. This demo requires web search.",
                "red",
            )
        )
        return

    client = OgxClient(
        base_url=f"http://{host}:{port}",
        provider_data={"tavily_search_api_key": tavily_api_key},
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

    print(colored(f"Using model: {model_id}", "green"))
    react_tools = [{"type": "web_search"}, torchtune]
    try:
        agent = ReActAgent(
            client=client,
            model=model_id,
            tools=react_tools,
        )
    except TypeError as exc:
        # Fallback for older clients that only accept string tool names.
        if "Unsupported tool type" not in str(exc):
            raise
        agent = ReActAgent(
            client=client,
            model=model_id,
            tools=["builtin::websearch", torchtune],
        )

    session_id = agent.create_session(f"test-session-{uuid.uuid4().hex}")
    user_prompt = "Whats the best place in new york for a pizza slice at 2am ?"
    print(colored(f"User> {user_prompt}", "blue"))
    response = agent.create_turn(
        messages=[{"role": "user", "content": user_prompt}],
        session_id=session_id,
        stream=True,
    )
    for log in AgentEventLogger().log(response):
        print(log, end="", flush=True)

    user_prompt2 = "What are the popular llms supported in torchtune?"
    print(colored(f"User> {user_prompt2}", "blue"))
    response2 = agent.create_turn(
        messages=[{"role": "user", "content": user_prompt2}],
        session_id=session_id,
        stream=True,
    )
    for log in AgentEventLogger().log(response2):
        print(log, end="", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
