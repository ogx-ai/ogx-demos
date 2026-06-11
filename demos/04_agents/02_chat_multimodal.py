# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Demo: Multimodal Chat

Description:
This demo teaches how to build agents that can process and understand images alongside text in conversations.

Learning Objectives:
- Create agents with vision-capable models
- Encode images as data URLs for multimodal input
- Combine image and text inputs in agent messages
- Build context-aware conversations referencing multiple images
"""

import base64
import mimetypes
from pathlib import Path

import fire

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None
from ogx_client import Agent, OgxClient
from termcolor import colored

from demos.shared.utils import check_model_is_available

THIS_DIR = Path(__file__).parent


def _data_url_from_image(file_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for {file_path}")
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"


def main(host: str, port: int, model_id: str | None = None):
    if load_dotenv is not None:
        load_dotenv()
    client = OgxClient(
        base_url=f"http://{host}:{port}",
    )

    if model_id is None:
        print(
            colored(
                "No model id provided. Specify a model that supports vision.", "red"
            )
        )
        return
    else:
        if not check_model_is_available(client, model_id):
            print(
                colored(
                    "Model not available. Specify a model that supports vision.", "red"
                )
            )
            return

    agent = Agent(
        client,
        model=model_id,
        instructions="",
    )

    files = [
        THIS_DIR / "resources" / "dog.png",
        THIS_DIR / "resources" / "pasta.jpeg",
    ]
    for file in files:
        if not file.exists():
            print(colored(f"Error: Required resource file not found: {file}", "red"))
            return
    image_urls = [_data_url_from_image(file) for file in files]

    prompts = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": image_urls[0],
                },
                {
                    "type": "input_text",
                    "text": "Write a haiku about this image",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": image_urls[1],
                },
                {
                    "type": "input_text",
                    "text": "Now update the haiku to include the second image",
                },
            ],
        },
    ]

    session_id = agent.create_session("test-session")
    for prompt in prompts:
        response = agent.create_turn(
            messages=[prompt],
            session_id=session_id,
            stream=False,
        )
        output_text = getattr(response, "output_text", None)
        if output_text is None:
            output_message = getattr(response, "output_message", None)
            output_text = getattr(output_message, "content", "")
        print(colored("model>", "green"), output_text)


if __name__ == "__main__":
    fire.Fire(main)
