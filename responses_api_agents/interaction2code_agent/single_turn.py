# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General single-turn webpage reconstruction and fenced HTML extraction."""

import base64
import re
from pathlib import Path

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


SYSTEM_PROMPT = """You implement interactive webpages from screenshots.
Recreate the initial appearance and the interaction that produces the target state.
Match the layout, text, colors, typography, spacing, and visible details.
Return only one complete HTML document inside a single ```html code block.
Embed all CSS and JavaScript in that document. Do not include explanations."""

USER_INSTRUCTIONS = """Reconstruct the webpage and implement the interaction connecting the two states.
Use the supplied relative file placeholder.jpg for image elements; do not use remote assets.
Give the main interactive trigger the id interact1. The HTML will be opened locally
with placeholder.jpg beside it."""


def build_request(directory: Path, *, max_output_tokens: int) -> NeMoGymResponseCreateParamsNonStreaming:
    """Provide one system and one user message, with only the two problem screenshots."""
    for name in ("before.png", "after.png", "placeholder.jpg"):
        if not (directory / name).resolve().is_relative_to(directory.resolve()):
            raise ValueError("Input asset escapes task directory")
    content = [{"type": "input_text", "text": USER_INSTRUCTIONS}]
    for label, name in [("Initial state:", "before.png"), ("Target state:", "after.png")]:
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Input asset escapes task directory")
        encoded = base64.b64encode(path.read_bytes()).decode()
        content.extend(
            [
                {"type": "input_text", "text": label},
                {"type": "input_image", "image_url": "data:image/png;base64," + encoded, "detail": "high"},
            ]
        )
    return NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {
            "model": "interaction2code-policy",
            "input": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
            "max_output_tokens": max_output_tokens,
            "tools": [],
            "tool_choice": "none",
            "parallel_tool_calls": False,
        }
    )


def response_text(response: NeMoGymResponse) -> str:
    """Extract assistant message text, excluding reasoning and tool-call arguments."""
    return "\n".join(
        part.text
        for item in response.output
        if item.type == "message" and item.role == "assistant"
        for part in item.content
        if part.type == "output_text"
    )


def extract_html(text: str) -> str:
    """Require one complete HTML document in an HTML-labelled or unlabelled fence."""
    blocks = re.findall(r"^[ \t]*```[ \t]*(?:html|htm)?[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*$", text, re.M | re.S | re.I)
    candidates = [block.strip() for block in blocks if re.search(r"<html\b", block, re.I)]
    if len(candidates) != 1 or not re.search(r"</html>\s*$", candidates[0], re.I):
        raise ValueError("Expected exactly one fenced, complete HTML document")
    return candidates[0] + "\n"
