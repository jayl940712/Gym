# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identify model-caused process termination without masking genuine infrastructure errors."""

import json
import shlex
from pathlib import Path


def self_termination_reason(execution: dict, prompt: str) -> str:
    """Recognize a pending broad pkill that matches OpenCode's own command line."""
    if execution.get("opencode_exit_code") not in (-15, -9):
        return ""
    try:
        exported = json.loads(Path(execution["opencode_results_fpath"]).read_text())
        tools = [
            part for message in exported["messages"] for part in message.get("parts", []) if part.get("type") == "tool"
        ]
        tool = tools[-1]
        state = tool.get("state", {})
        if tool.get("tool") != "bash" or state.get("status") != "running":
            return ""
        lexer = shlex.shlex(state.get("input", {}).get("command", ""), posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except (OSError, KeyError, IndexError, ValueError, TypeError):
        return ""
    for index, token in enumerate(tokens):
        if Path(token).name != "pkill":
            continue
        arguments = tokens[index + 1 :]
        arguments = arguments[
            : next((i for i, value in enumerate(arguments) if value in (";", "&", "&&", "|", "||")), len(arguments))
        ]
        if "-f" not in arguments:
            continue
        position = arguments.index("-f") + 1
        if position >= len(arguments):
            continue
        pattern = arguments[position]
        if pattern and (pattern in prompt or pattern == "opencode"):
            return f"Model issued pkill -f {pattern!r}, matching its own OpenCode command line."
    return ""
