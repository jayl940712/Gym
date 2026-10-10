# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse Qwen parameter tags without rescanning large incomplete HTML arguments."""

import json
import re


OPEN = re.compile(r"<\s*parameter\s*=\s*")
CLOSE = re.compile(r"<\s*/\s*parameter\s*>")


def convert_qwen3_arguments(raw_args: str, partial: bool) -> str:
    """Preserve vLLM 0.24 parameter boundaries and partial-argument semantics."""
    params = {}
    active = None
    cursor = 0
    next_end = raw_args.find(">")
    while True:
        position = raw_args.find("<", cursor)
        if position < 0:
            break
        if next_end >= 0 and next_end < position:
            next_end = raw_args.find(">", position)
        opening = OPEN.match(raw_args, position)
        if opening:
            if active is not None:
                name, start = active
                params[name] = raw_args[start:position].strip()
                active = None
            if next_end < 0:
                break
            name = raw_args[opening.end() : next_end]
            active = (name, next_end + 1)
            cursor = next_end + 1
            continue
        closing = CLOSE.match(raw_args, position)
        if closing:
            if active is not None:
                name, start = active
                params[name] = raw_args[start:position].strip()
                active = None
            cursor = closing.end()
        else:
            cursor = position + 1
    if partial and active is not None:
        name, start = active
        if name:
            params[name] = raw_args[start:].strip()
    return json.dumps(params, ensure_ascii=False)
