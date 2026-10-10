# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen argument semantics and bounded parsing of incomplete generated HTML."""

import json
import time

import pytest

from benchmarks.interaction2code.runtime.qwen3_arguments import convert_qwen3_arguments


@pytest.mark.parametrize(
    "raw,partial,expected",
    [
        ("plain text", False, {}),
        ("<parameter=x> one </parameter>", False, {"x": "one"}),
        ("<parameter=x> one", False, {}),
        ("<parameter=x> one", True, {"x": "one"}),
        ("<parameter=x> one<parameter=y> two", False, {"x": "one"}),
        ("<parameter=x> one<parameter=y> two", True, {"x": "one", "y": "two"}),
        ("<parameter=x> one<parameter=", True, {"x": "one"}),
        ("<parameter=x> one</parameter><parameter=x>two</parameter>", False, {"x": "two"}),
        ("< parameter = x > 雪 < / parameter >", False, {"x ": "雪"}),
        (
            "<parameter=content><html><div>text</div></html></parameter>",
            False,
            {"content": "<html><div>text</div></html>"},
        ),
    ],
)
def test_argument_boundaries(raw, partial, expected):
    assert json.loads(convert_qwen3_arguments(raw, partial)) == expected


def test_large_incomplete_html_does_not_rescan_for_missing_tag_end():
    """A large partial tool call must not freeze the API event loop."""
    content = "<unknown" * 100000
    raw = "<parameter=filePath>/workspace/index.html</parameter><parameter=content>" + content
    start = time.monotonic()
    parsed = json.loads(convert_qwen3_arguments(raw, True))
    assert parsed == {"filePath": "/workspace/index.html", "content": content}
    assert time.monotonic() - start < 5
