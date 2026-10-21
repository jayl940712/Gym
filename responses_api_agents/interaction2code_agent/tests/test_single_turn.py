# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest

from responses_api_agents.interaction2code_agent.single_turn import build_request, extract_html


@pytest.mark.parametrize("label", ["html", "HTML", ""])
def test_complete_fenced_html_is_extracted(label):
    html = "<html><body><script>const s = `x`;</script></body></html>"
    assert extract_html(f"Explanation\n```{label}\n{html}\n```\n") == html + "\n"


@pytest.mark.parametrize(
    "text",
    [
        "<html></html>",
        "```html\n<div>fragment</div>\n```",
        "```html\n<html>truncated\n```",
        "```python\n<html></html>\n```",
        "```html\n<html></html>\n```\n```html\n<html></html>\n```",
    ],
)
def test_missing_ambiguous_or_incomplete_deliverables_are_rejected(text):
    with pytest.raises(ValueError):
        extract_html(text)


def test_problem_has_exactly_two_messages_two_images_and_no_tools(tmp_path):
    for name in ["before.png", "after.png", "placeholder.jpg"]:
        (tmp_path / name).write_bytes(name.encode())
    params = build_request(tmp_path, max_output_tokens=1234).model_dump()
    assert [m["role"] for m in params["input"]] == ["system", "user"]
    images = [c for c in params["input"][1]["content"] if c["type"] == "input_image"]
    assert len(images) == 2
    assert [base64.b64decode(c["image_url"].split(",")[1]) for c in images] == [b"before.png", b"after.png"]
    assert params["tools"] == [] and params["tool_choice"] == "none"
    assert params["max_output_tokens"] == 1234
    assert "render.py" not in str(params) and "viewport.json" not in str(params)


@pytest.mark.parametrize("name", ["before.png", "after.png", "placeholder.jpg"])
def test_single_turn_rejects_escaping_assets(tmp_path, name):
    directory = tmp_path / "task"
    directory.mkdir()
    (tmp_path / "outside").write_bytes(b"private")
    for f in ["before.png", "after.png", "placeholder.jpg"]:
        (directory / f).write_bytes(b"image")
    (directory / name).unlink()
    (directory / name).symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="escapes"):
        build_request(directory, max_output_tokens=100)
