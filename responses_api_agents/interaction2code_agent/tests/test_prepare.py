# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import struct
from pathlib import Path

import pytest

from benchmarks.interaction2code.prepare import export, png_size, prepare


def test_export_has_one_row_per_interaction_and_only_unmarked_inputs(tmp_path):
    source = tmp_path / "source"
    page = source / "1"
    page.mkdir(parents=True)
    (page / "action.json").write_text(
        json.dumps({"topic": "shop", "1": {"src": "0", "dst": "1"}, "2": {"src": "0", "dst": "2"}})
    )
    for name in (
        "0.png",
        "1.png",
        "2.png",
        "0_mark.png",
        "1_mark.png",
        "2_mark.png",
        "interaction_1.png",
        "interaction_2.png",
    ):
        (page / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + struct.pack(">II", 1280, 720))
    placeholder = source / "placeholder.jpg"
    placeholder.write_bytes(b"placeholder")
    output = tmp_path / "prepared"
    result = export(source=source, output=output, pages=[1], placeholder=placeholder)
    rows = [json.loads(line) for line in result.read_text().splitlines()]
    assert [row["task_id"] for row in rows] == ["1-1", "1-2"]
    prompt = rows[0]["responses_create_params"]["input"][0]["content"]
    assert "/workspace/index.html" in prompt
    assert "Determine the browser viewport from the input image dimensions" in prompt
    assert "1280" not in prompt and "720" not in prompt
    assert "near-exact visual copy of both states" in prompt
    assert "detach it with setsid and redirect stdin, stdout, and stderr" in prompt
    assert "Playwright" in prompt
    assert "local HTTP server in the background" in prompt
    assert "both the initial state and the state after the interaction" in prompt
    assert "Read both screenshots with\nyour image-capable read tool" in prompt
    assert "compare them visually" in prompt
    assert "IoU" not in prompt
    assert "pixel differences" not in prompt
    assert "overlays" not in prompt
    assert "relative URL" in prompt
    assert "render.py" not in prompt
    assert "/opt/interaction2code" not in prompt
    assert "evaluator" not in prompt.lower()
    assert "test" not in prompt.lower()
    assert "shop" not in prompt
    assert "mark.png" not in prompt
    assert (output / "tasks/1-1/before.png").exists()
    assert not (output / "tasks/1-1/0_mark.png").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["interactions"] == 2
    assert len(manifest["asset_sha256"]) == 8
    (page / "2.png").unlink()
    with pytest.raises(FileNotFoundError):
        export(source=source, output=output, pages=[1], placeholder=placeholder)


@pytest.mark.parametrize("limit", [0, -1])
def test_prepare_rejects_invalid_limits(limit):
    with pytest.raises(ValueError, match="positive"):
        prepare(limit=limit)


def test_invalid_png(tmp_path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Invalid PNG"):
        png_size(path)


def test_committed_examples_are_real_full_dataset_interactions():
    path = Path(__file__).resolve().parents[3] / "benchmarks/interaction2code/example.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 5
    assert len({row["task_id"] for row in rows}) == 5
    assert any(row["page_id"] < 101 for row in rows)
