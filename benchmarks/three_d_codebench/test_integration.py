# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check dataset isolation and repeat-safe native metric input preparation."""

import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.three_d_codebench.evaluation import metric_commands, select_rows, task_reward, validate_completion
from benchmarks.three_d_codebench.prepare import DATA_REVISION, VIEWS, prepare
from benchmarks.three_d_codebench.runtime.prepare_references import has_surface_mesh, published_elkhorn_reference
from benchmarks.three_d_codebench.score import input_digest, materialize


def test_reference_glb_requires_surface_primitives(tmp_path: Path) -> None:
    def glb(document):
        payload = json.dumps(document).encode()
        payload += b" " * (-len(payload) % 4)
        return struct.pack("<4sIIII", b"glTF", 2, len(payload) + 20, len(payload), 0x4E4F534A) + payload

    path = tmp_path / "reference.glb"
    path.write_bytes(glb({"asset": {"version": "2.0"}, "scenes": [{"nodes": []}]}))
    assert not has_surface_mesh(path)
    document = {
        "accessors": [{"count": 3}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
    }
    path.write_bytes(glb(document))
    assert has_surface_mesh(path)
    document["accessors"][0]["count"] = 0
    path.write_bytes(glb(document))
    assert not has_surface_mesh(path)
    path.write_bytes(b"truncated GLB")
    assert not has_surface_mesh(path)


def test_published_reference_fallback_rejects_changed_script(tmp_path: Path) -> None:
    script = tmp_path / "reference.py"
    script.write_text("# A different reference must not silently receive the seed-zero fallback.")
    with pytest.raises(ValueError, match="reference changed"):
        published_elkhorn_reference(script, tmp_path / "reference.glb", {})


def test_prepared_tasks_do_not_expose_reference_program(tmp_path: Path) -> None:
    raw = tmp_path / "downloads"
    name = "Sphere_seed0"
    original = raw / "3DCodeBench" / name
    images = raw / "3DCodeBench_ModelLogs/inputs" / name / "images"
    original.mkdir(parents=True)
    images.mkdir(parents=True)
    (raw / "manifest.json").write_text(json.dumps({"revision": DATA_REVISION, "categories": [name]}))
    (original / f"{name}.py").write_text("# private reference program")
    for kind in ("description", "instruction"):
        (original / f"prompt_{kind}.txt").write_text(f"Original {kind} of a sphere.")
    for view in VIEWS:
        (images / view).write_bytes(b"reference image")
    output = tmp_path / "prepared"
    result = prepare(raw_root=raw, output=output)
    assert result["dataset_revision"] == DATA_REVISION
    assert result["counts"] == {"text_to_3d": 1, "image_to_3d": 1}
    assert (output / "reference" / name / (name + ".py")).is_file()
    for task in ("text_to_3d", "image_to_3d"):
        folder = output / task / name
        assert not list(folder.glob("*.py"))
        meta = json.loads((folder / "task.json").read_text())
        assert meta["images"] == (VIEWS if task == "image_to_3d" else [])
        assert sorted(file.name for file in folder.glob("*.png")) == sorted(meta["images"])


def test_native_materialization_preserves_repeats_and_failures(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    (artifact / "glb").mkdir(parents=True)
    (artifact / "glb/prediction.glb").write_bytes(b"generated geometry")
    good = {"task": "text_to_3d", "record_id": "Sphere_seed0", "artifact_dir": str(artifact), "reward": 1}
    failure = good | {"record_id": "Failed_seed0", "artifact_dir": str(tmp_path / "missing"), "reward": 0}
    masked = good | {"record_id": "Masked_seed0", "mask_sample": True}
    destination = tmp_path / "native"
    groups = materialize([good, good.copy(), failure, masked], destination)
    assert len(groups[("text_to_3d", 0)]) == 2
    assert len(groups[("text_to_3d", 1)]) == 1
    for repeat in (0, 1):
        assert (destination / "text_to_3d" / str(repeat) / "policy/Sphere_seed0/glb/Sphere_seed0.glb").is_file()
    assert (destination / "text_to_3d/0/policy/Failed_seed0").is_dir()
    assert not list(destination.rglob("Masked_seed0"))
    with pytest.raises(ValueError, match="All rollouts are masked"):
        materialize([masked], tmp_path / "all_masked")


def test_repeat_identity_does_not_depend_on_completion_order(tmp_path: Path) -> None:
    first = {
        "task": "image_to_3d",
        "record_id": "Sphere_seed0",
        "artifact_dir": str(tmp_path / "missing"),
        "_ng_rollout_index": 1,
    }
    second = first | {"_ng_rollout_index": 0, "mask_sample": True}
    groups = materialize([first, second], tmp_path / "native")
    assert set(groups) == {("image_to_3d", 1)}


def test_task_selection_filters_before_limit() -> None:
    rows = [
        {"task": "text_to_3d", "record_id": "Text"},
        {"task": "image_to_3d", "record_id": "Image1"},
        {"task": "image_to_3d", "record_id": "Image2"},
    ]
    assert select_rows(rows) == rows
    assert select_rows(rows, "image_to_3d", 1) == [rows[1]]
    assert select_rows(rows, "text_to_3d") == [rows[0]]
    with pytest.raises(ValueError, match="No tasks"):
        select_rows([rows[0]], "image_to_3d")


def test_text_scorer_gets_original_description_even_for_failed_program(tmp_path: Path) -> None:
    references = tmp_path / "reference"
    (references / "Sphere_seed0").mkdir(parents=True)
    description = "A small blue sphere.\n"
    (references / "Sphere_seed0/prompt_description.txt").write_text(description)
    row = {"task": "text_to_3d", "record_id": "Sphere_seed0", "artifact_dir": str(tmp_path / "missing")}
    materialize([row], tmp_path / "native", references)
    assert (tmp_path / "native/text_to_3d/0/policy/Sphere_seed0/prompt.txt").read_text() == description


@pytest.mark.parametrize("task", ["text_to_3d", "image_to_3d"])
def test_native_metric_routing_and_rewards_are_task_specific(task: str) -> None:
    commands = metric_commands(task, ["--model", "policy"], ["--data-root", "/references"], "cpu")
    scripts = [script for _, script, _ in commands]
    uni3d = next(options for _, script, options in commands if script == "shape_uni3d.py")
    assert uni3d[uni3d.index("--task") + 1] == task
    assert "--overwrite" in uni3d  # Content invalidation must bypass the native presence-only cache.
    metrics = {
        "text_image_similarity_siglip2": {"render_mean": 0.31},
        "image_similarity_siglip2": {"score_mean": 0.82},
    }
    reward, stage = task_reward(task, metrics)
    if task == "text_to_3d":
        assert "text_image_similarity.py" in scripts
        assert not any(name == "image_similarity_siglip2" for name, _, _ in commands)
        assert reward == 0.31
        assert stage == "native_siglip2_text_image_mean"
        metrics["text_image_similarity_siglip2"]["render_mean"] = None
    else:
        assert "text_image_similarity.py" not in scripts
        assert any(name == "image_similarity_siglip2" for name, _, _ in commands)
        assert reward == 0.82
        assert stage == "native_siglip2_view_paired"
        metrics["image_similarity_siglip2"]["score_mean"] = None
    assert task_reward(task, metrics)[0] == 0.0


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_task", "wrong_record"])
def test_incomplete_or_mismatched_results_cannot_be_scored(fault: str) -> None:
    row = {"task": "text_to_3d", "record_id": "Sphere", "_ng_task_index": 0, "_ng_rollout_index": 0}
    validate_completion([row], [row.copy()])
    candidates = {
        "missing": [],
        "duplicate": [row, row],
        "wrong_task": [row | {"task": "image_to_3d"}],
        "wrong_record": [row | {"record_id": "Other"}],
    }
    with pytest.raises(ValueError, match="exactly one completed rollout"):
        validate_completion([row], candidates[fault])


def test_metric_cache_changes_with_predictions_and_references(tmp_path: Path) -> None:
    prediction = tmp_path / "policy/Sphere"
    prediction.mkdir(parents=True)
    render = prediction / "render.png"
    render.write_bytes(b"original prediction")
    references = tmp_path / "reference"
    (references / "Sphere").mkdir(parents=True)
    ref = references / "Sphere/reference.png"
    ref.write_bytes(b"original reference")
    root = prediction.parent
    before = input_digest(root, references, ["Sphere"])
    (root / "_metrics").mkdir()
    (root / "_metrics/report.json").write_text("{}")
    assert input_digest(root, references, ["Sphere"]) == before
    render.write_bytes(b"revised prediction")
    revised = input_digest(root, references, ["Sphere"])
    assert revised != before
    ref.write_bytes(b"revised reference")
    assert input_digest(root, references, ["Sphere"]) != revised


def test_bundled_metric_runs_from_isolated_clone(tmp_path: Path) -> None:
    """Run the packaged CLI with no sibling checkout or inherited import paths."""
    clone = tmp_path / "fresh-gym"
    package = clone / "benchmarks/three_d_codebench"
    source = Path(__file__).parent
    shutil.copytree(source / "metrics", package / "metrics", ignore=shutil.ignore_patterns("__pycache__"))
    package.joinpath("__init__.py").write_text("")
    root = tmp_path / "independent-results"
    for name, status in (("good", "OK"), ("bad", "ERR_EXEC")):
        render = root / "policy" / name / "renders"
        render.mkdir(parents=True)
        (render / "render_log.json").write_text(json.dumps({"status": status, "error": ""}))
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
            "runpy.run_module('benchmarks.three_d_codebench.metrics.executability',run_name='__main__')",
            str(clone),
            "--model",
            "policy",
            "--results-root",
            str(root),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((root / "policy/_metrics/executability.json").read_text())
    assert report["n_total"] == 2
    assert report["n_pass"] == 1
    assert report["pass_rate"] == 0.5
    assert report["failed_instances"] == ["bad"]
