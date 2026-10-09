# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task selection and task-specific native scorer wiring."""

TASKS = ("text_to_3d", "image_to_3d")
PROTOCOL = "native-task-specific-v2"


def validate_completion(inputs: list[dict], rows: list[dict]) -> None:
    def identity(row: dict) -> tuple:
        return row["task"], row["record_id"], row["_ng_task_index"], row["_ng_rollout_index"]

    expected = {identity(row) for row in inputs}
    actual = [identity(row) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Scoring requires exactly one completed rollout for every selected materialized input")


def select_rows(rows: list[dict], task: str = "all", limit: int = 0) -> list[dict]:
    if task not in (*TASKS, "all"):
        raise ValueError(f"Unknown task selection: {task}")
    if limit < 0:
        raise ValueError("Limit must be nonnegative")
    if any(row["task"] not in TASKS for row in rows):
        raise ValueError("Input contains an unknown task")
    selected = [row for row in rows if task == "all" or row["task"] == task]
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No tasks selected")
    return selected


def metric_commands(task: str, common: list[str], references: list[str], device: str) -> list[tuple]:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    commands = [
        ("executability", "executability.py", common),
        ("shape_chamfer", "shape_chamfer.py", common + references),
    ]
    if task == "text_to_3d":
        commands.append(
            (
                "text_image_similarity_siglip2",
                "text_image_similarity.py",
                common + references + ["--prompt-type", "description", "--device", device],
            )
        )
    else:
        commands.append(
            (
                "image_similarity_siglip2",
                "image_similarity.py",
                common + references + ["--encoder", "siglip2", "--device", device],
            )
        )
    # Reference-image DINOv3 is a supplementary diagnostic on the text track.
    commands.extend(
        [
            (
                "image_similarity_dinov3",
                "image_similarity.py",
                common + references + ["--encoder", "dinov3", "--device", device],
            ),
            # Our content cache decides whether to run. Uni3D's own presence-only
            # cache must not retain a stale report after a reference or prediction changes.
            (
                "shape_uni3d",
                "shape_uni3d.py",
                common + references + ["--task", task, "--device", device, "--overwrite"],
            ),
        ]
    )
    return commands


def task_reward(task: str, metrics: dict) -> tuple[float, str]:
    if task == "text_to_3d":
        value = metrics["text_image_similarity_siglip2"]["render_mean"]
        stage = "native_siglip2_text_image_mean"
    elif task == "image_to_3d":
        value = metrics["image_similarity_siglip2"]["score_mean"]
        stage = "native_siglip2_view_paired"
    else:
        raise ValueError(f"Unknown task: {task}")
    return (float(value) if value is not None else 0.0), stage
