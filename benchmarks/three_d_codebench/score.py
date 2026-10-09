# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run pinned native metrics with task-specific conditioning and Gym rewards."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from benchmarks.three_d_codebench.evaluation import (
    PROTOCOL,
    TASKS,
    metric_commands,
    select_rows,
    task_reward,
    validate_completion,
)


def materialize(
    rows: list[dict], destination: Path, references: Path | None = None
) -> dict[tuple[str, int], list[dict]]:
    """Keep repeated instances separate and exclude infrastructure failures."""
    groups = defaultdict(list)
    counts = defaultdict(int)
    for row in rows:
        key = (row["task"], row["record_id"])
        repeat = row.get("_ng_rollout_index", counts[key])
        counts[key] += 1
        if row.get("mask_sample", False):
            continue
        group = (row["task"], repeat)
        groups[group].append(row)
        instance = destination / group[0] / str(repeat) / "policy" / row["record_id"]
        instance.mkdir(parents=True, exist_ok=True)
        if row["task"] == "text_to_3d" and references is not None:
            # The native text-image scorer reads prompt.txt even for failed predictions.
            shutil.copy2(references / row["record_id"] / "prompt_description.txt", instance / "prompt.txt")
        artifact = Path(row["artifact_dir"])
        for folder in ("renders", "glb"):
            if (artifact / folder).is_dir():
                shutil.copytree(artifact / folder, instance / folder, dirs_exist_ok=True)
        prediction = instance / "glb/prediction.glb"
        if prediction.exists():
            prediction.rename(prediction.with_name(row["record_id"] + ".glb"))
        if (artifact / "solution.py").exists():
            shutil.copy2(artifact / "solution.py", instance / (row["record_id"] + ".py"))
    if not groups:
        raise ValueError("All rollouts are masked; no benchmark measurements exist")
    return groups


def input_digest(root: Path, references: Path, names: list[str]) -> str:
    """Invalidate cached native reports when predictions or references change."""
    digest = hashlib.sha256(PROTOCOL.encode())
    for label, directory in [("predictions", root)] + [(name, references / name) for name in sorted(names)]:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "_metrics" not in path.relative_to(directory).parts:
                digest.update(f"{label}/{path.relative_to(directory)}".encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", choices=["all", *TASKS], default="all")
    args = parser.parse_args()
    args.run = args.run.resolve()
    metrics_root = Path(__file__).resolve().parent / "metrics"
    args.references = args.references.resolve()
    rows = select_rows(
        [
            json.loads(line)
            for line in (args.run / "rollouts_generation.jsonl").read_text().splitlines()
            if line.strip()
        ],
        args.task,
    )
    inputs_path = args.run / "rollouts_generation_materialized_inputs.jsonl"
    inputs = select_rows(
        [json.loads(line) for line in inputs_path.read_text().splitlines() if line.strip()], args.task
    )
    validate_completion(inputs, rows)
    (args.run / "scoring_inputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in inputs))
    for row in rows:
        if row.get("mask_sample", False):
            continue
        name = row["record_id"]
        reference = args.references / name
        required = [reference / "glb" / (name + ".glb"), reference / "prompt_description.txt"]
        required += [reference / "images" / f"Image_{view:03}.png" for view in (5, 15, 25, 35)]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError(f"Missing trusted reference assets: {missing}")
    native = args.run / "native"
    groups = materialize(rows, native, args.references)
    reports = {}
    for (task, repeat), measured in groups.items():
        root = native / task / str(repeat)
        common = ["--model", "policy", "--results-root", str(root)]
        references = ["--data-root", str(args.references)]
        commands = metric_commands(task, common, references, args.device)
        fingerprint = input_digest(root / "policy", args.references, [row["record_id"] for row in measured])
        metrics = {}
        for name, script, options in commands:
            report_path = root / "policy/_metrics" / (name + ".json")
            cache_path = root / (name + ".cache.json")
            signature = {
                "inputs": fingerprint,
                "source": hashlib.sha256((metrics_root / script).read_bytes()).hexdigest(),
                "options": options,
                "encoder": hashlib.sha256((metrics_root / "uni3d/point_encoder.py").read_bytes()).hexdigest()
                if script == "shape_uni3d.py"
                else None,
            }
            if report_path.is_file() and cache_path.is_file() and json.loads(cache_path.read_text()) == signature:
                metrics[name] = json.loads(report_path.read_text())
                print(f"Reused {task}/{repeat}/{name}", flush=True)
                continue
            print(f"Scoring {task}/{repeat}/{name}", flush=True)
            with (root / (name + ".log")).open("w") as log:
                result = subprocess.run(
                    [sys.executable, "-m", "benchmarks.three_d_codebench.metrics." + Path(script).stem, *options],
                    cwd=Path(__file__).resolve().parents[2],
                    env=os.environ.copy(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                detail = (root / (name + ".log")).read_text()
                if name == "image_similarity_dinov3" and "GatedRepoError" in detail:
                    metrics[name] = {"status": "unavailable", "reason": "Hugging Face checkpoint access denied"}
                    continue
                raise RuntimeError(f"Native {name} failed; inspect {root / (name + '.log')}")
            metrics[name] = json.loads(report_path.read_text())
            cache_path.write_text(json.dumps(signature, indent=2) + "\n")
        taxonomy_path = root / "failure_taxonomy.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.three_d_codebench.metrics.failure_taxonomy",
                "--roots",
                str(root),
                "--out",
                str(taxonomy_path),
            ],
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        metrics["failure_taxonomy"] = json.loads(taxonomy_path.read_text())
        reports[f"{task}/{repeat}"] = metrics
        indexed = {
            name: {item["instance"]: item for item in value.get("per_instance", [])} for name, value in metrics.items()
        }
        for row in measured:
            instance = row["record_id"]
            row["executability_reward"] = row["reward"]
            instance_metrics = {name: values[instance] for name, values in indexed.items() if instance in values}
            row["reward"], row["reward_stage"] = task_reward(task, instance_metrics)
            row["native_metrics"] = {
                name: values.get(instance, metrics[name] if metrics[name].get("status") == "unavailable" else None)
                for name, values in indexed.items()
                if values or metrics[name].get("status") == "unavailable"
            }
            artifact = Path(row["artifact_dir"])
            (artifact / "result_scored.json").write_text(json.dumps(row, indent=2) + "\n")
    (args.run / "native_metrics.json").write_text(json.dumps(reports, indent=2) + "\n")
    (args.run / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {"protocol": PROTOCOL, "task_selection": args.task, "tracks": {}}
    for task in TASKS:
        track = [row for row in rows if row["task"] == task]
        if track:
            measured = [row for row in track if not row.get("mask_sample", False)]
            summary["tracks"][task] = {
                "n_total": len(track),
                "n_measured": len(measured),
                "n_masked": len(track) - len(measured),
                "mean_reward": sum(row["reward"] for row in measured) / len(measured) if measured else None,
                "reward_stage": task_reward(
                    task,
                    {
                        "text_image_similarity_siglip2": {"render_mean": None},
                        "image_similarity_siglip2": {"score_mean": None},
                    },
                )[1],
            }
    (args.run / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Scored {sum(len(group) for group in groups.values())} measured rollouts; retained {len(rows)} rows")


if __name__ == "__main__":
    main()
