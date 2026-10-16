# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit saved image reads and revisions against retained agent preview files."""

import argparse
import json
import re
from pathlib import Path

from benchmarks.three_d_codebench.prepare import VIEWS


def audit(run: Path) -> dict:
    """Require successful reads of four reference views and two revised preview rounds."""
    tasks = []
    for line in (run / "rollouts_generation.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["task"] != "image_to_3d":
            continue
        artifact = run / "artifacts" / Path(row["artifact_dir"]).name
        output = row["response"]["output"]
        results = {
            item.get("call_id"): item.get("output") for item in output if item["type"] == "function_call_output"
        }
        reads = {}
        revisions = []
        for index, item in enumerate(output):
            if item["type"] != "function_call":
                continue
            try:
                args = json.loads(item.get("arguments", "{}"))
            except (ValueError, TypeError):
                continue
            result = str(results.get(item.get("call_id"), ""))
            if item["name"] == "read" and "Image read successfully" in result:
                reads[str(args.get("filePath", ""))] = index
            if item["name"] in ("write", "edit", "apply_patch", "bash"):
                serialized = json.dumps(args)
                command = str(args.get("command", ""))
                bash_write = re.search(
                    r"(?:>{1,2}\s*[\"']?(?:/workspace/)?solution\.py\b|\.write_text\(|"
                    r"\bsed\s+-[^\s]*i\b|\bopen\([^)]*[\"']w[\"'])",
                    command,
                )
                if "solution.py" in serialized and (item["name"] != "bash" or bash_write):
                    if result and "Error:" not in result:
                        revisions.append(index)
        rounds = []
        for name in ("round1", "round2", "round3"):
            paths = [f"/workspace/feedback/{name}/{view}" for view in VIEWS]
            inspected = all(path in reads for path in paths)
            log = artifact / "feedback" / name / "render_log.json"
            render = json.loads(log.read_text()) if log.is_file() else {}
            rounds.append(
                {
                    "round": name,
                    "all_four_images_saved": all((artifact / "feedback" / name / view).is_file() for view in VIEWS),
                    "all_four_images_read": inspected,
                    "last_read_index": max((reads[path] for path in paths if path in reads), default=-1),
                    "first_read_index": min((reads[path] for path in paths if path in reads), default=-1),
                    "render_status": render.get("status"),
                }
            )
        first, second = rounds[:2]
        revised = (
            first["all_four_images_read"]
            and second["all_four_images_read"]
            and any(first["last_read_index"] < index < second["first_read_index"] for index in revisions)
        )
        reference_read = all(f"/workspace/{view}" in reads for view in VIEWS)
        tasks.append(
            {
                "record_id": row["record_id"],
                "status": row["status"],
                "all_reference_images_read": reference_read,
                "rounds": rounds,
                "revision_between_inspected_rounds": revised,
                "visual_feedback_verified": reference_read
                and revised
                and all(r["all_four_images_saved"] and r["render_status"] == "OK" for r in rounds[:2]),
            }
        )
    return {
        "tasks": tasks,
        "evaluated": len(tasks),
        "visual_feedback_verified": sum(task["visual_feedback_verified"] for task in tasks),
        "note": "Saved images and successful image-read tool results establish visual inspection; revision detection is conservative.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.run.resolve())
    (args.run / "visual_feedback_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
