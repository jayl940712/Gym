# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit actual image reads, page revisions, and retained browser preview rounds."""

import argparse
import json
import re
from pathlib import Path


def inspect_episode(artifact: Path) -> dict:
    """Use successful tool results and captures as evidence of visual iteration."""
    output = json.loads((artifact / "response.json").read_text())["output"]
    results = {item.get("call_id"): item.get("output") for item in output if item["type"] == "function_call_output"}
    reads = {}
    edits = []
    for index, item in enumerate(output):
        if item["type"] != "function_call":
            continue
        try:
            args = json.loads(item.get("arguments", "{}"))
        except (ValueError, TypeError):
            continue
        result = str(results.get(item.get("call_id"), ""))
        if item["name"] == "read" and "Image read successfully" in result:
            reads.setdefault(str(args.get("filePath", "")), []).append(index)
        if "index.html" not in json.dumps(args) or not result or "Error:" in result:
            continue
        command = str(args.get("command", ""))
        bash_write = re.search(
            r"(?:>{1,2}\s*[\"']?(?:/workspace/)?index\.html\b|\.write_text\(|"
            r"\bsed\s+-[^\s]*i\b|\bopen\([^)]*[\"']w[\"'])",
            command,
        )
        file_mutation = item["name"] in ("write", "edit") and Path(str(args.get("filePath", ""))).name == "index.html"
        patch_mutation = item["name"] == "apply_patch" and bool(
            re.search(
                r"\*\*\* (?:Update|Add) File: (?:/workspace/)?index\.html(?:\n|$)",
                json.dumps(args).replace("\\n", "\n"),
            )
        )
        if file_mutation or patch_mutation or (item["name"] == "bash" and bash_write):
            edits.append(index)
    rounds = []
    for directory in sorted((artifact / "feedback").glob("round*"), key=lambda p: p.name):
        report = directory / "render.json"
        render = json.loads(report.read_text()) if report.is_file() else {}
        images = sorted(p for p in directory.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        initial = next(
            (p for p in images if re.search(r"(?:^|[_-])(before|initial|source)(?:[_-]|$)", p.stem, re.I)),
            None,
        )
        clicked = next(
            (p for p in images if re.search(r"(?:^|[_-])(after|clicked|click|changed)(?:[_-]|$)", p.stem, re.I)),
            None,
        )
        paths = [p for p in (initial, clicked) if p is not None]
        indices = [
            i
            for p in paths
            for i in reads.get("/workspace/feedback/" + p.relative_to(artifact / "feedback").as_posix(), [])
        ]
        all_read = len(paths) == 2 and all(
            reads.get("/workspace/feedback/" + p.relative_to(artifact / "feedback").as_posix()) for p in paths
        )
        rounds.append(
            {
                "round": directory.name,
                "render_status": render.get("status", "agent_managed"),
                "state_images": [p.name for p in paths],
                "click_errors": render.get("click_errors", []),
                "both_images_saved": len(paths) == 2 and all(p.is_file() for p in paths),
                "both_images_read": all_read,
                "first_read_index": min(indices, default=-1),
                "last_read_index": max(indices, default=-1),
            }
        )
    valid = [
        r
        for r in rounds
        if r["both_images_saved"] and r["both_images_read"] and r["render_status"] in ("ok", "agent_managed")
    ]
    valid.sort(key=lambda r: r["first_read_index"])
    revised = any(
        a["last_read_index"] < index < b["first_read_index"] for a, b in zip(valid, valid[1:]) for index in edits
    )
    references = all(reads.get("/workspace/" + name) for name in ("before.png", "after.png"))
    used_feedback = references and bool(valid)
    revised_after_feedback = bool(valid) and any(index > valid[0]["last_read_index"] for index in edits)
    task_id = artifact.name.rsplit("-", 1)[0]
    return {
        "task_id": task_id,
        "artifact_dir": str(artifact),
        "references_read": references,
        "rounds": rounds,
        "inspected_rounds": len(valid),
        "revision_between_inspected_rounds": revised,
        "visual_feedback_used": used_feedback,
        "post_feedback_html_revision": revised_after_feedback,
        "visual_feedback_verified": references and len(valid) >= 2 and revised,
    }


def audit(run: Path) -> dict:
    """Read small episode responses instead of multi-gigabyte model histories."""
    tasks = [inspect_episode(path.parent) for path in sorted((run / "artifacts/episodes").glob("*/response.json"))]
    return {
        "tasks": tasks,
        "evaluated": len(tasks),
        "visual_feedback_used": sum(t["visual_feedback_used"] for t in tasks),
        "visual_feedback_verified": sum(t["visual_feedback_verified"] for t in tasks),
        "note": "Successful image reads, saved initial/clicked captures, and intervening HTML mutations establish visual feedback; agent-managed state names are inferred from filenames and browser/click behavior must also be inspected in the trajectory.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run.resolve())
    (args.run / "visual_feedback_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
