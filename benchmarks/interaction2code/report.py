# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report native metrics and exact coverage for a one-repeat full evaluation."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def read_rows(path: Path) -> list[dict]:
    """Read saved Gym inputs or verified rollout rows."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def report(*, inputs: Path, rollouts: Path) -> dict:
    """Keep missing cases, masked failures and null OCR scores visible."""
    tasks, rows = read_rows(inputs), read_rows(rollouts)
    expected = {(row["page_id"], row["interaction_id"]) for row in tasks}
    counts = Counter((row["page_id"], row["interaction_id"]) for row in rows)
    missing = sorted(expected - counts.keys())
    unexpected = sorted(counts.keys() - expected)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    masked = [row for row in rows if row.get("mask_sample")]
    evaluated = [row for row in rows if not row.get("mask_sample")]
    retried = [row for row in rows if row.get("infrastructure_retry")]
    groups = {}
    for group, fields in (
        ("full_page", ("clip_similarity", "text_similarity", "structure_similarity")),
        (
            "interaction",
            (
                "clip_similarity",
                "text_similarity",
                "structure_similarity",
                "position_similarity",
                "position_similarity_after",
            ),
        ),
    ):
        groups[group] = {}
        for field in fields:
            values = [(row.get(group) or {}).get(field) for row in evaluated]
            present = [value for value in values if value is not None]
            groups[group][field] = {
                "mean": mean(present) if present else None,
                "scored_count": len(present),
                "null_or_missing_count": len(values) - len(present),
            }
    return {
        "protocol": "interaction2code-opencode",
        "expected_tasks": len(expected),
        "returned_rows": len(rows),
        "evaluated_rows": len(evaluated),
        "infrastructure_failures": len(masked),
        "infrastructure_retries": len(retried),
        "retried_task_ids": [f"{row['page_id']}-{row['interaction_id']}" for row in retried],
        "complete": not missing and not unexpected and not duplicates and not masked,
        "missing_task_ids": [f"{page}-{interaction}" for page, interaction in missing],
        "unexpected_task_ids": [f"{page}-{interaction}" for page, interaction in unexpected],
        "duplicate_task_ids": [f"{page}-{interaction}" for page, interaction in duplicates],
        "masked_task_ids": [f"{row['page_id']}-{row['interaction_id']}" for row in masked],
        "statuses": dict(Counter(row.get("status", "missing_status") for row in rows)),
        "interaction_flag_true": sum(bool(row.get("interaction_flag")) for row in evaluated),
        "interaction_flag_rate": mean(bool(row.get("interaction_flag")) for row in evaluated) if evaluated else None,
        "mean_reward": mean(row["reward"] for row in evaluated) if evaluated else None,
        "metrics": groups,
        "aggregation": "One row per interaction; masked infrastructure failures excluded, native zero scores included, null OCR scores excluded with counts reported.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=Path, default=Path(__file__).resolve().parent / "data/interaction2code_benchmark.jsonl"
    )
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = report(inputs=args.inputs, rollouts=args.rollouts)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if args.require_complete and not result["complete"] else 0)
