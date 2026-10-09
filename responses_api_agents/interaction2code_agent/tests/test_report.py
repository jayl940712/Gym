# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from benchmarks.interaction2code.report import report


def test_report_accounts_for_missing_masked_duplicate_and_null_scores(tmp_path):
    inputs, rollouts = tmp_path / "inputs.jsonl", tmp_path / "rollouts.jsonl"
    tasks = [{"page_id": 1, "interaction_id": i} for i in range(1, 4)]
    inputs.write_text("".join(json.dumps(row) + "\n" for row in tasks))
    good = tasks[0] | {
        "reward": 0.8,
        "status": "ok",
        "interaction_flag": True,
        "interaction": {"clip_similarity": 0.8, "text_similarity": None},
    }
    bad = tasks[1] | {"reward": 0.0, "status": "infrastructure_error", "mask_sample": True}
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in [good, bad, good]))
    summary = report(inputs=inputs, rollouts=rollouts)
    assert not summary["complete"]
    assert summary["missing_task_ids"] == ["1-3"]
    assert summary["duplicate_task_ids"] == ["1-1"]
    assert summary["masked_task_ids"] == ["1-2"]
    assert summary["metrics"]["interaction"]["text_similarity"]["null_or_missing_count"] == 2
    assert summary["mean_reward"] == 0.8
    zero = tasks[1] | {"reward": 0.0, "status": "no_interaction", "interaction_flag": False}
    rollouts.write_text(
        "".join(json.dumps(row) + "\n" for row in [good, zero, tasks[2] | good | {"interaction_id": 3}])
    )
    summary = report(inputs=inputs, rollouts=rollouts)
    assert summary["complete"]
    assert summary["mean_reward"] == 1.6 / 3
