# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep model-issued broad process termination in the benchmark denominator."""

import json

from responses_api_agents.interaction2code_agent.failures import self_termination_reason


def test_pending_broad_cleanup_is_model_failure(tmp_path):
    """The observed SIGTERM after pkill must not become an infrastructure retry."""
    exported = tmp_path / "export.json"
    exported.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "parts": [
                            {
                                "type": "tool",
                                "tool": "bash",
                                "state": {
                                    "status": "running",
                                    "input": {"command": 'pkill -f "http.server" 2>/dev/null; sleep 0.5'},
                                },
                            }
                        ]
                    }
                ]
            }
        )
    )
    execution = {"opencode_exit_code": -15, "opencode_results_fpath": str(exported)}
    assert self_termination_reason(execution, "Start python -m http.server 0")
    assert not self_termination_reason(execution, "Unrelated prompt")
    execution["opencode_exit_code"] = 1
    assert not self_termination_reason(execution, "Start python -m http.server 0")


def test_completed_cleanup_does_not_explain_external_termination(tmp_path):
    """Do not blame an earlier successful tool when the process dies externally."""
    exported = tmp_path / "export.json"
    exported.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "parts": [
                            {
                                "type": "tool",
                                "tool": "bash",
                                "state": {
                                    "status": "completed",
                                    "input": {"command": 'pkill -f "http.server"'},
                                },
                            }
                        ]
                    }
                ]
            }
        )
    )
    assert not self_termination_reason(
        {"opencode_exit_code": -15, "opencode_results_fpath": str(exported)}, "http.server"
    )


def test_missing_export_keeps_infrastructure_classification():
    """Absent evidence must not silently unmask a failed process."""
    assert not self_termination_reason({"opencode_exit_code": -15}, "http.server")
