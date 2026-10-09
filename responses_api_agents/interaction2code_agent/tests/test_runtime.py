# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from pathlib import Path

import pytest

from responses_api_agents.interaction2code_agent import runtime


def test_runtime_provisions_its_own_venv_and_locked_codec_dependencies(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    python = runtime.ensure_runtime(tmp_path)
    assert python == tmp_path / ".venv/bin/python"
    assert calls[0][0][:4] == ["uv", "venv", "--python", "3.12"]
    assert calls[1][0][:4] == ["uv", "--no-config", "pip", "install"]
    assert calls[1][0][-1].endswith("scoring.lock")
    assert calls[2][0][-1] == "--prefetch"
    assert calls[2][1]["env"]["I2C_MODEL_CACHE"] == str(tmp_path / "models")
    python.parent.mkdir(parents=True)
    python.touch()
    calls.clear()
    runtime.ensure_runtime(tmp_path)
    assert len(calls) == 2  # Reuse an existing interpreter, but check dependencies/weights.


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["success", "error", "timeout", "cancel"])
async def test_real_worker_success_failure_deadline_and_cancellation(tmp_path, monkeypatch, case):
    script = tmp_path / "scorer.py"
    scripts = {
        "success": "import json,os,sys; assert os.environ['I2C_SCORING_CONCURRENCY'] == '3'; assert os.environ['I2C_SCORING_THREADS'] == '1'; print(json.dumps({'reward':0.73})); print('diagnostic',file=sys.stderr)",
        "error": "import sys; print('bad metric input',file=sys.stderr); sys.exit(2)",
        "timeout": "import time; time.sleep(10)",
        "cancel": "import time; time.sleep(10)",
    }
    script.write_text(scripts[case])
    monkeypatch.setattr(runtime, "HERE", tmp_path)
    coroutine = runtime.score(
        python=Path(sys.executable),
        runtime_root=tmp_path,
        dataset_root=tmp_path,
        task_dir=tmp_path,
        artifacts=tmp_path,
        timeout=0.1 if case == "timeout" else 10,
        concurrency=3,
        threads=1,
    )
    if case == "success":
        assert await coroutine == {"reward": 0.73}
        assert (tmp_path / "scorer.stderr.txt").read_text() == "diagnostic\n"
    elif case == "error":
        with pytest.raises(RuntimeError, match="bad metric input"):
            await coroutine
    elif case == "timeout":
        with pytest.raises(TimeoutError):
            await coroutine
    else:
        task = asyncio.create_task(coroutine)
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
