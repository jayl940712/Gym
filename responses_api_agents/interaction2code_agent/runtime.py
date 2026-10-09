# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provision a Gym-owned scoring runtime without loading an external checkout."""

import asyncio
import json
import os
import signal
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent


def ensure_runtime(root: Path) -> Path:
    """Install the isolated metric dependencies once and prefetch model weights."""
    root = root.resolve()
    python = root / ".venv/bin/python"
    if not python.exists():
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["uv", "venv", "--python", "3.12", str(root / ".venv")], check=True)
    # Gym core deliberately excludes codec packages; this separate runtime needs them.
    subprocess.run(
        ["uv", "--no-config", "pip", "install", "--python", str(python), "-r", str(HERE / "scoring.lock")],
        check=True,
    )
    env = os.environ | {"I2C_MODEL_CACHE": str(root / "models"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    subprocess.run([str(python), str(HERE / "scorer.py"), "--prefetch"], env=env, check=True)
    return python


async def score(
    *,
    python: Path,
    runtime_root: Path,
    dataset_root: Path,
    task_dir: Path,
    artifacts: Path,
    timeout: float,
    concurrency: int = 2,
    threads: int = 2,
) -> dict:
    """Execute trusted metrics on downloaded images; never execute generated HTML."""
    env = os.environ | {
        "I2C_MODEL_CACHE": str(runtime_root / "models"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "I2C_SCORING_CONCURRENCY": str(concurrency),
        "I2C_SCORING_THREADS": str(threads),
    }
    process = await asyncio.create_subprocess_exec(
        str(python),
        str(HERE / "scorer.py"),
        "--dataset",
        str(dataset_root),
        "--task",
        str(task_dir),
        "--artifacts",
        str(artifacts),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException as error:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
        if isinstance(error, TimeoutError):
            raise TimeoutError(f"Interaction2Code scoring exceeded {timeout} seconds") from error
        raise
    (artifacts / "scorer.stderr.txt").write_text(stderr.decode(errors="replace"))
    if process.returncode:
        raise RuntimeError(
            f"Interaction2Code scorer exited {process.returncode}: {stderr.decode(errors='replace')[-3000:]}"
        )
    return json.loads(stdout.decode(errors="replace"))
