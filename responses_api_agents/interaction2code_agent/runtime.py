# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provision a Gym-owned scoring runtime without loading an external checkout."""

import asyncio
import json
import os
import signal
import subprocess
from pathlib import Path
from uuid import uuid4


HERE = Path(__file__).resolve().parent


def ensure_runtime(root: Path) -> Path:
    """Install isolated metric dependencies; persistent workers load the models."""
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
    return python


class _Worker:
    def __init__(self) -> None:
        self.process: asyncio.subprocess.Process | None = None
        self.log_path: Path | None = None

    async def stop(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await process.wait()


class ScorerPool:
    """Bounded persistent CPU workers in the isolated Python scoring environment.

    Each worker processes one request at a time and retains CLIP/EasyOCR in RAM.
    Failed or cancelled workers are killed before their slot can be reused.
    """

    def __init__(self, *, python: Path, runtime_root: Path, concurrency: int = 2, threads: int = 2) -> None:
        if concurrency < 1 or threads < 1:
            raise ValueError("Scoring concurrency and threads must be positive")
        self.python = python
        self.runtime_root = runtime_root
        self.threads = threads
        self._workers = [_Worker() for _ in range(concurrency)]
        self._available: asyncio.Queue[_Worker] = asyncio.Queue()
        for worker in self._workers:
            self._available.put_nowait(worker)
        self._closed = False

    async def _start(self, worker: _Worker) -> None:
        if worker.process is not None and worker.process.returncode is None:
            return
        await worker.stop()
        logs = self.runtime_root / "worker-logs"
        logs.mkdir(parents=True, exist_ok=True)
        worker.log_path = logs / f"{uuid4().hex}.log"
        env = os.environ | {
            "I2C_MODEL_CACHE": str(self.runtime_root / "models"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "I2C_SCORING_THREADS": str(self.threads),
        }
        with worker.log_path.open("wb") as log:
            worker.process = await asyncio.create_subprocess_exec(
                str(self.python),
                str(HERE / "scoring_worker.py"),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=log,
                start_new_session=True,
            )
        if self._closed:
            await worker.stop()
            raise RuntimeError("Scorer pool is closed")
        ready = await self._receive(worker)
        if not ready.get("ready"):
            raise RuntimeError(f"Scoring worker did not become ready; see {worker.log_path}")

    async def _receive(self, worker: _Worker) -> dict:
        assert worker.process is not None and worker.process.stdout is not None
        line = await worker.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Scoring worker exited; see {worker.log_path}")
        return json.loads(line)

    async def score(self, *, dataset_root: Path, task_dir: Path, artifacts: Path, timeout: float) -> dict:
        """Score downloaded images; timeout covers queueing, startup, and evaluation."""
        if self._closed:
            raise RuntimeError("Scorer pool is closed")
        try:
            async with asyncio.timeout(timeout):
                worker = await self._available.get()
                try:
                    if self._closed:
                        raise RuntimeError("Scorer pool is closed")
                    await self._start(worker)
                    process = worker.process
                    assert process is not None and process.stdin is not None
                    payload = {"dataset": str(dataset_root), "task": str(task_dir), "artifacts": str(artifacts)}
                    process.stdin.write((json.dumps(payload) + "\n").encode())
                    await process.stdin.drain()
                    reply = await self._receive(worker)
                except BaseException:
                    await asyncio.shield(worker.stop())
                    raise
                finally:
                    self._available.put_nowait(worker)
                if "error" in reply:
                    raise RuntimeError(f"Interaction2Code scoring failed: {reply['error']}")
                return reply["result"]
        except TimeoutError as error:
            raise TimeoutError(f"Interaction2Code scoring exceeded {timeout} seconds") from error

    async def close(self) -> None:
        """Stop all owned workers at server shutdown, including busy workers."""
        self._closed = True
        await asyncio.gather(*(worker.stop() for worker in self._workers))
