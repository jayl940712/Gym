# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise worker reuse, isolation and recovery over real subprocess pipes."""

import asyncio
import sys
from pathlib import Path

import pytest

from responses_api_agents.interaction2code_agent import runtime


@pytest.fixture
def pool(tmp_path, monkeypatch):
    worker = tmp_path / "scoring_worker.py"
    worker.write_text("""import json, os, sys, time
print(json.dumps({"ready": True, "pid": os.getpid()}), flush=True)
count = 0
for line in sys.stdin:
    r = json.loads(line)
    name = r["task"]
    if name == "crash": os._exit(3)
    if name == "wait": time.sleep(30)
    if name == "slow": time.sleep(0.1)
    if name == "error":
        print(json.dumps({"error": "bad image"}), flush=True)
        continue
    count += 1
    print(json.dumps({"result": {"pid": os.getpid(), "count": count, "task": name}}), flush=True)
""")
    monkeypatch.setattr(runtime, "HERE", tmp_path)
    return runtime.ScorerPool(python=Path(sys.executable), runtime_root=tmp_path, concurrency=1)


async def call(pool, task="ok", timeout=5):
    return await pool.score(
        dataset_root=Path("data"), task_dir=Path(task), artifacts=Path("artifacts"), timeout=timeout
    )


@pytest.mark.asyncio
async def test_reuses_one_process_for_distinct_tasks_and_closes(pool):
    try:
        first, second = await call(pool, "first"), await call(pool, "second")
        assert first["pid"] == second["pid"]
        assert [first["count"], second["count"]] == [1, 2]
        assert [first["task"], second["task"]] == ["first", "second"]
        process = pool._workers[0].process
    finally:
        await pool.close()
    assert process.returncode is not None
    with pytest.raises(RuntimeError, match="closed"):
        await call(pool)


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_mix_replies(pool):
    try:
        rows = await asyncio.gather(*(call(pool, str(i)) for i in range(6)))
        assert [r["task"] for r in rows] == list(map(str, range(6)))
        assert len({r["pid"] for r in rows}) == 1
        assert sorted(r["count"] for r in rows) == list(range(1, 7))
    finally:
        await pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["crash", "wait", "cancel"])
async def test_bad_worker_is_killed_and_replaced(pool, failure):
    try:
        before = await call(pool)
        process = pool._workers[0].process
        if failure == "cancel":
            task = asyncio.create_task(call(pool, "wait"))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(TimeoutError if failure == "wait" else RuntimeError):
                await call(pool, failure, timeout=0.2)
        assert process.returncode is not None
        after = await call(pool, "recovered")
        assert after["pid"] != before["pid"] and after["count"] == 1
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_queue_timeout_does_not_kill_busy_worker(pool):
    try:
        before = await call(pool)
        busy = asyncio.create_task(call(pool, "slow"))
        await asyncio.sleep(0.01)
        with pytest.raises(TimeoutError):
            await call(pool, "queued", timeout=0.01)
        assert (await busy)["pid"] == before["pid"]
        assert (await call(pool))["count"] == 3
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_metric_error_is_reported_without_discarding_healthy_worker(pool):
    try:
        before = await call(pool)
        with pytest.raises(RuntimeError, match="bad image"):
            await call(pool, "error")
        assert (await call(pool))["pid"] == before["pid"]
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_pool_bounds_live_workers(tmp_path, monkeypatch, pool):
    workers = runtime.ScorerPool(python=Path(sys.executable), runtime_root=tmp_path, concurrency=2)
    try:
        rows = await asyncio.gather(*(call(workers, "slow") for _ in range(6)))
        assert len({r["pid"] for r in rows}) == 2
    finally:
        await workers.close()
        await pool.close()


@pytest.mark.parametrize("concurrency,threads", [(0, 2), (2, 0)])
def test_invalid_pool_size(tmp_path, concurrency, threads):
    with pytest.raises(ValueError):
        runtime.ScorerPool(
            python=Path(sys.executable), runtime_root=tmp_path, concurrency=concurrency, threads=threads
        )


def test_runtime_installs_locked_dependencies_without_discarded_model_prefetch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda args, **kwargs: calls.append(args))
    python = runtime.ensure_runtime(tmp_path)
    assert python == tmp_path / ".venv/bin/python"
    assert calls[0][:4] == ["uv", "venv", "--python", "3.12"]
    assert calls[1][:4] == ["uv", "--no-config", "pip", "install"]
    assert calls[1][-1].endswith("scoring.lock")
    assert len(calls) == 2
    python.parent.mkdir(parents=True)
    python.touch()
    calls.clear()
    runtime.ensure_runtime(tmp_path)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_startup_failure_reports_worker_log_and_allows_retry(pool, tmp_path):
    script = tmp_path / "scoring_worker.py"
    original = script.read_text()
    script.write_text("import sys; print('cannot load checkpoint', file=sys.stderr); sys.exit(2)")
    try:
        with pytest.raises(RuntimeError, match="Scoring worker exited; see"):
            await call(pool)
        assert "cannot load checkpoint" in pool._workers[0].log_path.read_text()
        assert pool._workers[0].process is None
        script.write_text(original)
        assert (await call(pool))["count"] == 1
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_closing_busy_pool_terminates_worker_and_releases_waiter(pool):
    await call(pool)
    process = pool._workers[0].process
    busy = asyncio.create_task(call(pool, "wait"))
    waiting = asyncio.create_task(call(pool, "queued"))
    await asyncio.sleep(0.05)
    await pool.close()
    results = await asyncio.gather(busy, waiting, return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in results)
    assert process.returncode is not None


@pytest.mark.asyncio
async def test_production_worker_loads_models_once_and_separates_diagnostics(tmp_path, monkeypatch):
    source = runtime.HERE / "scoring_worker.py"
    (tmp_path / "scoring_worker.py").write_text(source.read_text())
    (tmp_path / "scorer.py").write_text("""from pathlib import Path
MODEL_CACHE = Path(__file__).parent / "models"
counts = {"clip": 0, "ocr": 0}
def clip_model():
    counts["clip"] += 1
    print("load CLIP")
def ocr_reader():
    counts["ocr"] += 1
    print("load OCR")
def evaluate(*, dataset, task, artifacts):
    print("scoring diagnostics")
    if task.name == "bad": raise ValueError("corrupt PNG")
    return dict(counts, task=task.name)
""")
    monkeypatch.setattr(runtime, "HERE", tmp_path)
    worker = runtime.ScorerPool(python=Path(sys.executable), runtime_root=tmp_path, concurrency=1)

    async def request(name):
        return await worker.score(dataset_root=tmp_path, task_dir=Path(name), artifacts=tmp_path, timeout=5)

    try:
        assert await request("first") == {"clip": 1, "ocr": 1, "task": "first"}
        with pytest.raises(RuntimeError, match="corrupt PNG"):
            await request("bad")
        assert "corrupt PNG" in (tmp_path / "scorer.stderr.txt").read_text()
        assert await request("second") == {"clip": 1, "ocr": 1, "task": "second"}
        assert (tmp_path / "scorer.stderr.txt").read_text() == "scoring diagnostics\n"
        assert worker._workers[0].log_path.read_text() == "load CLIP\nload OCR\n"
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_timeout_during_initialization_kills_unready_worker(pool, tmp_path):
    script = tmp_path / "scoring_worker.py"
    original = script.read_text()
    script.write_text("import time; time.sleep(30)")
    try:
        with pytest.raises(TimeoutError):
            await call(pool, timeout=0.1)
        assert pool._workers[0].process is None
        script.write_text(original)
        assert (await call(pool))["count"] == 1
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_shutdown_while_spawning_does_not_leave_an_orphan(pool, monkeypatch):
    original = asyncio.create_subprocess_exec
    spawned, release = asyncio.Event(), asyncio.Event()
    processes = []

    async def delayed_spawn(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        spawned.set()
        await release.wait()
        return process

    monkeypatch.setattr(runtime.asyncio, "create_subprocess_exec", delayed_spawn)
    request = asyncio.create_task(call(pool))
    await spawned.wait()
    await pool.close()
    release.set()
    with pytest.raises(RuntimeError, match="closed"):
        await request
    assert processes[0].returncode is not None
