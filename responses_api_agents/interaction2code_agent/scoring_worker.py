# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent trusted metric worker; stdin/stdout carry one JSON object per line."""

import contextlib
import fcntl
import json
import os
import sys
import traceback
from pathlib import Path

from scorer import MODEL_CACHE, clip_model, evaluate, ocr_reader


def main() -> None:
    """Load models once, then score successive screenshot sets without HTML execution."""
    protocol = sys.stdout
    # Fresh runtimes must not race when CLIP/EasyOCR download shared weights.
    MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with (MODEL_CACHE.parent / "model-initialization.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with contextlib.redirect_stdout(sys.stderr):
            clip_model()
            ocr_reader()
        fcntl.flock(lock, fcntl.LOCK_UN)
    print(json.dumps({"ready": True, "pid": os.getpid()}), file=protocol, flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        artifacts = Path(request["artifacts"])
        with (artifacts / "scorer.stderr.txt").open("w") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    result = evaluate(
                        dataset=Path(request["dataset"]), task=Path(request["task"]), artifacts=artifacts
                    )
                    reply = {"result": result, "pid": os.getpid()}
                except Exception:
                    error = traceback.format_exc()
                    log.write(error)
                    reply = {"error": error, "pid": os.getpid()}
        print(json.dumps(reply, allow_nan=False), file=protocol, flush=True)


if __name__ == "__main__":
    main()
