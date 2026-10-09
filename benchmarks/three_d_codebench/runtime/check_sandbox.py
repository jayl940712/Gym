# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real prediction verifier with valid and failing Blender programs."""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, create_provider
from responses_api_agents.three_d_codebench_agent.app import ThreeDCodeBenchAgent


image, output = map(Path, sys.argv[1:])
scratch = Path(os.environ["TMPDIR"])
configuration = {
    "enroot": {
        "create": {
            "data_path": str(scratch / "data"),
            "cache_path": str(scratch / "cache"),
            "runtime_path": str(scratch / "runtime"),
            "bypass_entrypoint": False,
            "extra_start_args": ["--conf", str(output / "enroot-start.sh")],
        }
    }
}
output.mkdir(parents=True, exist_ok=True)
(output / "enroot-start.sh").write_text("unset ENROOT_MOUNT_HOME\n")


async def start():
    sandbox = AsyncSandbox(create_provider(configuration))
    await sandbox.start(
        SandboxSpec(
            image=str(image),
            workdir="/workspace",
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"},
        )
    )
    return sandbox


async def main():
    verifier = SimpleNamespace(config=SimpleNamespace(execution_timeout=300), _start_sandbox=start)
    cases = {
        "valid": 'import bpy\nbpy.ops.object.select_all(action="SELECT")\nbpy.ops.object.delete()\n'
        "bpy.ops.mesh.primitive_uv_sphere_add()\n",
        "invalid": 'raise RuntimeError("intentional verification test")\n',
    }
    results = {}
    for name, code in cases.items():
        folder = output / name
        folder.mkdir(exist_ok=True)
        program = folder / "solution.py"
        program.write_text(code)
        result = await ThreeDCodeBenchAgent._grade(verifier, program, folder)
        print(name, result, flush=True)
        results[name] = result
    assert results["valid"]["status"] == "ok", results
    assert results["valid"]["reward"] == 1
    assert results["invalid"]["status"] == "ERR_EXEC", results
    assert results["invalid"]["reward"] == 0
    (output / "result.json").write_text(json.dumps(results, indent=2) + "\n")


asyncio.run(main())
