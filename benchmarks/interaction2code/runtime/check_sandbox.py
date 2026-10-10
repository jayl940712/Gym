# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check concurrent Enroot web servers and process isolation before GPU rollout."""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, create_provider


async def check(image: Path, output: Path) -> None:
    """Verify independent pages, distinct assigned ports, and scoped server cleanup."""
    output.mkdir(parents=True, exist_ok=True)
    conf = output / "enroot-start.sh"
    conf.write_text("unset ENROOT_MOUNT_HOME\n")
    with tempfile.TemporaryDirectory(prefix="i2c-preflight-") as scratch:
        provider = create_provider(
            {
                "enroot": {
                    "exec": {"concurrency": 128},
                    "create": {
                        "base_dir": scratch,
                        "bypass_entrypoint": False,
                        "extra_start_args": ["--conf", str(conf.resolve())],
                    },
                }
            }
        )
        sandboxes = [AsyncSandbox(provider), AsyncSandbox(provider)]
        try:
            for sandbox in sandboxes:
                await sandbox.start(SandboxSpec(image=str(image.resolve()), workdir="/"))
            ports, namespaces = [], []
            for index, sandbox in enumerate(sandboxes):
                result = await sandbox.exec(
                    f"mkdir -p /workspace; printf 'page-{index}' > /workspace/index.html; "
                    "setsid python3 -u -m http.server 0 --bind 127.0.0.1 --directory /workspace "
                    "> /tmp/server.log 2>&1 < /dev/null & echo $! > /tmp/server.pid; "
                    "sleep 1; cat /tmp/server.log; readlink /proc/self/ns/pid",
                    timeout_s=30,
                )
                assert result.return_code == 0, result
                lines = result.stdout.splitlines()
                port = int(lines[0].split(" port ")[1].split()[0])
                ports.append(port)
                namespaces.append(lines[-1])
                response = await sandbox.exec(f"curl -fsS --max-time 5 http://127.0.0.1:{port}/index.html")
                assert response.stdout == f"page-{index}", response
            assert ports[0] != ports[1], ports
            assert namespaces[0] != namespaces[1], namespaces
            stopped = await sandboxes[0].exec("kill $(cat /tmp/server.pid)")
            assert stopped.return_code == 0, stopped
            survived = await sandboxes[1].exec(f"curl -fsS --max-time 5 http://127.0.0.1:{ports[1]}/index.html")
            assert survived.stdout == "page-1", survived
            report = {
                "status": "passed",
                "ports": ports,
                "pid_namespaces": namespaces,
                "scoped_cleanup_preserves_other_episode": True,
            }
            (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
        finally:
            for sandbox in sandboxes:
                await sandbox.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    asyncio.run(check(args.image, args.output))
