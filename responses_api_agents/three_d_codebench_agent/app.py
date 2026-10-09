# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode Blender episodes; native learned scoring runs after generation."""

import json
import logging
from pathlib import Path
from typing import Literal
from uuid import uuid4

from anyio import CancelScope
from fastapi import Request
from pydantic import ConfigDict, Field, JsonValue, field_validator

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_observability import AgentObservationBundle, TrajectoryRecord
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from responses_api_agents.opencode_agent.observability import scope_opencode_trajectory
from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgent, OpenCodeSandboxedAgentConfig


LOG = logging.getLogger(__name__)
GYM_ROOT = Path(__file__).resolve().parents[2]
Task = Literal["text_to_3d", "image_to_3d"]
VIEWS = ["Image_005.png", "Image_015.png", "Image_025.png", "Image_035.png"]


def assistant_answer(response: NeMoGymResponse) -> str:
    """Extract the final assistant source without tool outputs or earlier attempts."""
    for item in reversed(response.output):
        if item.type == "message" and item.role == "assistant":
            return "\n".join(part.text for part in item.content if part.type == "output_text")
    return ""


class ThreeDCodeBenchConfig(OpenCodeSandboxedAgentConfig):
    """Prepared references never enter generation or prediction sandboxes."""

    resources_server: ResourcesServerRef | None = None
    dataset_root: Path
    results_dir: Path
    execution_timeout: float = Field(default=300, gt=0)

    @field_validator("dataset_root", "results_dir")
    @classmethod
    def resolve_paths(cls, value: Path) -> Path:
        return (GYM_ROOT / value).resolve()

    @field_validator("artifacts_dir")
    @classmethod
    def resolve_artifacts(cls, value: str | None) -> str | None:
        return str((GYM_ROOT / value).resolve()) if value else None


class ThreeDRunRequest(BaseRunRequest):
    """One text or four-view image reconstruction episode."""

    model_config = ConfigDict(extra="allow")
    task: Task
    record_id: str = Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


class ThreeDVerifyResponse(BaseVerifyResponse):
    """Execution-stage result; offline native scoring adds learned rewards."""

    model_config = ConfigDict(extra="allow")
    task: Task
    record_id: str
    status: str
    artifact_dir: str
    protocol: str = "3dcodebench-opencode"
    upstream_revision: str = "42c7780ed3fcbd466f17f058f62e7996233777f7"
    dataset_revision: str = "0db485b63a2991231f0fad6c6e0dc01bcbbced1e"
    reward_stage: str = "executability"
    opencode_execution: dict[str, JsonValue] = Field(default_factory=dict)
    ng_agent_observations: AgentObservationBundle | None = Field(default=None, exclude_if=lambda value: value is None)
    ng_trajectory: TrajectoryRecord | None = Field(default=None, exclude_if=lambda value: value is None)


def directory_for(root: Path, *, task: str, record_id: str) -> Path:
    directory = (root / task / record_id).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError("Task path escapes prepared dataset")
    return directory


class ThreeDCodeBenchAgent(OpenCodeSandboxedAgent):
    """Collect solution.py, destroy agent sandbox, and reexecute independently."""

    ray_enabled = False
    config: ThreeDCodeBenchConfig

    async def _grade(self, program: Path, artifacts: Path) -> dict[str, JsonValue]:
        sandbox = await self._start_sandbox()
        try:
            await sandbox.upload(program, "/workspace/solution.py")
            for helper in ("render.py", "export_glb.py"):
                await sandbox.upload(
                    GYM_ROOT / "benchmarks/three_d_codebench/runtime" / helper, f"/workspace/{helper}"
                )
            commands = {
                "render": "/usr/bin/blender --background --factory-startup -t 2 --python-exit-code 1 "
                "--python /workspace/render.py -- --blender-render "
                "--script /workspace/solution.py --output-dir /workspace/renders "
                "--resolution 512 --samples 64 --engine CYCLES",
                "export": "/usr/bin/blender --background --factory-startup -t 2 --python-exit-code 1 "
                "--python /workspace/export_glb.py -- --blender-export "
                "--script /workspace/solution.py --out-glb /workspace/prediction.glb "
                "--log-path /workspace/export_log.json",
            }
            for stage, command in commands.items():
                execution_reward = 1.0 if stage == "export" else 0.0
                result = await sandbox.exec(command, timeout_s=self.config.execution_timeout)
                (artifacts / f"{stage}.stdout").write_text(result.stdout or "")
                (artifacts / f"{stage}.stderr").write_text(result.stderr or "")
                if result.error_type == "timeout":
                    return {"reward": execution_reward, "status": f"{stage}_timeout"}
                if result.error_type:
                    raise RuntimeError(f"Blender sandbox {stage} failed: {result.error_type}: {result.stderr}")
                if result.return_code != 0:
                    return {"reward": execution_reward, "status": f"{stage}_process_error"}
                local_log = artifacts / ("renders/render_log.json" if stage == "render" else "glb/export_log.json")
                local_log.parent.mkdir(parents=True, exist_ok=True)
                remote_log = (
                    "/workspace/renders/render_log.json" if stage == "render" else "/workspace/export_log.json"
                )
                exists = await sandbox.exec(f"test -s {remote_log}", timeout_s=30)
                if exists.error_type:
                    raise RuntimeError(f"Cannot check verification log: {exists.stderr}")
                if exists.return_code != 0:
                    return {"reward": execution_reward, "status": f"{stage}_missing_log"}
                await sandbox.download(remote_log, local_log)
                log = json.loads(local_log.read_text())
                if log["status"] != "OK":
                    # Native similarity metrics also report partially rendered predictions.
                    if stage == "render":
                        for view in VIEWS:
                            present = await sandbox.exec(f"test -s /workspace/renders/{view}", timeout_s=30)
                            if present.error_type:
                                raise RuntimeError(f"Cannot inspect partial render: {present.stderr}")
                            if present.return_code == 0:
                                await sandbox.download(f"/workspace/renders/{view}", artifacts / "renders" / view)
                    return {"reward": execution_reward, "status": log["status"], "native_error": log.get("error")}
                if stage == "render":
                    if log.get("n_views_rendered") != 4 or log.get("n_meshes", 0) < 1:
                        return {"reward": 0.0, "status": "invalid_render_log"}
                    for view in VIEWS:
                        await sandbox.download(f"/workspace/renders/{view}", artifacts / "renders" / view)
                else:
                    await sandbox.download("/workspace/prediction.glb", artifacts / "glb/prediction.glb")
            return {"reward": 1.0, "status": "ok"}
        finally:
            with CancelScope(shield=True):
                await sandbox.stop()

    async def run(self, request: Request, body: ThreeDRunRequest) -> ThreeDVerifyResponse:
        async with self._sem:
            directory = directory_for(self.config.dataset_root, task=body.task, record_id=body.record_id)
            metadata = json.loads((directory / "task.json").read_text())
            if (metadata["task"], metadata["record_id"]) != (body.task, body.record_id):
                raise ValueError("Requested task does not match prepared metadata")
            artifacts = self.config.results_dir / uuid4().hex
            artifacts.mkdir(parents=True)
            (artifacts / "input.json").write_text(body.model_dump_json())
            key = request.session[SESSION_ID_KEY]
            request._cookies = request.cookies | {"sandbox_id": key}
            rollout_id = self.rollout_id_from_run(body)
            request.state._ng_observation_invocation_id = rollout_id
            response = NeMoGymResponse(
                id=uuid4().hex,
                created_at=0,
                model="3dcodebench",
                object="response",
                output=[],
                parallel_tool_calls=True,
                tool_choice="auto",
                tools=[],
            )
            sandbox = None
            run_result = {}
            observations = trajectory = None
            try:
                sandbox = await self._start_sandbox()
                self._sandbox_id_to_sandbox[key] = sandbox
                for name in metadata["images"]:
                    image = (directory / name).resolve()
                    if Path(name).name != name or not image.is_relative_to(directory):
                        raise ValueError("Unsafe image path")
                    await sandbox.upload(image, f"/workspace/{name}")
                response = await self.responses(request, body.responses_create_params)
                run_result = self._sandbox_id_to_run_result.get(key, {}).copy()
                observations = run_result.pop("_ng_agent_observations", None)
                trajectory = run_result.pop("_ng_trajectory", None)
                if trajectory is not None:
                    trajectory = scope_opencode_trajectory(trajectory, body, rollout_id)
                (artifacts / "response.json").write_text(response.model_dump_json())
                (artifacts / "answer.txt").write_text(assistant_answer(response))
                if run_result.get("opencode_failed"):
                    timeout = (
                        run_result.get("opencode_error_type") == "timeout"
                        or run_result.get("opencode_exit_code") == 124
                    )
                    budget = timeout or response.status == "incomplete"
                    result = {
                        "reward": 0.0,
                        "status": "agent_timeout" if timeout else "context_limit" if budget else "agent_error",
                        "mask_sample": not budget,
                    }
                else:
                    exists = await sandbox.exec("test -s /workspace/solution.py", timeout_s=30)
                    if exists.error_type:
                        raise RuntimeError(f"Cannot collect program: {exists.stderr}")
                    if exists.return_code != 0:
                        result = {"reward": 0.0, "status": "missing_program"}
                    else:
                        program = artifacts / "solution.py"
                        await sandbox.download("/workspace/solution.py", program)
                        await sandbox.stop()
                        sandbox = None
                        result = await self._grade(program, artifacts)
            except Exception as exc:
                LOG.exception("3DCodeBench infrastructure failure for %s/%s", body.task, body.record_id)
                result = {
                    "reward": 0.0,
                    "status": "infrastructure_error",
                    "mask_sample": True,
                    "failure_kind": "3dcodebench:infrastructure_error",
                    "failure_reason": str(exc),
                }
            finally:
                with CancelScope(shield=True):
                    try:
                        if sandbox is not None:
                            await sandbox.stop()
                    finally:
                        self._sandbox_id_to_sandbox.pop(key, None)
                        self._sandbox_id_to_run_result.pop(key, None)
                        del request.state._ng_observation_invocation_id
            verified = ThreeDVerifyResponse(
                **body.model_dump(),
                response=response,
                artifact_dir=str(artifacts),
                opencode_execution=run_result,
                ng_agent_observations=observations,
                ng_trajectory=trajectory,
                **result,
            )
            (artifacts / "result.json").write_text(verified.model_dump_json())
            return verified


if __name__ == "__main__":
    ThreeDCodeBenchAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ThreeDCodeBenchAgent.run_webserver()
