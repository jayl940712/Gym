# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interaction2Code with isolated OpenCode generation and native visual metrics."""

import json
import logging
from pathlib import Path
from uuid import uuid4

from anyio import CancelScope
from fastapi import Request
from pydantic import ConfigDict, Field, JsonValue, field_validator

from benchmarks.interaction2code.prepare import DATASET_REVISION, PROTOCOL, UPSTREAM_REVISION
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_observability import AgentObservationBundle, TrajectoryRecord
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from responses_api_agents.interaction2code_agent.runtime import ensure_runtime, score
from responses_api_agents.opencode_agent.observability import scope_opencode_trajectory
from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgent, OpenCodeSandboxedAgentConfig


LOG = logging.getLogger(__name__)
GYM_ROOT = Path(__file__).resolve().parents[2]


class Interaction2CodeConfig(OpenCodeSandboxedAgentConfig):
    """Keep prepared assets, results and the metric runtime inside the Gym tree."""

    resources_server: ResourcesServerRef | None = None
    dataset_root: Path = GYM_ROOT / "benchmarks/interaction2code/data"
    runtime_root: Path = GYM_ROOT / "benchmarks/interaction2code/.cache/scorer"
    results_dir: Path = GYM_ROOT / "results/interaction2code"
    execution_timeout: float = Field(default=300, gt=0)
    scoring_timeout: float = Field(default=900, gt=0)
    scoring_concurrency: int = Field(default=2, ge=1)
    scoring_threads: int = Field(default=2, ge=1)

    @field_validator("dataset_root", "runtime_root", "results_dir")
    @classmethod
    def resolve_paths(cls, value: Path) -> Path:
        """Resolve paths independently of per-server working directories."""
        return (GYM_ROOT / value).resolve()

    @field_validator("artifacts_dir")
    @classmethod
    def resolve_artifacts(cls, value: str | None) -> str | None:
        """Store the native OpenCode export alongside verification artifacts."""
        return str((GYM_ROOT / value).resolve()) if value else None


class Interaction2CodeRunRequest(BaseRunRequest):
    """One interaction is one independent screenshot-pair coding trial."""

    model_config = ConfigDict(extra="allow")
    page_id: int = Field(ge=1, le=127)
    interaction_id: int = Field(ge=1)


class Interaction2CodeVerifyResponse(BaseVerifyResponse):
    """All native metrics, with interaction CLIP as the Gym scalar reward."""

    model_config = ConfigDict(extra="allow")
    page_id: int
    interaction_id: int
    status: str
    protocol: str = PROTOCOL
    upstream_revision: str = UPSTREAM_REVISION
    dataset_revision: str = DATASET_REVISION
    artifact_dir: str
    full_page: dict[str, float | None] = Field(default_factory=dict)
    interaction: dict[str, float | None] = Field(default_factory=dict)
    interaction_flag: bool = False
    diagnostic: str = ""
    opencode_execution: dict[str, JsonValue] = Field(default_factory=dict)
    ng_agent_observations: AgentObservationBundle | None = Field(default=None, exclude_if=lambda value: value is None)
    ng_trajectory: TrajectoryRecord | None = Field(default=None, exclude_if=lambda value: value is None)


def task_directory(root: Path, body: Interaction2CodeRunRequest) -> Path:
    """Prevent task-directory symlinks from escaping the prepared dataset."""
    directory = (root / "tasks" / f"{body.page_id}-{body.interaction_id}").resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError("Interaction2Code task path escapes dataset_root")
    return directory


def failed_result(*, status: str, infrastructure: bool, reason: str = "") -> dict:
    """Preserve the native field set for both task failures and infrastructure errors."""
    full = {"clip_similarity": 0.0, "text_similarity": 0.0, "structure_similarity": 0.0}
    return {
        "reward": 0.0,
        "status": status,
        "mask_sample": infrastructure,
        "diagnostic": reason,
        "full_page": full,
        "interaction": full | {"position_similarity": 0.0, "position_similarity_after": 0.0},
        "interaction_flag": False,
        "failure_kind": f"interaction2code:{status}",
    }


class Interaction2CodeAgent(OpenCodeSandboxedAgent):
    """Code in one sandbox, render in another, score images in a trusted process."""

    ray_enabled = False
    config: Interaction2CodeConfig

    def model_post_init(self, context: object, /) -> None:
        """Provision the isolated native scorer before accepting episodes."""
        super().model_post_init(context)
        ensure_runtime(self.config.runtime_root)

    async def _grade(self, directory: Path, artifacts: Path, metadata: dict) -> dict:
        sandbox = await self._start_sandbox()
        try:
            await sandbox.upload(artifacts / "index.html", "/workspace/index.html")
            await sandbox.upload(directory / "placeholder.jpg", "/workspace/placeholder.jpg")
            result = await sandbox.exec(
                f"python /opt/interaction2code/render.py --html /workspace/index.html --output /workspace/render --width {metadata['width']} --height {metadata['height']}",
                timeout_s=self.config.execution_timeout,
            )
            (artifacts / "browser-execution.json").write_text(
                json.dumps(
                    {
                        "exit_code": result.return_code,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "error_type": result.error_type,
                    },
                    indent=2,
                )
            )
            if result.error_type == "timeout" or result.return_code == 124:
                return failed_result(status="browser_timeout", infrastructure=False)
            if result.error_type or result.return_code:
                raise RuntimeError(f"Browser sandbox failed: {result.error_type}, {result.stderr}")
            await sandbox.download("/workspace/render/render.json", artifacts / "render.json")
            rendered = json.loads((artifacts / "render.json").read_text())
            if rendered.get("status") == "task_error":
                return failed_result(status="browser_error", infrastructure=False, reason=rendered.get("error", ""))
            if rendered.get("status") != "ok":
                raise RuntimeError(f"Browser failed: {rendered}")
            for name in rendered["screenshots"]:
                if not isinstance(name, str) or Path(name).name != name or not name.endswith(".png"):
                    raise ValueError("Invalid browser screenshot path")
                await sandbox.download(f"/workspace/render/{name}", artifacts / name)
        except TimeoutError:
            return failed_result(status="browser_timeout", infrastructure=False)
        finally:
            with CancelScope(shield=True):
                await sandbox.stop()
        return await score(
            python=self.config.runtime_root / ".venv/bin/python",
            runtime_root=self.config.runtime_root,
            dataset_root=self.config.dataset_root,
            task_dir=directory,
            artifacts=artifacts,
            timeout=self.config.scoring_timeout,
            concurrency=self.config.scoring_concurrency,
            threads=self.config.scoring_threads,
        )

    async def run(self, request: Request, body: Interaction2CodeRunRequest) -> Interaction2CodeVerifyResponse:
        """Execute and verify one isolated native OpenCode episode."""
        async with self._sem:
            directory = task_directory(self.config.dataset_root, body)
            metadata = json.loads((directory / "task.json").read_text())
            if (metadata["page_id"], metadata["interaction_id"]) != (body.page_id, body.interaction_id):
                raise ValueError("Interaction2Code metadata does not match the requested interaction")
            artifacts = self.config.results_dir / "episodes" / f"{body.page_id}-{body.interaction_id}-{uuid4().hex}"
            artifacts.mkdir(parents=True)
            (artifacts / "input.json").write_text(body.model_dump_json())
            key = request.session[SESSION_ID_KEY]
            request._cookies = request.cookies | {"sandbox_id": key}
            rollout_id = self.rollout_id_from_run(body)
            request.state._ng_observation_invocation_id = rollout_id
            response = NeMoGymResponse(
                id=uuid4().hex,
                created_at=0,
                model="interaction2code",
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
                for name in ("before.png", "after.png", "placeholder.jpg"):
                    path = (directory / name).resolve()
                    if not path.is_relative_to(directory):
                        raise ValueError("Input asset escapes task directory")
                    await sandbox.upload(path, f"/workspace/{name}")
                response = await self.responses(request, body.responses_create_params)
                run_result = self._sandbox_id_to_run_result.get(key, {}).copy()
                observations = run_result.pop("_ng_agent_observations", None)
                trajectory = run_result.pop("_ng_trajectory", None)
                if trajectory is not None:
                    trajectory = scope_opencode_trajectory(trajectory, body, rollout_id)
                (artifacts / "response.json").write_text(response.model_dump_json())
                if run_result.get("opencode_failed"):
                    timed_out = (
                        run_result.get("opencode_error_type") == "timeout"
                        or run_result.get("opencode_exit_code") == 124
                    )
                    limited = response.status == "incomplete"
                    status = "agent_timeout" if timed_out else "context_limit" if limited else "agent_error"
                    result = failed_result(status=status, infrastructure=not (timed_out or limited))
                else:
                    exists = await sandbox.exec(
                        "test -f /workspace/index.html && test ! -L /workspace/index.html && test $(wc -c < /workspace/index.html) -le 16777216",
                        timeout_s=30,
                    )
                    if exists.error_type:
                        raise RuntimeError(f"Deliverable check failed: {exists.error_type}")
                    if exists.return_code:
                        result = failed_result(status="missing_html", infrastructure=False)
                    else:
                        await sandbox.download("/workspace/index.html", artifacts / "index.html")
                        await sandbox.stop()
                        sandbox = None
                        result = await self._grade(directory, artifacts, metadata)
            except Exception as error:
                LOG.exception("Interaction2Code infrastructure failure: %s/%s", body.page_id, body.interaction_id)
                result = failed_result(status="infrastructure_error", infrastructure=True, reason=str(error))
            finally:
                with CancelScope(shield=True):
                    try:
                        if sandbox is not None:
                            await sandbox.stop()
                    except Exception:
                        LOG.exception("Failed to stop Interaction2Code agent sandbox")
                    finally:
                        self._sandbox_id_to_sandbox.pop(key, None)
                        self._sandbox_id_to_run_result.pop(key, None)
                        del request.state._ng_observation_invocation_id
            verified = Interaction2CodeVerifyResponse(
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
    Interaction2CodeAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = Interaction2CodeAgent.run_webserver()
