# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interaction2Code with selectable generation and isolated native verification."""

import hashlib
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

from anyio import CancelScope
from fastapi import FastAPI, Request
from pydantic import ConfigDict, Field, JsonValue, PrivateAttr, field_validator

from benchmarks.interaction2code.prepare import DATASET_REVISION, PROTOCOL, UPSTREAM_REVISION
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymChatCompletion, NeMoGymResponse
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.rollout_observability import AgentObservationBundle, TrajectoryRecord
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from responses_api_agents.interaction2code_agent.failures import self_termination_reason
from responses_api_agents.interaction2code_agent.runtime import ScorerPool, ensure_runtime
from responses_api_agents.interaction2code_agent.single_turn import build_request, extract_html, response_text
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
    agent_mode: Literal["opencode", "single_turn"] = "opencode"
    single_turn_max_output_tokens: int = Field(default=32768, gt=0)
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
    agent_mode: Literal["opencode", "single_turn"] = "opencode"
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
    """Generate webpage code, then independently render and score it."""

    ray_enabled = False
    config: Interaction2CodeConfig
    _scorer_pool: ScorerPool | None = PrivateAttr(default=None)

    def model_post_init(self, context: object, /) -> None:
        """Provision the isolated native scorer before accepting episodes."""
        super().model_post_init(context)
        ensure_runtime(self.config.runtime_root)

    def setup_webserver(self) -> FastAPI:
        """Keep scorer processes alive between tasks and stop them with the server."""
        app = super().setup_webserver()
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application):
            try:
                async with original_lifespan(application) as state:
                    yield state
            finally:
                if self._scorer_pool is not None:
                    with CancelScope(shield=True):
                        await self._scorer_pool.close()

        app.router.lifespan_context = lifespan
        return app

    def _get_scorer_pool(self) -> ScorerPool:
        if self._scorer_pool is None:
            self._scorer_pool = ScorerPool(
                python=self.config.runtime_root / ".venv/bin/python",
                runtime_root=self.config.runtime_root,
                concurrency=self.config.scoring_concurrency,
                threads=self.config.scoring_threads,
            )
        return self._scorer_pool

    async def _single_turn(
        self, request: Request, body: Interaction2CodeRunRequest, directory: Path, artifacts: Path, metadata: dict
    ) -> tuple[NeMoGymResponse, dict]:
        """Make one tool-free model call, then verify its fenced HTML independently."""
        if self.config.model_server is None:
            raise ValueError("single_turn requires a Gym model_server")
        params = body.responses_create_params
        converter = ResponsesConverter(return_token_id_information=True)
        chat_params = converter.responses_to_chat_completion_create_params(params)
        chat_params.tools = []
        chat_params.tool_choice = "none"
        chat_params.parallel_tool_calls = False
        (artifacts / "model_request.json").write_text(chat_params.model_dump_json())
        raw = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self.url_path_for_run("/v1/chat/completions", body),
            json=chat_params,
            cookies=request.cookies,
        )
        await raise_for_status(raw)
        chat = NeMoGymChatCompletion.model_validate(await get_response_json(raw))
        (artifacts / "model_response.json").write_text(chat.model_dump_json())
        params = params.model_copy(update={"model": chat.model})
        response = converter.chat_completion_to_response(params, chat, preserve_envelope_id=True)
        (artifacts / "response.json").write_text(response.model_dump_json())
        text = response_text(response)
        (artifacts / "assistant_response.txt").write_text(text)
        if response.status == "incomplete":
            return response, failed_result(status="context_limit", infrastructure=False)
        if any(item.type == "function_call" for item in response.output):
            return response, failed_result(status="agent_error", infrastructure=False, reason="Unexpected tool call")
        try:
            html = extract_html(text)
        except ValueError as error:
            return response, failed_result(status="missing_html", infrastructure=False, reason=str(error))
        (artifacts / "index.html").write_text(html)
        return response, await self._grade(directory, artifacts, metadata)

    async def _collect_feedback(self, sandbox, artifacts: Path) -> None:
        """Retain optional preview images before disposing of the coding sandbox."""
        try:
            result = await sandbox.exec(
                "find /workspace/feedback -mindepth 1 -maxdepth 2 -type f 2>/dev/null || true",
                timeout_s=30,
            )
            for name in result.stdout.splitlines():
                relative = Path(name).relative_to("/workspace/feedback")
                if ".." in relative.parts or len(relative.parts) not in (1, 2):
                    continue
                if relative.suffix.lower() not in (".png", ".jpg", ".jpeg", ".json"):
                    continue
                destination = artifacts / "feedback" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                await sandbox.download(name, destination)
        except Exception:
            LOG.warning("Could not retain optional Interaction2Code previews", exc_info=True)

    async def _audit_inputs(self, sandbox, directory: Path, artifacts: Path) -> None:
        """Record input integrity without giving the policy any audit code or feedback."""
        paths = {name: directory / name for name in ("before.png", "after.png", "placeholder.jpg")}
        expected = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
        report = {"expected_sha256": expected, "observed_sha256": {}, "verified": False}
        try:
            result = await sandbox.exec(
                "sha256sum /workspace/before.png /workspace/after.png /workspace/placeholder.jpg",
                timeout_s=30,
            )
            observed = {}
            for line in result.stdout.splitlines():
                digest, filename = line.split(maxsplit=1)
                if len(digest) == 64:
                    observed[Path(filename.strip()).name] = digest
            changed = [name for name, digest in expected.items() if observed.get(name) != digest]
            report.update(
                observed_sha256=observed,
                changed_inputs=changed,
                verified=not changed and not result.return_code and not result.error_type,
            )
        except Exception as error:
            report["error"] = str(error)
            LOG.warning("Could not audit Interaction2Code input integrity", exc_info=True)
        (artifacts / "input_assets_audit.json").write_text(json.dumps(report, indent=2) + "\n")

    async def _grade(self, directory: Path, artifacts: Path, metadata: dict) -> dict:
        sandbox = await self._start_sandbox()
        try:
            # Provision browser capture only after coding, in a separate sandbox.
            await sandbox.upload(Path(__file__).with_name("render.py"), "/opt/interaction2code/render.py")
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
        return await self._get_scorer_pool().score(
            dataset_root=self.config.dataset_root,
            task_dir=directory,
            artifacts=artifacts,
            timeout=self.config.scoring_timeout,
        )

    async def run(self, request: Request, body: Interaction2CodeRunRequest) -> Interaction2CodeVerifyResponse:
        """Generate with the selected mode and verify using the same native evaluator."""
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
                if self.config.agent_mode == "single_turn":
                    body = body.model_copy(
                        update={
                            "responses_create_params": build_request(
                                directory, max_output_tokens=self.config.single_turn_max_output_tokens
                            )
                        }
                    )
                    (artifacts / "input.json").write_text(body.model_dump_json())
                    response, result = await self._single_turn(request, body, directory, artifacts, metadata)
                else:
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
                        self_termination = self_termination_reason(
                            run_result, body.responses_create_params.model_dump_json()
                        )
                        status = "agent_timeout" if timed_out else "context_limit" if limited else "agent_error"
                        result = failed_result(
                            status=status,
                            infrastructure=not (timed_out or limited or self_termination),
                            reason=self_termination,
                        )
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
                            await self._collect_feedback(sandbox, artifacts)
                            await self._audit_inputs(sandbox, directory, artifacts)
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
                agent_mode=self.config.agent_mode,
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
