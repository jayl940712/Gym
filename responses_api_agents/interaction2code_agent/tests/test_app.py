# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from fastapi import Request
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.interaction2code_agent.app import (
    GYM_ROOT,
    Interaction2CodeAgent,
    Interaction2CodeConfig,
    Interaction2CodeRunRequest,
    task_directory,
)


def response():
    return NeMoGymResponse(
        id="test",
        created_at=0,
        model="test",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


@pytest.fixture
def body():
    return Interaction2CodeRunRequest(
        page_id=106,
        interaction_id=1,
        responses_create_params={"input": [{"role": "user", "content": "Create index.html"}]},
    )


@pytest.fixture
def agent(tmp_path, body, monkeypatch):
    monkeypatch.setattr("responses_api_agents.interaction2code_agent.app.ensure_runtime", lambda _: None)
    config = Interaction2CodeConfig.model_construct(dataset_root=tmp_path / "data", results_dir=tmp_path / "results")
    instance = Interaction2CodeAgent.model_construct(
        config=config, server_client=SimpleNamespace(global_config_dict={})
    )
    instance._sem = asyncio.Semaphore(1)
    instance._sandbox_id_to_sandbox = {}
    instance._sandbox_id_to_run_result = {}
    directory = task_directory(config.dataset_root, body)
    directory.mkdir(parents=True)
    (directory / "task.json").write_text(
        json.dumps({"page_id": 106, "interaction_id": 1, "width": 1280, "height": 720})
    )
    for name in ("before.png", "after.png", "placeholder.jpg"):
        (directory / name).write_bytes(b"asset")
    return instance


@pytest.mark.parametrize("page,interaction", [(0, 1), (128, 1), (1, 0), ("../106", 1)])
def test_invalid_task_ids(page, interaction):
    with pytest.raises(ValidationError):
        Interaction2CodeRunRequest(page_id=page, interaction_id=interaction, responses_create_params={"input": []})


def test_paths_are_standalone_and_independent_of_working_directory(tmp_path, monkeypatch):
    values = yaml.safe_load((GYM_ROOT / "benchmarks/interaction2code/config.yaml").read_text())[
        "interaction2code_agent"
    ]["responses_api_agents"]["interaction2code_agent"]
    monkeypatch.chdir(tmp_path)
    config = Interaction2CodeConfig.model_validate(
        values | {"name": "interaction2code_agent", "host": "127.0.0.1", "port": 8010}
    )
    assert config.dataset_root == GYM_ROOT / "benchmarks/interaction2code/data"
    assert config.runtime_root.is_relative_to(GYM_ROOT)
    assert Path(config.artifacts_dir).is_relative_to(GYM_ROOT)


def test_task_symlink_cannot_escape(tmp_path, body):
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks/106-1").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        task_directory(tmp_path, body)


def test_nvidia_config_uses_one_reasoning_alias_and_one_provider(monkeypatch):
    monkeypatch.setenv("NVINFERENCE_API_KEY", "test-key")
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create({"config_paths": ["benchmarks/interaction2code/nvidia.yaml"]}),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    agent_config = config.interaction2code_agent.responses_api_agents.interaction2code_agent
    assert set(resolve_provider_config(agent_config.sandbox_provider, config)) == {"docker"}
    assert config.policy_model.responses_api_models.vllm_model.reasoning_field == "reasoning_content"
    assert config.policy_model_name == "nvidia/nvidia/nemotron-3.5-super-vl"
    client = ServerClient(head_server_config=dict(config.head_server), global_config_dict=config)
    assert client.assistant_message_header("policy_model") == b"x-opencode-assistant-message-id"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,status,masked",
    [
        ("ok", "ok", False),
        ("missing", "missing_html", False),
        ("timeout", "agent_timeout", False),
        ("context", "context_limit", False),
        ("agent", "agent_error", True),
        ("scorer", "infrastructure_error", True),
        ("asset", "infrastructure_error", True),
    ],
)
async def test_run_isolation_failures_artifacts_and_cleanup(agent, body, monkeypatch, case, status, masked):
    sandbox = SimpleNamespace(
        stop=AsyncMock(),
        upload=AsyncMock(),
        download=AsyncMock(),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=1 if case == "missing" else 0, error_type=None)),
    )
    monkeypatch.setattr(Interaction2CodeAgent, "_start_sandbox", AsyncMock(return_value=sandbox))
    generated = response()
    if case == "context":
        generated.status = "incomplete"
    monkeypatch.setattr(Interaction2CodeAgent, "responses", AsyncMock(return_value=generated))
    grade = AsyncMock(
        return_value={
            "reward": 0.63,
            "status": "ok",
            "interaction_flag": True,
            "full_page": {"clip_similarity": 0.81},
            "interaction": {"clip_similarity": 0.63},
        }
    )
    if case == "scorer":
        grade.side_effect = RuntimeError("metric weights unavailable")
    monkeypatch.setattr(Interaction2CodeAgent, "_grade", grade)
    if case in {"timeout", "context", "agent"}:
        agent._sandbox_id_to_run_result["session"] = {"opencode_failed": True}
        if case == "timeout":
            agent._sandbox_id_to_run_result["session"]["opencode_error_type"] = "timeout"
    if case == "asset":
        path = task_directory(agent.config.dataset_root, body) / "before.png"
        path.unlink()
        path.symlink_to(path.parent.parent.parent / "private.png")
    request = Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "session"}})
    verified = await agent.run(request, body)
    assert verified.status == status
    assert verified.mask_sample is masked
    assert verified.reward == (0.63 if case == "ok" else 0)
    assert not agent._sandbox_id_to_sandbox
    assert not agent._sandbox_id_to_run_result
    assert not hasattr(request.state, "_ng_observation_invocation_id")
    sandbox.stop.assert_awaited_once()
    if case == "asset":
        sandbox.upload.assert_not_awaited()
    else:
        assert [call.args[1] for call in sandbox.upload.await_args_list] == [
            "/workspace/before.png",
            "/workspace/after.png",
            "/workspace/placeholder.jpg",
        ]
    if case == "ok":
        assert verified.full_page["clip_similarity"] == 0.81
        assert verified.interaction["clip_similarity"] == 0.63
        assert grade.await_count == 1
    saved = json.loads((Path(verified.artifact_dir) / "result.json").read_text())
    assert saved["protocol"] == "interaction2code-opencode"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["ok", "timeout", "provider", "browser", "unsafe", "alert"])
async def test_render_sandbox_never_receives_reference_images(agent, body, monkeypatch, tmp_path, case):
    directory = task_directory(agent.config.dataset_root, body)
    (tmp_path / "index.html").write_text("<html></html>")
    rendered = {
        "status": "task_error" if case == "alert" else "browser_error" if case == "browser" else "ok",
        "screenshots": ["../private.png"] if case == "unsafe" else ["0_source.png", "1_10_20_click.png"],
    }
    (tmp_path / "render.json").write_text(json.dumps(rendered))
    sandbox = SimpleNamespace(
        upload=AsyncMock(),
        download=AsyncMock(),
        stop=AsyncMock(),
        exec=AsyncMock(
            return_value=SimpleNamespace(
                return_code=124 if case == "timeout" else 0,
                error_type="provider" if case == "provider" else None,
                stdout="",
                stderr="",
            )
        ),
    )
    monkeypatch.setattr(Interaction2CodeAgent, "_start_sandbox", AsyncMock(return_value=sandbox))
    score = AsyncMock(return_value={"reward": 0.75, "status": "ok"})
    monkeypatch.setattr("responses_api_agents.interaction2code_agent.app.score", score)
    if case in {"provider", "browser", "unsafe"}:
        with pytest.raises((RuntimeError, ValueError)):
            await agent._grade(directory, tmp_path, {"width": 1280, "height": 720})
    else:
        result = await agent._grade(directory, tmp_path, {"width": 1280, "height": 720})
        assert result["reward"] == (0.75 if case == "ok" else 0)
        assert score.await_count == (1 if case == "ok" else 0)
        if case == "alert":
            assert result["status"] == "browser_error"
            assert result["mask_sample"] is False
    assert [call.args[1] for call in sandbox.upload.await_args_list] == [
        "/opt/interaction2code/render.py",
        "/workspace/index.html",
        "/workspace/placeholder.jpg",
    ]
    sandbox.stop.assert_awaited_once()
