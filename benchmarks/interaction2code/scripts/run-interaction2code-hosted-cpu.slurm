#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run a hosted Kimi policy with isolated browser verification and native CPU scoring.
#SBATCH --job-name=i2c-hosted-cpu
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
set -euo pipefail

repo="$(realpath "${1:?Pass your Gym checkout}")"
output="$(realpath -m "${2:?Pass an output directory (reuse it to resume)}")"
input="$(realpath "${3:?Pass a prepared task JSONL}")"
shift 3
limit=10
repeats=1
if (( $# )) && [[ "$1" != --* ]]; then limit="$1"; shift; fi
if (( $# )) && [[ "$1" != --* ]]; then repeats="$1"; shift; fi
agent_mode="${I2C_AGENT_MODE:-opencode}"
while (( $# )); do
    case "$1" in
        --agent-mode) agent_mode="${2:?Pass opencode or single_turn}"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ "$agent_mode" == opencode || "$agent_mode" == single_turn ]] || {
    echo 'Agent mode must be opencode or single_turn' >&2; exit 1
}
export I2C_AGENT_MODE="$agent_mode"
export I2C_SINGLE_TURN_MAX_OUTPUT_TOKENS="${I2C_SINGLE_TURN_MAX_OUTPUT_TOKENS:-32768}"
[[ "$limit" =~ ^[0-9]+$ && "$repeats" =~ ^[1-9][0-9]*$ ]] || {
    echo 'Limit must be nonnegative; repeats must be positive.' >&2
    exit 1
}
[[ "$repeats" == 1 ]] || { echo "The coverage report supports one repeat per interaction." >&2; exit 1; }
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
mkdir -p "$output"
exec 9> "$output/.run.lock"
flock -n 9 || { echo 'Another job is using this output directory.' >&2; exit 1; }
data="$(realpath "${I2C_DATASET:-$repo/benchmarks/interaction2code/data}")"
scorer_root="$(realpath "${I2C_SCORER_ROOT:-$repo/benchmarks/interaction2code/.cache/scorer}")"
task_image="${I2C_TASK_IMAGE:?Set I2C_TASK_IMAGE to your browser/OpenCode sqsh image}"
if [[ -z "${WEBARENA_JUDGE_API_KEY:-}" ]]; then
    set +u
    source "$HOME/.bashrc" > /dev/null 2>&1
    set -u
fi
: "${WEBARENA_JUDGE_API_KEY:?Export the hosted endpoint API key}"
export WEBARENA_JUDGE_API_KEY
# Keep an explicit CLI selection even when .bashrc also sets a default mode.
export I2C_AGENT_MODE="$agent_mode"
export I2C_POLICY_BASE_URL="${I2C_POLICY_BASE_URL:-https://inference-api.nvidia.com/v1}"
export I2C_POLICY_MODEL="${I2C_POLICY_MODEL:-nvidia/moonshotai/kimi-k3}"
export I2C_OPENCODE_STEPS="${I2C_OPENCODE_STEPS:-50}"
[[ "$I2C_OPENCODE_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid OpenCode step limit' >&2; exit 1; }
export I2C_CAPTURE_MODEL_CALLS="${I2C_CAPTURE_MODEL_CALLS:-1}"
[[ "$I2C_CAPTURE_MODEL_CALLS" == 0 || "$I2C_CAPTURE_MODEL_CALLS" == 1 ]] || { echo 'Invalid capture flag' >&2; exit 1; }
for path in "$input" "$data/manifest.json" "$task_image"; do
    [[ -r "$path" ]] || { echo "Missing or unreadable asset: $path" >&2; exit 1; }
done
[[ -x "$repo/.venv/bin/gym" ]] || { echo 'Missing Gym environment.' >&2; exit 1; }
command -v enroot >/dev/null
command -v uv >/dev/null
scratch="$(mktemp -d "/tmp/i2c-run-${SLURM_JOB_ID}-XXXXXX")"
mkdir -p "$output" "$scratch/tmp"
export PATH="$repo/.venv/bin:$PATH"
export VIRTUAL_ENV="$repo/.venv"
export PYTHONPATH="$repo"
export TMPDIR="$scratch/tmp"
# Ignore inherited login-shell paths; keep rootfs, cache, and runtime distinct.
export ENROOT_DATA_PATH="$scratch/sandboxes/data"
export ENROOT_CACHE_PATH="$scratch/sandboxes/cache"
export ENROOT_RUNTIME_PATH="$scratch/sandboxes/runtime"
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_RUNTIME_PATH"
# Ray appends /ray/session_<timestamp>_<pid>/sockets/<name> to this path.
# Keep the prefix short enough for Linux's 107-byte Unix socket path limit.
export RAY_TMPDIR="$(mktemp -d "/tmp/br-${SLURM_JOB_ID}-XXXXXX")"
export UV_CACHE_DIR="$scratch/uv-cache"
export I2C_SCORER_ROOT="$scorer_root"
export RAY_OVERRIDE_RESOURCES='{"CPU":4}'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ENROOT_MAX_PROCESSORS=2 PYTHONUNBUFFERED=1
cd "$repo"
cp "$task_image" "$scratch/interaction2code-opencode.sqsh"
printf '%s\n' 'unset ENROOT_MOUNT_HOME' > "$scratch/enroot-start.sh"
head_port="$(python - <<'PY_PORT'
import socket
with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    print(listener.getsockname()[1])
PY_PORT
)"
python - "$repo" "$output" "$scratch" "$data" "$input" "$limit" "$head_port" "$repeats" "$I2C_POLICY_MODEL" <<'PY_CONFIG'
import json
import os
import socket
import sys
from pathlib import Path
from benchmarks.interaction2code.prepare import PROTOCOL
from responses_api_agents.interaction2code_agent.single_turn import SYSTEM_PROMPT, USER_INSTRUCTIONS, build_request

repo, output, scratch, data, source = map(Path, sys.argv[1:6])
limit, head_port = map(int, sys.argv[6:8])
repeats = int(sys.argv[8])
policy_model = sys.argv[9]
rows = []
with source.open() as stream:
    for line in stream:
        if not line.strip():
            continue
        row = json.loads(line)
        directory = (data / "tasks" / row["task_id"]).resolve()
        if not directory.is_relative_to(data.resolve()) or not (directory / "task.json").is_file():
            raise ValueError(f"Missing or unsafe prepared assets for {row['task_id']}")
        rows.append(row)
rows = rows[:limit] if limit else rows
if not rows:
    raise ValueError("No tasks selected")
if os.environ["I2C_AGENT_MODE"] == "single_turn":
    for row in rows:
        row["responses_create_params"] = build_request(
            data / "tasks" / row["task_id"],
            max_output_tokens=int(os.environ["I2C_SINGLE_TURN_MAX_OUTPUT_TOKENS"]),
        ).model_dump(exclude_none=True)
selected = "".join(json.dumps(row) + "\n" for row in rows)
input_path = output / "input.jsonl"
if input_path.exists() and input_path.read_text() != selected:
    raise ValueError("Resume requires the same selected inputs and ordering; use a new output directory")
settings_path = output / "run_settings.json"
settings = {"num_repeats": repeats, "model": policy_model, "endpoint": os.environ["I2C_POLICY_BASE_URL"], "capture_model_calls": os.environ["I2C_CAPTURE_MODEL_CALLS"] == "1", "dataset_root": str(data.resolve()),
            "scoring_protocol": PROTOCOL}
opencode_steps = int(os.environ["I2C_OPENCODE_STEPS"])
if opencode_steps != 50:
    settings["opencode_steps"] = opencode_steps
if os.environ["I2C_AGENT_MODE"] != "opencode":
    import hashlib
    settings["agent_mode"] = os.environ["I2C_AGENT_MODE"]
    settings["single_turn_max_output_tokens"] = int(os.environ["I2C_SINGLE_TURN_MAX_OUTPUT_TOKENS"])
    settings["single_turn_prompt_sha256"] = hashlib.sha256((SYSTEM_PROMPT + USER_INSTRUCTIONS).encode()).hexdigest()
    (output / "single_turn_prompt.json").write_text(json.dumps({"system": SYSTEM_PROMPT, "user": USER_INSTRUCTIONS}, indent=2) + "\n")
if settings_path.exists():
    if json.loads(settings_path.read_text()) != settings:
        raise ValueError("Resume requires the same repeat count, model, capture settings and dataset; use a new output directory")
elif (output / "rollouts_generation.jsonl").exists():
    raise ValueError("This older run has no resume settings; use a fresh output directory")
settings_path.write_text(json.dumps(settings, indent=2) + "\n")
input_path.write_text(selected)
for component in ("responses_api_agents/interaction2code_agent", "responses_api_models/openai_model", "environment_servers/legacy_agent"):
    folder = scratch / "servers" / component
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ".venv").symlink_to(repo / ".venv")
config = {
    "config_paths": ["benchmarks/interaction2code/config.yaml", "responses_api_models/openai_model/configs/openai_model.yaml"],
    "policy_base_url": "${oc.env:I2C_POLICY_BASE_URL}",
    "policy_api_key": "${oc.env:WEBARENA_JUDGE_API_KEY}",
    "policy_model_name": policy_model,
    "policy_model": {"responses_api_models": {"openai_model": {
        "upstream_request_timeout_seconds": 900,
        "max_concurrent_requests": 64,
        "drop_input_reasoning_items": False,
        "reject_empty_chat_completions": True,
        "upstream_max_num_tries": 1,
        "upstream_retry_policy": {"max_attempts": 3, "backoff_initial_seconds": 2},
    }}},
    "skip_venv_if_present": True,
    "uv_venv_dir": str(scratch / "servers"),
    "uv_cache_dir": str(scratch / "uv-cache"),
    "head_server": {"host": "127.0.0.1", "port": head_port},
    "model_endpoint_readiness_timeout_seconds": 1800,
    "interaction2code_agent": {"responses_api_agents": {"interaction2code_agent": {
        "agent_mode": os.environ["I2C_AGENT_MODE"],
        "single_turn_max_output_tokens": int(os.environ["I2C_SINGLE_TURN_MAX_OUTPUT_TOKENS"]),
        "dataset_root": str(data),
        "results_dir": str(output / "artifacts"),
        "artifacts_dir": str(output / "opencode"),
        "runtime_root": str(Path(os.environ["I2C_SCORER_ROOT"]).resolve()),
        "scoring_concurrency": int(os.environ.get("I2C_SCORING_CONCURRENCY", "2")),
        "scoring_threads": int(os.environ.get("I2C_SCORING_THREADS", "2")),
        "scoring_timeout": 3600,
        "opencode_max_context_window": 131072,
        "output_token_policy": "fixed",
        "opencode_config": {
            "compaction": {"auto": False, "prune": False},
            "agent": {"build": {"steps": opencode_steps}},
            "provider": {"nemo_gym": {
                "npm": "@ai-sdk/openai-compatible",
                "models": {"dummy_model": {"limit": {"output": 32768}, "interleaved": {"field": "reasoning_content"}}},
            }},
        },
        "concurrency": 64,
        "sandbox_config": {
            "image": str(scratch / "interaction2code-opencode.sqsh"),
            "env": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PLAYWRIGHT_BROWSERS_PATH": "/opt/playwright"},
        },
    }}},
    "sandbox": {
        "_delete_key": "docker",
        "default_metadata": {"sandbox-api": "enroot"},
        "enroot": {"exec": {"concurrency": 128}, "create": {
            "base_dir": str(scratch / "sandboxes"),
            "data_path": str(scratch / "sandboxes/data"),
            "cache_path": str(scratch / "sandboxes/cache"),
            "runtime_path": str(scratch / "sandboxes/runtime"),
            "bypass_entrypoint": False,
            "rw": True,
            "extra_start_args": ["--conf", str(scratch / "enroot-start.sh")],
        }},
    },
}
if os.environ["I2C_CAPTURE_MODEL_CALLS"] == "1":
    config["observability_enabled"] = True
    config["model_call_capture_dir"] = str(output / "model_calls")
(output / "config.yaml").write_text(json.dumps(config, indent=2) + "\n")
print(f"Prepared {len(rows)} tasks; output: {output}")
PY_CONFIG

[[ ! -f "$output/complete" ]] || { echo 'This run is already complete.'; exit 0; }

gym_pid=""
cleanup() {
    [[ -z "$gym_pid" ]] || kill "$gym_pid" 2>/dev/null || true
}
trap cleanup EXIT
if [[ ! -f "$output/generation_complete" ]]; then
python "$repo/benchmarks/interaction2code/runtime/check_sandbox.py" "$task_image" "$output/sandbox-preflight" \
    > "$output/sandbox-preflight.log" 2>&1
gym env start --config "$output/config.yaml" > "$output/gym.log" 2>&1 &
gym_pid=$!
HEAD_MAX_WAIT=600 bash scripts/wait_for_servers.sh "$gym_pid" "$head_port" 1800
gym eval run --no-serve --resume --config "$output/config.yaml" --agent interaction2code_agent \
    --input "$output/input.jsonl" --output "$output/rollouts_generation.jsonl" \
    --num-repeats "$repeats" --concurrency 64 --health-check-workers 4 \
    ++dispatch_budget_s=10800 ++drain_margin_s=1800 ++require_complete=true \
    > "$output/eval.log" 2>&1
touch "$output/generation_complete"
fi
python -m benchmarks.interaction2code.report --inputs "$output/input.jsonl" \
    --rollouts "$output/rollouts_generation.jsonl" --output "$output/summary.json" --require-complete \
    > "$output/report.log" 2>&1
gym eval profile --inputs "$output/rollouts_generation_materialized_inputs.jsonl" \
    --rollouts "$output/rollouts_generation.jsonl" > "$output/profile.log" 2>&1
touch "$output/complete"
