#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run OpenCode generation, fresh Blender verification, and native offline scoring.
#SBATCH --job-name=three_d_codebench-opencode
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=04:00:00
set -euo pipefail

repo="$(realpath "${1:?Pass your Gym checkout}")"
output="$(realpath -m "${2:?Pass an output directory (reuse it to resume)}")"
input="$(realpath "${3:?Pass a prepared task JSONL}")"
limit="${4:-0}"
repeats="${5:-1}"
task=all
if (( $# > 5 )); then
    shift 5
    while (( $# )); do
        case "$1" in
            --task) task="${2:?Pass all, text_to_3d, or image_to_3d}"; shift 2 ;;
            *) echo "Unknown argument: $1" >&2; exit 1 ;;
        esac
    done
fi
[[ "$task" == all || "$task" == text_to_3d || "$task" == image_to_3d ]] || { echo "Invalid --task: $task" >&2; exit 1; }
[[ "$limit" =~ ^[0-9]+$ && "$repeats" =~ ^[1-9][0-9]*$ ]] || {
    echo 'Limit must be nonnegative; repeats must be positive.' >&2
    exit 1
}
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
mkdir -p "$output"
exec 9> "$output/.run.lock"
flock -n 9 || { echo 'Another job is using this output directory.' >&2; exit 1; }
runtime="$repo/benchmarks/three_d_codebench/runtime"
data="$(realpath "${THREED_DATASET:-$repo/data/three_d_codebench}")"
references="$(realpath "${THREED_REFERENCES:-$data/reference}")"
task_image="${THREED_TASK_IMAGE:?Set THREED_TASK_IMAGE to your Blender/OpenCode sqsh image}"
scorer_image="${THREED_SCORER_IMAGE:?Set THREED_SCORER_IMAGE to your scorer sqsh image}"
serving_image="${THREED_SERVING_IMAGE:?Set THREED_SERVING_IMAGE to your vLLM image}"
checkpoint="${THREED_CHECKPOINT:?Set THREED_CHECKPOINT to your policy checkpoint}"
serving_python="${THREED_SERVING_PYTHON:-/usr/local/bin/python}"
mounts="$repo:$repo,$data:$data,$references:$references,$checkpoint:$checkpoint,$output:$output"
[[ -z "${HF_HOME:-}" ]] || mounts+=",$HF_HOME:$HF_HOME"
container_mounts="${THREED_CONTAINER_MOUNTS:-$mounts}"
for path in "$input" "$data/manifest.json" "$task_image" "$scorer_image" "$serving_image" "$checkpoint/config.json"; do
    [[ -r "$path" ]] || { echo "Missing or unreadable asset: $path" >&2; exit 1; }
done
[[ -x "$repo/.venv/bin/gym" ]] || { echo 'Missing Gym environment.' >&2; exit 1; }
command -v enroot >/dev/null
command -v uv >/dev/null
scratch="$(mktemp -d "/tmp/3dcb-run-${SLURM_JOB_ID}-XXXXXX")"
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
export RAY_OVERRIDE_RESOURCES='{"CPU":4}'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ENROOT_MAX_PROCESSORS=2 PYTHONUNBUFFERED=1
cd "$repo"
cp "$task_image" "$scratch/three_d_codebench-opencode.sqsh"
printf '%s\n' 'unset ENROOT_MOUNT_HOME' > "$scratch/enroot-start.sh"
head_port="$(python - <<'PY_PORT'
import socket
with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    print(listener.getsockname()[1])
PY_PORT
)"
python - "$repo" "$output" "$scratch" "$data" "$input" "$limit" "$head_port" "$repeats" "$checkpoint" "$task" <<'PY_CONFIG'
import json
import socket
import sys
from pathlib import Path
from benchmarks.three_d_codebench.evaluation import PROTOCOL, select_rows

repo, output, scratch, data, source = map(Path, sys.argv[1:6])
limit, head_port = map(int, sys.argv[6:8])
repeats = int(sys.argv[8])
checkpoint = str(Path(sys.argv[9]).resolve())
task = sys.argv[10]
rows = []
with source.open() as stream:
    for line in stream:
        if not line.strip():
            continue
        row = json.loads(line)
        directory = (data / row["task"] / row["record_id"]).resolve()
        if not directory.is_relative_to(data.resolve()) or not (directory / "task.json").is_file():
            raise ValueError(f"Missing or unsafe prepared assets for {row['task']}/{row['record_id']}")
        rows.append(row)
rows = select_rows(rows, task, limit)
selected = "".join(json.dumps(row) + "\n" for row in rows)
input_path = output / "input.jsonl"
if input_path.exists() and input_path.read_text() != selected:
    raise ValueError("Resume requires the same selected inputs and ordering; use a new output directory")
settings_path = output / "run_settings.json"
settings = {"num_repeats": repeats, "checkpoint": checkpoint, "dataset_root": str(data.resolve()),
            "task_selection": task, "scoring_protocol": PROTOCOL}
if settings_path.exists():
    if json.loads(settings_path.read_text()) != settings:
        raise ValueError("Resume requires the same repeat count, checkpoint and dataset; use a new output directory")
elif (output / "rollouts_generation.jsonl").exists():
    raise ValueError("This older run has no resume settings; rescore it with score-3dcodebench.slurm")
settings_path.write_text(json.dumps(settings, indent=2) + "\n")
input_path.write_text(selected)
for component in ("responses_api_agents/three_d_codebench_agent", "responses_api_models/vllm_model", "environment_servers/legacy_agent"):
    folder = scratch / "servers" / component
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ".venv").symlink_to(repo / ".venv")
config = {
    "config_paths": ["benchmarks/three_d_codebench/config.yaml", "responses_api_models/vllm_model/configs/vllm_model.yaml"],
    "policy_base_url": f"http://{socket.getfqdn()}:8000/v1",
    "policy_api_key": "EMPTY",
    "policy_model_name": "three_d_codebench-policy",
    "skip_venv_if_present": True,
    "uv_venv_dir": str(scratch / "servers"),
    "uv_cache_dir": str(scratch / "uv-cache"),
    "head_server": {"host": "127.0.0.1", "port": head_port},
    "model_endpoint_readiness_timeout_seconds": 1800,
    "three_d_codebench_agent": {"responses_api_agents": {"three_d_codebench_agent": {
        "dataset_root": str(data),
        "results_dir": str(output / "artifacts"),
        "artifacts_dir": str(output / "opencode"),
        "opencode_max_context_window": 131072,
        "concurrency": 64,
        "sandbox_config": {
            "image": str(scratch / "three_d_codebench-opencode.sqsh"),
            "env": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        },
    }}},
    "sandbox": {
        "_delete_key": "docker",
        "default_metadata": {"sandbox-api": "enroot"},
        "enroot": {"create": {
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
(output / "config.yaml").write_text(json.dumps(config, indent=2) + "\n")
print(f"Prepared {len(rows)} tasks; output: {output}")
PY_CONFIG

[[ ! -f "$output/complete" ]] || { echo 'This run is already complete.'; exit 0; }

parser_flags=(--enable-auto-tool-choice --tool-call-parser "${THREED_TOOL_CALL_PARSER:?Set the tool parser for your served policy}")
[[ -z "${THREED_REASONING_PARSER:-}" ]] || parser_flags+=(--reasoning-parser "$THREED_REASONING_PARSER")
hf_flags=()
[[ -z "${THREED_HF_OVERRIDES:-}" ]] || hf_flags=(--hf-overrides "$THREED_HF_OVERRIDES")
model_pid=""
gym_pid=""
cleanup() {
    [[ -z "$gym_pid" ]] || kill "$gym_pid" 2>/dev/null || true
    [[ -z "$model_pid" ]] || kill "$model_pid" 2>/dev/null || true
}
trap cleanup EXIT
if [[ ! -f "$output/generation_complete" ]]; then
# Exercise the real fresh-sandbox verifier before loading the policy model.
python "$runtime/check_sandbox.py" "$task_image" "$output/sandbox-preflight" \
    > "$output/sandbox-preflight.log" 2>&1
TMPDIR=/tmp ENROOT_MAX_PROCESSORS=16 srun --overlap --no-container-mount-home --container-image="$serving_image" --container-mounts="$container_mounts" \
    --container-workdir=/tmp /usr/bin/env VLLM_USE_FASTOKENS=0 \
    "$serving_python" -m vllm.entrypoints.cli.main serve \
    "$checkpoint" --host 0.0.0.0 --port 8000 --served-model-name three_d_codebench-policy \
    --tensor-parallel-size 4 --trust-remote-code --max-model-len 131072 \
    --max-num-seqs 64 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 \
    "${parser_flags[@]}" \
    --limit-mm-per-prompt '{"image":8,"video":0,"audio":0}' \
    "${hf_flags[@]}" \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}' \
    > "$output/model.log" 2>&1 &
model_pid=$!
ready=false
for ((attempt=0; attempt<360; attempt++)); do
    if curl -sf http://127.0.0.1:8000/v1/models > "$output/models.json"; then
        ready=true
        break
    fi
    kill -0 "$model_pid" || { echo 'Model server exited; inspect model.log.' >&2; exit 1; }
    sleep 10
done
[[ "$ready" == true ]] || { echo 'Model readiness timed out.' >&2; exit 1; }
python - "$output/models.json" <<'PY_MODELS'
import json
import sys
models = json.load(open(sys.argv[1]))["data"]
assert any(m["id"] == "three_d_codebench-policy" and m["max_model_len"] == 131072 for m in models), models
PY_MODELS
gym env start --config "$output/config.yaml" > "$output/gym.log" 2>&1 &
gym_pid=$!
HEAD_MAX_WAIT=600 bash scripts/wait_for_servers.sh "$gym_pid" "$head_port" 1800
gym eval run --no-serve --resume --config "$output/config.yaml" --agent three_d_codebench_agent \
    --input "$output/input.jsonl" --output "$output/rollouts_generation.jsonl" \
    --num-repeats "$repeats" --concurrency 64 --health-check-workers 4 \
    ++dispatch_budget_s=10800 ++drain_margin_s=1800 ++require_complete=true \
    > "$output/eval.log" 2>&1
touch "$output/generation_complete"
fi
# Release model memory before loading the native learned evaluators.
kill "$gym_pid" "$model_pid" 2>/dev/null || true
wait "$gym_pid" 2>/dev/null || true
wait "$model_pid" 2>/dev/null || true
gym_pid=""
model_pid=""
[[ "${THREED_GENERATION_ONLY:-0}" != 1 ]] || { echo 'All selected rollouts collected; scoring is a separate job.'; exit 0; }
TMPDIR=/tmp ENROOT_MAX_PROCESSORS=16 srun --overlap --no-container-mount-home --container-image="$scorer_image" \
    --container-mounts="$container_mounts" --container-workdir="$repo" \
    /usr/bin/env \
    /usr/local/bin/3dcodebench-score-python -m benchmarks.three_d_codebench.score \
    --run "$output" --references "$references" --task "$task" \
    > "$output/scoring.log" 2>&1
gym eval profile --inputs "$output/scoring_inputs.jsonl" \
    --rollouts "$output/rollouts.jsonl" > "$output/profile.log" 2>&1
touch "$output/complete"
