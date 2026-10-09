#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=3dcodebench-rescore
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
set -euo pipefail
repo="$(realpath "${THREED_GYM_REPO:-$SLURM_SUBMIT_DIR}")"
references="$(realpath "${1:?Pass the trusted reference directory}")"
runtime="$repo/benchmarks/three_d_codebench/runtime"
output="$(realpath "${2:?Pass an existing generation run}")"
task=all
if (( $# > 2 )); then
    shift 2
    while (( $# )); do
        case "$1" in
            --task) task="${2:?Pass all, text_to_3d, or image_to_3d}"; shift 2 ;;
            *) echo "Unknown argument: $1" >&2; exit 1 ;;
        esac
    done
fi
[[ "$task" == all || "$task" == text_to_3d || "$task" == image_to_3d ]] || { echo "Invalid --task: $task" >&2; exit 1; }

[[ -s "$output/rollouts_generation.jsonl" ]]
exec 9> "$output/.run.lock"
flock -n 9 || { echo 'Another job is using this output directory.' >&2; exit 1; }
scoring_scratch="$(mktemp -d "/tmp/3dcb-score-${SLURM_JOB_ID}-XXXXXX")"
export ENROOT_DATA_PATH="$scoring_scratch/data"
export ENROOT_CACHE_PATH="$scoring_scratch/cache"
export ENROOT_RUNTIME_PATH="$scoring_scratch/runtime"
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_RUNTIME_PATH"
trap 'rm -rf "$scoring_scratch"' EXIT
rm -f "$output/complete"
ENROOT_MAX_PROCESSORS=16 srun --no-container-mount-home --container-image="${THREED_SCORER_IMAGE:?Set THREED_SCORER_IMAGE}" \
    --container-mounts="${THREED_CONTAINER_MOUNTS:-$repo:$repo,$references:$references,$output:$output${HF_HOME:+,$HF_HOME:$HF_HOME}}" --container-workdir="$repo" \
    /usr/bin/env \
    /usr/local/bin/3dcodebench-score-python -m benchmarks.three_d_codebench.score \
    --run "$output" --references "$references" --task "$task" > "$output/scoring.log" 2>&1
cd "$repo"
.venv/bin/gym eval profile --inputs "$output/scoring_inputs.jsonl" \
    --rollouts "$output/rollouts.jsonl" > "$output/profile.log" 2>&1
python - "$output" "$task" <<'PY_COMPLETE'
import json
import sys
from pathlib import Path
output = Path(sys.argv[1])
tasks = {json.loads(line)["task"] for line in (output / "rollouts_generation_materialized_inputs.jsonl").read_text().splitlines() if line.strip()}
marker = "complete" if sys.argv[2] == "all" or tasks == {sys.argv[2]} else f"complete_{sys.argv[2]}"
(output / marker).touch()
PY_COMPLETE
