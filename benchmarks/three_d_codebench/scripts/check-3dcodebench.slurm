#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=3dcodebench-sandbox-check
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=3dcodebench-sandbox-%j.log
set -euo pipefail
repo="$(realpath "${THREED_GYM_REPO:-$SLURM_SUBMIT_DIR}")"
image="$(realpath "${1:?Pass a Blender/OpenCode sqsh image}")"
output="$(realpath -m "${2:?Pass an output report directory}")"
scratch="$(mktemp -d /tmp/3dcb-check-${SLURM_JOB_ID}-XXXXXX)"
export TMPDIR="$scratch" PYTHONPATH="$repo"
export ENROOT_DATA_PATH="$scratch/data" ENROOT_CACHE_PATH="$scratch/cache" ENROOT_RUNTIME_PATH="$scratch/runtime"
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_RUNTIME_PATH"
trap 'rm -rf "$scratch"' EXIT
"$repo/.venv/bin/python" "$repo/benchmarks/three_d_codebench/runtime/check_sandbox.py" "$image" "$output"
