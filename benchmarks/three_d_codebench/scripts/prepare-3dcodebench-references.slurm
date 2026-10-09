#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=3dcodebench-references
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=3dcodebench-references-%j.log
set -euo pipefail
repo="$(realpath "${THREED_GYM_REPO:-$SLURM_SUBMIT_DIR}")"
data="$(realpath "${1:?Pass the prepared dataset directory}")"
image="${THREED_TASK_IMAGE:?Set THREED_TASK_IMAGE to a Blender/OpenCode sqsh image}"
scratch="$(mktemp -d /tmp/3dcb-ref-${SLURM_JOB_ID}-XXXXXX)"
export ENROOT_DATA_PATH="$scratch/data" ENROOT_CACHE_PATH="$scratch/cache" ENROOT_RUNTIME_PATH="$scratch/runtime"
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_RUNTIME_PATH"
trap 'rm -rf "$scratch"' EXIT
srun --no-container-mount-home --container-image="$image" \
    --container-mounts="$data:/dataset,$repo:/gym" --container-workdir=/tmp \
    python3 /gym/benchmarks/three_d_codebench/runtime/prepare_references.py \
    --prepared-root /dataset --limit "${2:-0}"
