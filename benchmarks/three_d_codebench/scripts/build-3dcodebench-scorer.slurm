#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=3dcodebench-scorer-image
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=3dcodebench-scorer-build-%j.log
set -euo pipefail
repo="$(realpath "${THREED_GYM_REPO:-$SLURM_SUBMIT_DIR}")"
image="$(realpath -m "${1:?Pass the output sqsh image path}")"
[[ ! -e "$image" ]] || { echo "Image already exists: $image" >&2; exit 1; }
mkdir -p "$(dirname "$image")" "$image.validation"
base="${THREED_SCORER_BASE:?Set THREED_SCORER_BASE to a compatible CUDA/PyTorch image}"
scratch="$(mktemp -d /tmp/3dcb-build-${SLURM_JOB_ID}-XXXXXX)"
export ENROOT_DATA_PATH="$scratch/data" ENROOT_CACHE_PATH="$scratch/cache" ENROOT_RUNTIME_PATH="$scratch/runtime"
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_RUNTIME_PATH"
trap 'rm -rf "$scratch"' EXIT
srun --no-container-mount-home --container-image="$base" --container-save="$image" \
    --container-remap-root --container-writable \
    --container-mounts="$image.validation:/reports,$repo:/gym" --container-workdir=/tmp \
    /usr/bin/env THREED_VALIDATION_DIR=/reports bash /gym/benchmarks/three_d_codebench/runtime/install-scorer.sh
[[ -s "$image" ]]
sha256sum "$image" > "$image.sha256"
