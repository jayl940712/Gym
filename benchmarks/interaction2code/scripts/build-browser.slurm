#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=i2c-browser-image
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
repo="$(realpath "${1:?Pass Gym checkout}")"
image="$(realpath -m "${2:?Pass new sqsh path}")"
[[ ! -e "$image" ]] || { echo "Image already exists: $image" >&2; exit 1; }
staged="$image.partial-${SLURM_JOB_ID}"
scratch="$(mktemp -d /tmp/i2c-build-${SLURM_JOB_ID}-XXXXXX)"
export ENROOT_DATA_PATH="$scratch/data" ENROOT_CACHE_PATH="$scratch/cache" ENROOT_RUNTIME_PATH="$scratch/runtime"
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_RUNTIME_PATH"
trap 'rm -rf "$scratch"; rm -f "$staged"' EXIT
enroot import -o "$scratch/base.sqsh" docker://python:3.12-slim-bookworm
srun --no-container-mount-home --container-image="$scratch/base.sqsh" --container-save="$staged" \
 --container-remap-root --container-writable --container-mounts="$repo:/gym" --container-workdir=/tmp \
 bash /gym/benchmarks/interaction2code/runtime/install-browser.sh
[[ -s "$staged" ]]
mv "$staged" "$image"
sha256sum "$image" > "$image.sha256"
