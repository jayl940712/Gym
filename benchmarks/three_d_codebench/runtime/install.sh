#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Runs only inside the disposable image build container.
set -euo pipefail
runtime_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
validation_dir="${THREED_VALIDATION_DIR:-/var/log/3dcodebench}"
export THREED_VALIDATION_DIR="$validation_dir"
mkdir -p "$validation_dir"
export DEBIAN_FRONTEND=noninteractive
mkdir -p /opt/3dcodebench /var/log/3dcodebench
# Debian supplies Blender 5.0 on ARM64. These changes affect this image only.
rm -f /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list
printf '%s\n' 'deb https://deb.debian.org/debian sid main' > /etc/apt/sources.list
apt-get update
version="$(apt-cache policy blender | awk '/Candidate:/ {print $2}')"
[[ "$version" == 5.0.1+dfsg-* ]] || { echo "Unexpected Blender candidate: $version" >&2; exit 1; }
apt-get install -y --no-install-recommends "blender=$version" python3-numpy python3-scipy \
    python3-skimage python3-shapely ca-certificates
# Debian's ARM64 package disables OpenVDB. Rebuild with voxel-remeshing support.
if ! blender --background --factory-startup --python-exit-code 1 \
    --python-expr 'import bpy; assert bpy.app.build_options.openvdb'; then
    bash "$runtime_dir/build-openvdb.sh"
fi
if ! command -v opencode >/dev/null; then
    apt-get install -y --no-install-recommends nodejs npm
    npm install --global opencode-ai@1.17.11
fi
opencode --version | tee "$validation_dir/opencode-version.txt"
blender --version | tee "$validation_dir/blender-version.txt"
blender --background --factory-startup --python-exit-code 1 \
    --python "$runtime_dir/validate_blender.py"
dpkg-query -W > "$validation_dir/packages.tsv"
cp "$validation_dir/blender-version.txt" /opt/3dcodebench/
# The benchmark renderer/exporter are trusted helpers, not reference assets.

apt-get clean
rm -rf /var/lib/apt/lists/*
