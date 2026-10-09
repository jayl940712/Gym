#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
runtime_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
validation_dir="${THREED_VALIDATION_DIR:-/var/log/3dcodebench}"
export THREED_VALIDATION_DIR="$validation_dir"
mkdir -p "$validation_dir"
export DEBIAN_FRONTEND=noninteractive
work=/tmp/blender-openvdb-build
mkdir -p "$work" /var/log/3dcodebench
apt-mark showmanual | sort > "$work/manual-before.txt"
printf '%s\n' 'deb https://deb.debian.org/debian sid main' 'deb-src https://deb.debian.org/debian sid main' > /etc/apt/sources.list
apt-get update
apt-get install -y --no-install-recommends build-essential dpkg-dev devscripts libopenvdb-dev libblosc-dev
apt-get build-dep -y --no-install-recommends blender
cd "$work"
apt-get source blender=5.0.1+dfsg-6
cd blender-5.0.1+dfsg
sed -i 's/SETVDB = OFF/SETVDB = ON/; s/dh_dwz -- --max-die-limit none/true/' debian/rules
cp debian/rules "$validation_dir/blender-openvdb-rules"
export DEB_BUILD_OPTIONS="parallel=${SLURM_CPUS_PER_TASK:-32} nocheck"
dpkg-buildpackage -b -uc -us
cd "$work"
apt-get install -y --allow-downgrades --reinstall --no-install-recommends ./blender_*.deb ./blender-data_*.deb
blender --background --factory-startup -t 2 --python-exit-code 1 --python "$runtime_dir/validate_blender.py"
# Retain runtime dependencies through the newly installed Blender package.
apt-mark showmanual | sort > "$work/manual-after.txt"
comm -13 "$work/manual-before.txt" "$work/manual-after.txt" | xargs -r apt-mark auto
apt-mark manual blender blender-data
apt-get autoremove -y --purge
apt-get clean
rm -rf "$work" /var/lib/apt/lists/*
dpkg-query -W > "$validation_dir/packages.tsv"
