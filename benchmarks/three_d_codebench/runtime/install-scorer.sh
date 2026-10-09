#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
runtime_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
validation_dir="${THREED_VALIDATION_DIR:-/var/log/3dcodebench}"
export THREED_VALIDATION_DIR="$validation_dir"
mkdir -p "$validation_dir"
python="${THREED_SCORER_PYTHON:-$(command -v python3)}"
[[ -x "$python" ]] || python="$(command -v python3)"
command -v uv >/dev/null || "$python" -m pip install uv
mkdir -p /opt/3dcodebench /var/log/3dcodebench
export UV_CACHE_DIR=/tmp/3dcodebench-uv-cache
"$python" - <<'PY' > /tmp/3dcodebench-torch-constraints.txt
from importlib.metadata import version
for name in ('torch', 'torchvision'):
    print(f'{name}=={version(name)}')
PY
uv pip install --python "$python" --constraint /tmp/3dcodebench-torch-constraints.txt \
    transformers==4.57.1 timm==1.0.22 open_clip_torch==3.2.0 trimesh==4.10.1 scipy==1.16.3 easydict==1.13
"$python" - <<'PY'
import torch, torchvision, transformers, timm, open_clip, scipy, trimesh
print('Native scorer imports OK', torch.__version__, transformers.__version__)
PY
uv pip freeze --python "$python" > "$validation_dir/scorer-packages.txt"
printf '#!/bin/sh\nexec "%s" "$@"\n' "$python" > /usr/local/bin/3dcodebench-score-python
chmod +x /usr/local/bin/3dcodebench-score-python
rm -rf "$UV_CACHE_DIR" /root/.cache/uv /root/.cache/pip /root/.cache/pypoetry /root/.cache/ms-playwright /root/.npm
# Keep this check after cache removal: environments must be self-contained.
export PYTHONPATH="$(realpath "$runtime_dir/../../.."):${PYTHONPATH:-}"
for metric in executability image_similarity text_image_similarity shape_chamfer shape_uni3d failure_taxonomy; do
    "$python" -m "benchmarks.three_d_codebench.metrics.$metric" --help >/dev/null
done
