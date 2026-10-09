# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download native evaluation weights into the standard Hugging Face cache."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face cache root; defaults to HF_HOME")
args = parser.parse_args()
cache = args.cache_dir.resolve() / "hub" if args.cache_dir else None
if args.cache_dir:
    os.environ["HF_HOME"] = str(args.cache_dir.resolve())
models = {
    "facebook/dinov3-vitl16-pretrain-lvd1689m": ["config.json", "preprocessor_config.json", "model.safetensors"],
    "google/siglip2-so400m-patch16-naflex": None,
    "BAAI/Uni3D": ["modelzoo/uni3d-g/model.pt"],
    "timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k": ["*.json", "*.txt", "open_clip_model.safetensors"],
}
manifest = {}
for model, patterns in models.items():
    info = HfApi().model_info(model)
    print("Downloading", model, info.sha, flush=True)
    snapshot_download(
        model,
        revision=info.sha,
        cache_dir=cache,
        allow_patterns=patterns,
        ignore_patterns=["*.bin"] if model.startswith("google/") else None,
        max_workers=4,
    )
    manifest[model] = info.sha

print(json.dumps(manifest, indent=2), flush=True)
