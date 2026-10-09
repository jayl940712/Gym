# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the Gym-owned Uni3D encoder and learned checkpoints."""

from benchmarks.three_d_codebench.metrics.shape_uni3d import load_clip, load_uni3d_giant


if __name__ == "__main__":
    model = load_uni3d_giant("cpu")
    print("Uni3D encoder and weights loaded", flush=True)
    del model
    model, tokenizer, preprocess = load_clip("cpu")
    print("OpenCLIP encoder and weights loaded", flush=True)
