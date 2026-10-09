# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare all native metrics with pinned upstream functions on a real episode.

This optional development audit downloads source functions for comparison only.
Neither preparation nor evaluation requires an upstream checkout or this audit.
Run with the isolated scorer's Python and --artifacts pointing to a scored episode.
"""

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import clip
import cv2
import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu
from PIL import Image
from skimage.metrics import structural_similarity
from torch.nn.functional import cosine_similarity


GYM_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REVISION = "29d8af1572888d33b028ef001dc8452f9e88497b"


def functions(filename: str, namespace: dict) -> None:
    """Compile pinned function definitions without upstream GUI or import side effects."""
    url = f"https://raw.githubusercontent.com/WebPAI/Interaction2Code/{UPSTREAM_REVISION}/code/metric/{filename}"
    with urllib.request.urlopen(url, timeout=60) as response:
        tree = ast.parse(response.read().decode())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(tree, url, "exec"), namespace)


def verify(*, artifacts: Path, dataset: Path, output: Path) -> dict:
    """Compare all eight metric fields and the native diagnostic flag."""
    spec = importlib.util.spec_from_file_location(
        "gym_scorer", GYM_ROOT / "responses_api_agents/interaction2code_agent/scorer.py"
    )
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    saved = json.loads((artifacts / "result.json").read_text())
    page_id, interaction_id = saved["page_id"], saved["interaction_id"]
    model, preprocess = native.clip_model()
    reader = native.ocr_reader()
    namespace = {
        "np": np,
        "Image": Image,
        "cv2": cv2,
        "os": os,
        "re": re,
        "time": time,
        "json": json,
        "shutil": shutil,
        "subprocess": subprocess,
        "ssim": structural_similarity,
        "torch": torch,
        "clip": clip,
        "cosine_similarity": cosine_similarity,
        "sentence_bleu": sentence_bleu,
        "device": "cpu",
        "model": model,
        "preprocess": preprocess,
        "easyocr": SimpleNamespace(Reader=lambda _: reader),
    }
    utilities = namespace.copy()
    functions("metric_utils.py", utilities)
    namespace.update({name: value for name, value in utilities.items() if callable(value)})
    functions("calculate_metric.py", namespace)
    captured = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="i2c-upstream-") as temporary, contextlib.redirect_stdout(captured):
        root = Path(temporary)
        page = root / str(page_id)
        shutil.copytree(dataset / "source" / str(page_id), page)
        generated = page / "result" / f"{interaction_id}-direct_prompt-test"
        generated.mkdir(parents=True)
        shutil.copyfile(artifacts / "index.html", generated.with_suffix(".html"))
        for name in json.loads((artifacts / "render.json").read_text())["screenshots"]:
            shutil.copyfile(artifacts / name, generated / name)
        namespace["prediction_path"] = str(root) + "/"
        previous = Path.cwd()
        try:
            os.chdir(root)
            full, interaction, diagnostic = namespace["get_all_score"](
                str(page_id), str(interaction_id), "test", "direct_prompt"
            )
        finally:
            os.chdir(previous)
    differences = {}
    for group, values in (("full_page", full), ("interaction", interaction)):
        for name, value in values.items():
            expected = saved[group][name]
            equal = (
                value is None and expected is None
                if value is None or expected is None
                else abs(float(value) - expected) <= 1e-6
            )
            if not equal:
                differences[f"{group}.{name}"] = {"upstream": value, "gym": expected}
    if diagnostic["flag"] != saved["interaction_flag"]:
        differences["interaction_flag"] = {"upstream": diagnostic["flag"], "gym": saved["interaction_flag"]}
    result = {
        "upstream_revision": UPSTREAM_REVISION,
        "task_id": f"{page_id}-{interaction_id}",
        "metric_fields_checked": 8,
        "flag_checked": True,
        "tolerance": 1e-6,
        "passed": not differences,
        "differences": differences,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    output.with_suffix(".stdout.txt").write_text(captured.getvalue())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=GYM_ROOT / "benchmarks/interaction2code/data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(artifacts=args.artifacts.resolve(), dataset=args.dataset.resolve(), output=args.output.resolve())
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)
