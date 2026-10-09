# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone native Interaction2Code metrics, following upstream 29d8af1.

Keep upstream character BLEU, first-contour position, exclusive crop bounds,
and full-page CLIP even when the diagnostic interaction flag is false. This
trusted process reads images only; generated HTML runs in a different sandbox.
"""

import argparse
import contextlib
import fcntl
import functools
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from PIL import Image
from skimage.metrics import structural_similarity


MODEL_CACHE = Path(os.environ.get("I2C_MODEL_CACHE", Path.home() / ".cache/interaction2code"))


@contextlib.contextmanager
def scoring_slot(limit: int = 2):
    """Bound native scorer memory across processes sharing a model cache.

    Acquire before loading neural models. Kernel-owned locks are released even
    when the parent cancels or kills a scoring process.
    """
    if limit < 1:
        raise ValueError("Scoring concurrency must be positive")
    directory = MODEL_CACHE.parent / "scoring-slots"
    directory.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context((directory / f"{slot}.lock").open("a")) for slot in range(limit)]
        selected = None
        while selected is None:
            for handle in handles:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                selected = handle
                break
            if selected is None:
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(selected, fcntl.LOCK_UN)


@functools.lru_cache(maxsize=1)
def clip_model() -> tuple:
    """Load the upstream ViT-B/32 encoder on CPU once per scoring process."""
    import clip
    import torch

    torch.set_num_threads(int(os.environ.get("I2C_SCORING_THREADS", "2")))
    return clip.load("ViT-B/32", device="cpu", download_root=str(MODEL_CACHE / "clip"))


@functools.lru_cache(maxsize=1)
def ocr_reader() -> object:
    """Use the upstream English EasyOCR model, with a Gym-owned weight cache."""
    import easyocr

    return easyocr.Reader(["en"], gpu=False, model_storage_directory=str(MODEL_CACHE / "easyocr"), verbose=False)


def clip_similarity(first: Path, second: Path) -> float:
    """Return upstream cosine similarity between normalized CLIP embeddings."""
    import torch
    from torch.nn.functional import cosine_similarity

    model, preprocess = clip_model()
    with Image.open(first) as image1, Image.open(second) as image2, torch.no_grad():
        features1 = model.encode_image(preprocess(image1).unsqueeze(0))
        features2 = model.encode_image(preprocess(image2).unsqueeze(0))
        features1 = features1 / features1.norm(p=2, dim=-1, keepdim=True)
        features2 = features2 / features2.norm(p=2, dim=-1, keepdim=True)
        return float(cosine_similarity(features1, features2).item())


def ssim_similarity(first: Path, second: Path) -> float:
    """Compare grayscale images after OpenCV's default resize, as upstream does."""
    image1 = cv2.imread(str(first), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(str(second), cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    return float(structural_similarity(image1, resized, full=True)[0])


def image_text(path: Path) -> str:
    """Concatenate OCR detections with upstream's leading space."""
    with Image.open(path) as image:
        detections = ocr_reader().readtext(np.array(image))
    return "".join(" " + text for _, text, _ in detections)


def text_similarity(first: Path, second: Path) -> float | None:
    """Preserve character BLEU and upstream's null for one-sided empty OCR."""
    reference, candidate = image_text(first), image_text(second)
    if reference and candidate:
        return float(sentence_bleu([reference], candidate))
    return 1.0 if not reference and not candidate else None


def annotated_position(path: Path) -> tuple[float, float]:
    """Use the first red contour, rather than replacing upstream's selection rule."""
    image = cv2.imread(str(path))
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) | cv2.inRange(
        hsv, np.array([170, 120, 70]), np.array([180, 255, 255])
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No red annotation contour in {path}")
    x, y, width_box, height_box = cv2.boundingRect(contours[0])
    return (x + width_box // 2) / width, (y + height_box // 2) / height


def difference_bounds(first: np.ndarray, second: np.ndarray) -> tuple[int, int]:
    """Find changed rows, preserving the original crop coordinate conventions.

    Upstream's unequal-height branch invokes git diff on untracked CSVs and can
    produce no output. --no-index makes that intended diff independent of a repo.
    """
    if first.shape[1] != second.shape[1]:
        return 0, second.shape[0]
    if first.shape == second.shape:
        changed = np.flatnonzero(np.any(first != second, axis=1))
        if not len(changed):
            raise ValueError("Images have no changed rows")
        return int(changed.min()), int(changed.max())
    with tempfile.TemporaryDirectory(prefix="i2c-diff-") as temporary:
        before, after = Path(temporary) / "before.csv", Path(temporary) / "after.csv"
        np.savetxt(before, first, fmt="%d", delimiter=",")
        np.savetxt(after, second, fmt="%d", delimiter=",")
        result = subprocess.run(
            ["git", "diff", "--no-index", "--", str(before), str(after)], capture_output=True, check=False
        )
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"git diff failed: {result.stderr.decode(errors='replace')}")
    positions = []
    position = None
    for line in result.stdout.decode(errors="replace").splitlines():
        match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if match:
            position = int(match[1])
        elif position is not None and line[:1] in {" ", "+", "-"}:
            positions.append(position)
            if line[0] != "-":
                position += 1
    if not positions:
        raise ValueError("Images have no differing rows")
    return min(positions), max(positions)


def interaction_crop(before: Path, after: Path, destination: Path) -> tuple[float, float]:
    """Crop the changed area with the upstream exclusive maximum coordinates."""
    with Image.open(before) as image1, Image.open(after) as image2:
        a, b = np.array(image1.convert("RGB")), np.array(image2.convert("RGB"))
        y_min, y_max = difference_bounds(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1))
        a_rot, b_rot = np.rot90(a, -1), np.rot90(b, -1)
        x_min, x_max = difference_bounds(a_rot.reshape(a_rot.shape[0], -1), b_rot.reshape(b_rot.shape[0], -1))
        image2.crop((x_min, y_min, x_max, y_max)).save(destination)
    return (x_min + x_max) / 2, (y_min + y_max) / 2


def zero_metrics() -> tuple[dict, dict]:
    """Create all original metric fields for an unimplemented interaction."""
    full = {"clip_similarity": 0.0, "text_similarity": 0.0, "structure_similarity": 0.0}
    return full, full | {"position_similarity": 0.0, "position_similarity_after": 0.0}


def evaluate(*, dataset: Path, task: Path, artifacts: Path) -> dict:
    """Evaluate downloaded browser screenshots with private reference images."""
    metadata = json.loads((task / "task.json").read_text())
    render = json.loads((artifacts / "render.json").read_text())
    full, interaction = zero_metrics()
    if render.get("status") != "ok":
        raise RuntimeError(f"Browser execution failed: {render}")
    screenshots = []
    for name in render["screenshots"]:
        if Path(name).name != name or not name.endswith(".png"):
            raise ValueError("Unsafe browser screenshot name")
        path = artifacts / name
        with Image.open(path) as image:
            image.verify()
        screenshots.append(path)
    references = {}
    page = (dataset / "source" / str(metadata["page_id"])).resolve()
    for key, name in metadata["references"].items():
        path = (page / name).resolve()
        if not path.is_relative_to(page) or Path(name).name != name:
            raise ValueError("Reference image escapes dataset page")
        references[key] = path
    candidates = [path for path in screenshots if path.name != "0_source.png" and "interact" not in path.name]
    best = None
    for candidate in candidates:
        similarity = clip_similarity(references["after"], candidate)
        if similarity > full["clip_similarity"]:
            full["clip_similarity"], best = similarity, candidate
    html = (artifacts / "index.html").read_text(errors="replace")
    flag, message = True, "Good"
    if "</html>" not in html:
        flag, message = False, "No html tag: [Page generate fail]"
    elif len(screenshots) != 2:
        flag, message = False, f"{len(screenshots)} images: [No interaction]"
    else:
        with Image.open(screenshots[0]) as image1, Image.open(screenshots[1]) as image2:
            if image1.mode == image2.mode and np.array_equal(np.array(image1), np.array(image2)):
                flag, message = False, "Same image: [No interaction or Interaction Fail]"
    if best is not None and flag:
        full["text_similarity"] = text_similarity(references["after"], best)
        full["structure_similarity"] = ssim_similarity(references["after"], best)
        src_x, src_y = annotated_position(references["before_mark"])
        dst_x, dst_y = annotated_position(references["after_mark"])
        with Image.open(artifacts / "0_source.png") as source_image:
            width, height = source_image.size
        _, x, y, _ = best.name.split("_")
        interaction["position_similarity"] = 1 - max(abs(int(x) / width - src_x), abs(int(y) / height - src_y))
        try:
            crop = artifacts / "interact.png"
            center_x, center_y = interaction_crop(artifacts / "0_source.png", best, crop)
            with Image.open(best) as clicked:
                after_width, after_height = clicked.size
            interaction["position_similarity_after"] = 1 - max(
                abs(int(center_x) / after_width - dst_x), abs(int(center_y) / after_height - dst_y)
            )
            interaction["structure_similarity"] = ssim_similarity(references["crop"], crop)
            interaction["clip_similarity"] = clip_similarity(references["crop"], crop)
            interaction["text_similarity"] = text_similarity(references["crop"], crop)
        except (SystemError, ValueError) as error:
            flag, message = False, f"Interaction crop failed: {error}"
    return {
        "reward": interaction["clip_similarity"],
        "status": "ok" if flag else "no_interaction",
        "full_page": full,
        "interaction": interaction,
        "interaction_flag": flag,
        "diagnostic": message,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    if args.prefetch:
        clip_model()
        ocr_reader()
        print("Interaction2Code metric weights are ready")
    else:
        with scoring_slot(int(os.environ.get("I2C_SCORING_CONCURRENCY", "2"))):
            print(
                json.dumps(evaluate(dataset=args.dataset, task=args.task, artifacts=args.artifacts), allow_nan=False)
            )
