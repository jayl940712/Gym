# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the complete pinned dataset without an Interaction2Code checkout."""

import argparse
import hashlib
import json
import logging
import shutil
import struct
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmarks.interaction2code.prompt import build_prompt


LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
DATASET_ID = "whale99/Interaction2Code"
DATASET_REVISION = "7a2d55fe666ddd319d26bdcb1dddb01c764a8602"
UPSTREAM_REVISION = "29d8af1572888d33b028ef001dc8452f9e88497b"
PROTOCOL = "interaction2code-opencode"


def fetch(url: str) -> bytes:
    """Download a preparation asset with bounded retries for transient failures."""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def png_size(path: Path) -> tuple[int, int]:
    """Read the PNG IHDR dimensions without adding image libraries to Gym core."""
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width == 0 or height == 0:
        raise ValueError(f"Empty image: {path}")
    return width, height


def export(*, source: Path, output: Path, pages: list[int], placeholder: Path) -> Path:
    """Validate every numbered interaction and export one task per screenshot pair."""
    rows = []
    counts = {}
    hashes = {}
    output.mkdir(parents=True, exist_ok=True)
    for page_id in pages:
        page = source / str(page_id)
        annotation = json.loads((page / "action.json").read_text())
        interactions = sorted((key for key in annotation if key.isdigit()), key=int)
        if not interactions:
            raise ValueError(f"Page {page_id} has no interactions")
        counts[str(page_id)] = len(interactions)
        for interaction_id in interactions:
            entry = annotation[interaction_id]
            src, dst = str(entry["src"]), str(entry["dst"])
            if not src.isdigit() or not dst.isdigit() or int(interaction_id) < 1:
                raise ValueError(f"Invalid screenshot/interaction ID on page {page_id}")
            names = [
                f"{src}.png",
                f"{dst}.png",
                f"{src}_mark.png",
                f"{dst}_mark.png",
                f"interaction_{interaction_id}.png",
            ]
            for name in names:
                asset = page / name
                png_size(asset)
                hashes[f"{page_id}/{name}"] = hashlib.sha256(asset.read_bytes()).hexdigest()
            width, height = png_size(page / f"{src}.png")
            record_id = f"{page_id}-{interaction_id}"
            directory = output / "tasks" / record_id
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(page / f"{src}.png", directory / "before.png")
            shutil.copyfile(page / f"{dst}.png", directory / "after.png")
            shutil.copyfile(placeholder, directory / "placeholder.jpg")
            metadata = {
                "page_id": page_id,
                "interaction_id": int(interaction_id),
                "record_id": record_id,
                "width": width,
                "height": height,
                "topic": annotation.get("topic", ""),
                "framework": annotation.get("framework", ""),
                "references": {
                    "before_mark": f"{src}_mark.png",
                    "after_mark": f"{dst}_mark.png",
                    "crop": f"interaction_{interaction_id}.png",
                    "after": f"{dst}.png",
                },
            }
            (directory / "task.json").write_text(json.dumps(metadata, indent=2) + "\n")
            rows.append(
                {
                    "page_id": page_id,
                    "interaction_id": int(interaction_id),
                    "task_id": record_id,
                    "responses_create_params": {
                        "input": [{"role": "user", "content": build_prompt(width=width, height=height)}],
                        "temperature": 0.7,
                    },
                }
            )
    destination = output / "interaction2code_benchmark.jsonl"
    destination.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "protocol": PROTOCOL,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "upstream_revision": UPSTREAM_REVISION,
        "pages": len(pages),
        "interactions": len(rows),
        "counts": counts,
        "asset_sha256": hashes,
        "placeholder_sha256": hashlib.sha256(placeholder.read_bytes()).hexdigest(),
        "published_interaction_count": 374,
        "note": "The pinned full release has 375 numbered interactions; none are silently excluded.",
        "attribution": "Interaction2Code, Xiao et al., ASE 2025, https://github.com/WebPAI/Interaction2Code",
        "license": "Upstream does not declare a dataset license; reference screenshots and annotations are downloaded, and the upstream placeholder is bundled.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def prepare(*, limit: int | None = None, output: Path = ROOT / "data") -> Path:
    """Gym preparation entry point; limit restricts pages only for smoke tests."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    output = Path(output).resolve()
    source = output / "source"
    source.mkdir(parents=True, exist_ok=True)
    data = json.loads(fetch(f"https://huggingface.co/api/datasets/{DATASET_ID}/revision/{DATASET_REVISION}"))
    files = [entry["rfilename"] for entry in data["siblings"]]
    pages = sorted(int(Path(name).parent.name) for name in files if name.endswith("/action.json"))
    if len(pages) != 127 or len(set(pages)) != len(pages):
        raise ValueError("Pinned dataset must have exactly 127 distinct annotated pages")
    pages = pages[:limit] if limit is not None else pages
    selected = [
        name
        for name in files
        if name.split("/")[0] in {str(page) for page in pages}
        and (name.endswith(".png") or name.endswith("/action.json"))
    ]

    def download(name: str) -> None:
        path = (source / name).resolve()
        if not path.is_relative_to(source) or len(Path(name).parts) != 2:
            raise ValueError(f"Unsafe dataset path: {name}")
        if path.exists() and path.stat().st_size:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = fetch(f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{name}")
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(payload)
        temporary.replace(path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, _ in enumerate(pool.map(download, selected), 1):
            if index % 100 == 0:
                LOG.info("Downloaded/checked %s/%s assets", index, len(selected))
    # Bundle the upstream solid-blue placeholder so preparation never needs its code repository.
    placeholder = ROOT / "placeholder.jpg"
    return export(source=source, output=output, pages=pages, placeholder=placeholder)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    print(prepare(limit=args.limit, output=args.output))
