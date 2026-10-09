# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download the pinned dataset; all executable benchmark code is included in Gym."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from benchmarks.three_d_codebench.prepare import CODE_REVISION, DATA_REVISION, VIEWS, prepare


def setup_assets(output: Path) -> dict:
    """Download only the benchmark programs, prompts, and canonical input images."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = output / ".downloads"
    manifest_path = raw / "manifest.json"
    existing_manifest = manifest_path.exists()
    if existing_manifest:
        manifest = json.loads(manifest_path.read_text())
        if manifest["revision"] != DATA_REVISION:
            raise ValueError("Existing dataset revision differs; choose a new output directory")
        names = manifest["categories"]
    else:
        entries = HfApi().list_repo_tree(
            "YipengGao/3DCode", repo_type="dataset", revision=DATA_REVISION, path_in_repo="3DCodeBench"
        )
        names = sorted(Path(entry.path).name for entry in entries if entry.path.endswith("_seed0"))
    if len(names) != 212:
        raise ValueError(f"Expected 212 canonical benchmark objects, found {len(names)}")
    paths = []
    for name in names:
        paths.extend(
            f"3DCodeBench/{name}/{file}" for file in (f"{name}.py", "prompt_description.txt", "prompt_instruction.txt")
        )
        paths.extend(f"3DCodeBench_ModelLogs/inputs/{name}/images/{view}" for view in VIEWS)

    def download(relative: str) -> dict:
        target = raw / relative
        if not target.exists():
            target = Path(
                hf_hub_download(
                    "YipengGao/3DCode", relative, repo_type="dataset", revision=DATA_REVISION, local_dir=raw
                )
            )
        return {
            "path": relative,
            "size_bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(download, paths))
    manifest = {
        "dataset": "YipengGao/3DCode",
        "revision": DATA_REVISION,
        "code_revision": CODE_REVISION,
        "categories": names,
        "category_count": len(names),
        "files": files,
    }
    if existing_manifest:
        expected = {item["path"]: item["sha256"] for item in json.loads(manifest_path.read_text())["files"]}
        for file in files:
            if expected.get(file["path"], file["sha256"]) != file["sha256"]:
                raise ValueError(f"Cached dataset file differs from its recorded hash: {file['path']}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    prepared = output
    summary = prepare(raw_root=raw, output=prepared)
    rows = []
    for task in ("text_to_3d", "image_to_3d"):
        rows.extend((prepared / f"{task}.jsonl").read_text().splitlines()[:5])
    (prepared / "smoke-5-each.jsonl").write_text("\n".join(rows) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/three_d_codebench"))
    args = parser.parse_args()
    print(json.dumps(setup_assets(args.output), indent=2))


if __name__ == "__main__":
    main()
