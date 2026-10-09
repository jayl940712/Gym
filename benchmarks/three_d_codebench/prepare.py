# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare pinned text/image OpenCode inputs without exposing reference programs."""

import argparse
import json
import shutil
from pathlib import Path


CODE_REVISION = "42c7780ed3fcbd466f17f058f62e7996233777f7"
DATA_REVISION = "0db485b63a2991231f0fad6c6e0dc01bcbbced1e"
VIEWS = ["Image_005.png", "Image_015.png", "Image_025.png", "Image_035.png"]


def prepare(*, raw_root: Path, output: Path, instances: list[str] | None = None) -> dict:
    """Separate task inputs from trusted references at the pinned dataset revision."""
    manifest = json.loads((raw_root / "manifest.json").read_text())
    if manifest["revision"] != DATA_REVISION:
        raise ValueError("Dataset revision differs from the pinned 3DCodeBench snapshot")
    names = instances or manifest["categories"]
    if set(names) - set(manifest["categories"]):
        raise ValueError("Requested unknown instances")
    output.mkdir(parents=True, exist_ok=True)
    rows = {"text_to_3d": [], "image_to_3d": []}
    for name in names:
        original = raw_root / "3DCodeBench" / name
        inputs = raw_root / "3DCodeBench_ModelLogs/inputs" / name
        reference = output / "reference" / name
        reference.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / f"{name}.py", reference / f"{name}.py")
        for kind in ("description", "instruction"):
            shutil.copyfile(original / f"prompt_{kind}.txt", reference / f"prompt_{kind}.txt")
        (reference / "images").mkdir(exist_ok=True)
        for view in VIEWS:
            shutil.copyfile(inputs / "images" / view, reference / "images" / view)
        for task in rows:
            directory = output / task / name
            directory.mkdir(parents=True, exist_ok=True)
            images = VIEWS if task == "image_to_3d" else []
            for view in images:
                shutil.copyfile(inputs / "images" / view, directory / view)
            system = (Path(__file__).resolve().parent / "prompts" / f"{task}_system_prompt.txt").read_text().strip()
            user = (
                (original / "prompt_description.txt").read_text().strip()
                if task == "text_to_3d"
                else "Reconstruct the object shown in the following reference image(s) as a "
                "Blender 5.0 Python script. Treat all images as views of the SAME object."
            )
            instruction = "\n".join(f"Use the read tool to inspect /workspace/{view}." for view in images)
            workflow = (
                "\n\nYou have a terminal and Blender 5.0 on PATH. Write your complete program to "
                "/workspace/solution.py. Test with `blender --background --factory-startup "
                "--python-exit-code 1 --python /workspace/solution.py` and revise as needed. "
                "Your final response must contain the complete Python source, without prose or fences. "
                "Evaluation will independently execute solution.py in a fresh sandbox."
            )
            metadata = {
                "task": task,
                "record_id": name,
                "images": images,
                "prompt_type": "description" if not images else "image",
                "code_revision": CODE_REVISION,
                "dataset_revision": DATA_REVISION,
            }
            (directory / "task.json").write_text(json.dumps(metadata, indent=2) + "\n")
            rows[task].append(
                {
                    "task": task,
                    "record_id": name,
                    "responses_create_params": {
                        "input": [{"role": "user", "content": system + "\n\n" + user + "\n" + instruction + workflow}]
                    },
                }
            )
    for task, values in rows.items():
        (output / f"{task}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in values))
    combined = rows["text_to_3d"] + rows["image_to_3d"]
    (output / "benchmark.jsonl").write_text("".join(json.dumps(row) + "\n" for row in combined))
    summary = {
        "code_revision": CODE_REVISION,
        "dataset_revision": DATA_REVISION,
        "instances": names,
        "counts": {task: len(values) for task, values in rows.items()},
        "protocol": "3dcodebench-opencode",
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", nargs="+")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                raw_root=args.raw_root.resolve(),
                output=args.output.resolve(),
                instances=args.instances,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
