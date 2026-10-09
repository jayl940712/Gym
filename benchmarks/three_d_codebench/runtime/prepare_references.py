# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted one-time GLB generation using pinned reference scripts and exporter."""

import argparse
import hashlib
import json
import re
import struct
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def has_surface_mesh(path: Path) -> bool:
    """Reject GLBs that the exporter labels OK despite containing no faces."""
    try:
        with path.open("rb") as stream:
            magic, version, length = struct.unpack("<4sII", stream.read(12))
            chunk_length, chunk_type = struct.unpack("<II", stream.read(8))
            if magic != b"glTF" or version != 2 or chunk_type != 0x4E4F534A or length != path.stat().st_size:
                return False
            document = json.loads(stream.read(chunk_length))
        accessors = document.get("accessors", [])
        return any(
            primitive.get("mode", 4) in (4, 5, 6)
            and accessors[primitive["attributes"]["POSITION"]]["count"] >= 3
            and ("indices" not in primitive or accessors[primitive["indices"]]["count"] >= 3)
            for mesh in document.get("meshes", [])
            for primitive in mesh.get("primitives", [])
        )
    except (OSError, ValueError, KeyError, IndexError, struct.error):
        return False


def published_elkhorn_reference(script: Path, destination: Path, runtime: dict) -> dict:
    """Use the publisher's canonical seed-zero mesh for its corrupted eval script."""
    revision = "0db485b63a2991231f0fad6c6e0dc01bcbbced1e"
    source = "3DCodeData/ElkhornCoral_000/ElkhornCoral_000_geo.glb"
    digest = "01755bcff56f3ec5020038548a72550dc92a121bef9b572892a4feb5d95e47c9"
    original_digest = hashlib.sha256(script.read_bytes()).hexdigest()
    # Check the broken specialization before applying this category-specific fallback.
    if original_digest != "14a3d98e72a1fcfb0c7dcab4c75fd031e1b42aff3f5612241d81b5c1080bc514":
        raise ValueError("Elkhorn reference changed; review the published-mesh fallback")
    url = f"https://huggingface.co/datasets/YipengGao/3DCode/resolve/{revision}/{source}"
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("Published Elkhorn reference checksum mismatch")
    temporary = destination.with_suffix(".download")
    temporary.write_bytes(payload)
    if not has_surface_mesh(temporary):
        temporary.unlink()
        raise ValueError("Published Elkhorn reference contains no surface mesh")
    temporary.replace(destination)
    provenance = {
        "reason": "Benchmark specialization corrupts XY jitter and ring_interpolation, producing zero faces",
        "dataset": "YipengGao/3DCode",
        "revision": revision,
        "source_path": source,
        "source_sha256": digest,
        "original_script_sha256": original_digest,
        "canonical_script_path": "3DCodeData/ElkhornCoral_000/ElkhornCoral_000_geo.py",
        "scope": "Reference shape only; original script, prompts, conditioning images, and native scorers unchanged",
    }
    (destination.parent / "reference_compat.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return {
        "status": "OK",
        "out_glb": str(destination),
        "size_kb": round(len(payload) / 1024, 1),
        "runtime": runtime,
        "reference_source": provenance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--runtime-manifest", type=Path)
    args = parser.parse_args()
    runtime_path = args.runtime_manifest
    if runtime_path is not None:
        runtime = json.loads(runtime_path.read_text())
    else:
        probe = subprocess.run(
            [
                "blender",
                "--background",
                "--factory-startup",
                "--python-expr",
                'import bpy,json; print("GYM_RUNTIME="+json.dumps({"openvdb":bpy.app.build_options.openvdb,"blender":bpy.app.version_string}))',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        runtime = json.loads(
            next(
                line.removeprefix("GYM_RUNTIME=")
                for line in probe.stdout.splitlines()
                if line.startswith("GYM_RUNTIME=")
            )
        )
    if not runtime.get("openvdb"):
        raise RuntimeError("Reference preparation requires a validated Blender build with OpenVDB enabled")
    manifest = json.loads((args.prepared_root / "manifest.json").read_text())
    names = manifest["instances"][: args.limit] if args.limit else manifest["instances"]

    def export(name):
        directory = args.prepared_root / "reference" / name
        glb = directory / "glb"
        glb.mkdir(exist_ok=True)
        log_path = glb / "export_log.json"
        script = directory / f"{name}.py"
        needs_openvdb = bool(re.search(r"voxel_remesh|['\"]REMESH['\"]", script.read_text()))
        if (glb / f"{name}.glb").is_file() and log_path.is_file():
            log = json.loads(log_path.read_text())
            if (
                log["status"] == "OK"
                and (not needs_openvdb or log.get("runtime", {}).get("openvdb"))
                and has_surface_mesh(glb / f"{name}.glb")
            ):
                return name, log
        if name == "ElkhornCoral_seed0":
            log = published_elkhorn_reference(script, glb / f"{name}.glb", runtime)
            log_path.write_text(json.dumps(log, indent=2) + "\n")
            return name, log
        # The shipped FanCoral freezes five endpoints but sizes its sparse arrays
        # from a newly computed candidate count. Use the frozen list's length.
        # Preserve the original file and record the exact repair provenance.
        if name == "FanCoral_seed0":
            original = script.read_text()
            needle = "endpoints = np.array([572, 248, 14, 253, 26])"
            if original.count(needle) != 1:
                raise ValueError("FanCoral reference changed; review the compatibility repair")
            repaired = original.replace(needle, needle + "\nn_ep = len(endpoints)")
            script = glb / "reference_compat.py"
            script.write_text(repaired)
            provenance = {
                "reason": "Size sparse virtual-root arrays from the existing frozen endpoint list",
                "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "repaired_sha256": hashlib.sha256(repaired.encode()).hexdigest(),
                "replacement": "n_ep = len(endpoints)",
            }
            (glb / "reference_compat.json").write_text(json.dumps(provenance, indent=2) + "\n")
        result = subprocess.run(
            [
                "blender",
                "--background",
                "--factory-startup",
                "-t",
                "2",
                "--python-exit-code",
                "1",
                "--python",
                str(Path(__file__).with_name("export_glb.py")),
                "--",
                "--blender-export",
                "--script",
                str(script),
                "--out-glb",
                str(glb / f"{name}.glb"),
                "--log-path",
                str(log_path),
            ],
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        (glb / "export.stdout").write_text(result.stdout)
        (glb / "export.stderr").write_text(result.stderr)
        if not log_path.is_file():
            raise RuntimeError(f"{name}: reference export produced no log; exit {result.returncode}")
        log = json.loads(log_path.read_text())
        if log["status"] != "OK":
            raise RuntimeError(f"{name}: reference preparation failed: {log}")
        if not has_surface_mesh(glb / f"{name}.glb"):
            raise RuntimeError(f"{name}: reference export contains no surface mesh despite status OK")
        log["runtime"] = runtime
        log_path.write_text(json.dumps(log, indent=2) + "\n")
        return name, log

    records = {}
    failures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(export, name): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                _, log = future.result()
                records[name] = log
            except Exception as error:
                failures[name] = str(error)
                print(f"Reference failed: {name}: {error}", flush=True)
    (args.prepared_root / f"reference-validation-{len(names)}.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.prepared_root / "reference-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} reference exports failed; successful exports are retained")
    print(f"Validated {len(records)} native reference GLBs")


if __name__ == "__main__":
    main()
