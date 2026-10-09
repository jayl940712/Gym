#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3DCodeBench; see third_party/NOTICE.md for provenance.
"""Aggregate per-stage + fine-grained failure breakdown across all model
directories under one or more results-roots.

Top-level statuses come from each instance's render_log.json:
    OK, ERR_EXEC, ERR_NO_MESH, ERR_TIMEOUT, ERR_NOLOG, ERR_PARSE, MISSING

We then sub-categorise the ERR_EXEC bucket — by far the biggest source of
failures — into four families based on regex-matching the first non-traceback
line of the captured Python error:

    B5-API : Blender 4.x → 5.0 API drift (renamed sockets / props / enums /
             operator kwargs, removed operators or modules). Diagnostic of a
             model trained on Blender 4 docs.
    BMSH   : bmesh state mistakes (outdated lookup table, stale BMVert/BMEdge,
             duplicate verts/faces, etc.). Model's mesh code is semantically
             wrong even on the new API.
    CTX    : bpy.ops invoked in the wrong execution context
             ("ValueError: 1-2 args execution context is supported").
    OTHER  : every other Python-level failure (IndexError, NoneType,
             KeyError on dicts, type errors that aren't API drift).

Usage:
    python metrics/failure_taxonomy.py
        [--roots results/text_to_3D results/image_to_3D ...]
        [--out  /tmp/failure_taxonomy.json]
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


_PY_FRAME_RE = re.compile(r'File ".*?\.py", line \d+')


def fingerprint(err: str) -> str:
    """Match metrics/executability.py — first non-traceback line, frame-stripped."""
    if not err:
        return ""
    for line in err.splitlines():
        line = line.strip()
        if line and not line.startswith("Traceback") and not line.startswith("File "):
            line = _PY_FRAME_RE.sub("File <...>", line)
            return line[:200]
    return err.splitlines()[0][:200]


# Order matters — first match wins. CTX is checked before OTHER because it's
# also a ValueError but with a distinctive message.
_B5_API_PATTERNS = [
    # Material / shader socket renames (Principled BSDF).
    re.compile(
        r'KeyError:.*key "(Specular|Subsurface|Subsurface Color|Transmission|Sheen|Sheen Tint|Clearcoat|Clearcoat Roughness|Anisotropic|Specular Tint)" not found'
    ),
    # Mesh / object / material props removed in 5.0.
    re.compile(
        r"AttributeError:.*'Mesh' object has no attribute '(use_auto_smooth|calc_normals|calc_normals_split|calc_loop_triangles|auto_smooth_angle)'"
    ),
    re.compile(r"AttributeError:.*'Object' object has no attribute '(shade_smooth|use_auto_smooth)'"),
    re.compile(
        r"AttributeError:.*'Material' object has no attribute '(shadow_method|blend_method|use_screen_refraction|use_sss_translucency)'"
    ),
    # Modifier prop renames.
    re.compile(
        r"AttributeError:.*'(Bevel|SimpleDeform|Subsurf|Solidify|Mirror|Array|Bisect|Decimate|Smooth|Wave|Cast|Hook|Lattice|MeshDeform|SurfaceDeform|VolumeToMesh|VolumeDisplace|Particle|ParticleSystem)Modifier' object has no attribute"
    ),
    re.compile(
        r"AttributeError:.*'BevelModifier' object has no attribute '(clamp_overlap|amount|angle|width|miter_outer|miter_inner)'"
    ),
    re.compile(r"AttributeError:.*'ParticleSettings' object has no attribute"),
    re.compile(r"AttributeError:.*'BMLayerAccessEdge' object has no attribute 'crease'"),  # bmesh layer rename
    re.compile(r"AttributeError:.*'NoiseTexture' object has no attribute 'noise_scale'"),
    # Enum identifier renames.
    re.compile(
        r'enum "(BLENDER_EEVEE_NEXT|FAST|SUBSURFACE|SUBDIVISION_SURFACE|SUBDIVISION|SMOOTH_BY_ANGLE|ARC|WAVE|PLANAR)" not found'
    ),
    re.compile(r"enum.*not found in \('GREASE_PENCIL"),  # any modifier-type enum miss
    # Operator signature changes (kwargs renamed/removed).
    re.compile(
        r"TypeError:.*(create_cone|create_cylinder|create_uvsphere|create_icosphere|create_grid|create_circle|create_torus): keyword"
    ),
    re.compile(r'TypeError:.*subdivide_edges: keyword "number_cuts"'),
    re.compile(r"TypeError:.*(bevel|spin|extrude|inset|loop_cut): keyword"),
    re.compile(
        r'TypeError: Converting py args to operator properties:: keyword "(use_auto_smooth|subdivisions|smoothness)" unrecognized'
    ),
    # Removed operators / modules.
    re.compile(
        r'AttributeError: BMeshOpsModule: operator "(create_cylinder|recalc_normals|subdivide|delete|remove_doubles|spin|connect_verts|bridge_loops|extrude_face_region|extrude_edge_only|smooth_vert)" doesn\'t exist'
    ),
    re.compile(r"ImportError: cannot import name '(Noise|noise)' from 'mathutils'"),
    re.compile(r"NameError: name 'mathutils' is not defined"),
    re.compile(r"RuntimeError: Error: Node type ShaderNodeTex(Musgrave|Brick|Checker|Magic|Wave|Gradient) undefined"),
    re.compile(r"RuntimeError: Error: Node type GeometryNode\w+ undefined"),
    # Modifier .new() with deprecated type enum.
    re.compile(r'TypeError: ObjectModifiers\.new\(\): error with keyword argument "type"'),
    re.compile(r'TypeError: BlendDataTextures\.new\(\): error with keyword argument "type"'),
]

_BMSH_PATTERNS = [
    re.compile(r"IndexError: BMElemSeq\[index\]: outdated internal index table"),
    re.compile(r"ReferenceError: BMesh data of type \w+ has been removed"),
    re.compile(r"ReferenceError: StructRNA of type \w+ has been removed"),
    re.compile(r"ValueError: faces\.new\(\.\.\.\): face already exists"),
    re.compile(r"ValueError: faces\.new\(\.\.\.\): found the same \(BMVert\) used multiple times"),
    re.compile(r"KeyError: '(geom|verts|edges|faces|loops)'"),  # bmesh.ops result-dict key misuse
]

_CTX_PATTERNS = [
    re.compile(r"ValueError: 1-2 args execution context is supported"),
    re.compile(r"RuntimeError: Operator bpy\.ops\.\S+ poll\(\) failed"),
]


def categorize(fp: str) -> str:
    """Map a fingerprint to one of {B5-API, BMSH, CTX, OTHER}."""
    if not fp:
        return "OTHER"
    for pat in _B5_API_PATTERNS:
        if pat.search(fp):
            return "B5-API"
    for pat in _BMSH_PATTERNS:
        if pat.search(fp):
            return "BMSH"
    for pat in _CTX_PATTERNS:
        if pat.search(fp):
            return "CTX"
    return "OTHER"


def aggregate_model(model_dir: Path):
    """Walk every <inst>/renders/render_log.json under one model dir and
    return a dict of {n_total, by_status, by_subcategory_for_err_exec, top_errors}."""
    by_status = Counter()
    sub = Counter()  # only ERR_EXEC instances
    examples = defaultdict(Counter)  # sub-category -> Counter(fingerprint)
    n_missing = 0
    n_total = 0

    for inst in sorted(model_dir.iterdir()):
        if not inst.is_dir() or inst.name.startswith("_"):
            continue
        n_total += 1
        log = inst / "renders" / "render_log.json"
        if not log.exists():
            by_status["MISSING"] += 1
            n_missing += 1
            continue
        try:
            rec = json.loads(log.read_text())
        except Exception:
            by_status["BAD_LOG"] += 1
            continue

        status = rec.get("status", "?")
        by_status[status] += 1
        if status != "ERR_EXEC":
            continue

        fp = fingerprint(rec.get("error") or "")
        cat = categorize(fp)
        sub[cat] += 1
        examples[cat][fp] += 1

    return {
        "n_total": n_total,
        "by_status": dict(by_status),
        "err_exec_subcategory": dict(sub),
        "examples": {k: v.most_common(5) for k, v in examples.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--roots",
        nargs="+",
        default=[
            "results/text_to_3D",
            "results/image_to_3D",
        ],
    )
    p.add_argument("--out", default="/tmp/failure_taxonomy.json")
    args = p.parse_args()

    out = {}
    for root in args.roots:
        rp = Path(root)
        if not rp.exists():
            print(f"  SKIP {rp} (not found)")
            continue
        track_key = rp.name  # text_to_3D / image_to_3D
        out[track_key] = {}
        for model_dir in sorted(rp.iterdir()):
            if not model_dir.is_dir():
                continue
            r = aggregate_model(model_dir)
            out[track_key][model_dir.name] = r
            n_ok = r["by_status"].get("OK", 0)
            print(
                f"{track_key:>12s}  {model_dir.name:<30s}  "
                f"OK={n_ok:>3d}/{r['n_total']:<3d}  "
                f"ERR_EXEC subs: "
                + " ".join(f"{k}={v}" for k, v in sorted(r["err_exec_subcategory"].items(), key=lambda x: -x[1]))
            )

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
