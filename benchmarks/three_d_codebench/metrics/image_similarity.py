#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3DCodeBench; see third_party/NOTICE.md for provenance.
"""Image-image similarity metric for image-to-3D eval — two aggregates.

For each instance, we have:
    rendered output:  results/<root>/<model>/<inst>/renders/Image_{005,015,025,035}.png
    reference input:  data/<inst>/images/Image_{005,015,025,035}.png

Both come from the same Infinigen turntable convention (azimuths
45/135/225/315 deg), so the *intended* pairing is positional. But the
generative model picks its own canonical orientation when it places the
object at the origin, and a 90/180 deg yaw mismatch can drag the
view-paired score way down even when geometry is correct (verified
empirically — e.g. CellShelf_seed0 var=0.26 across views).

We therefore report TWO per-instance aggregates over the same 4×4 cosine
matrix between rendered and reference embeddings:

  view-paired   score_v = cos(rend[v], ref[v]),  mean over v ∈ {0..3}
                — what was originally reported. Penalises yaw rotation.

  best-assignment   Hungarian 1-to-1 matching on the full 4×4 cosine
                    matrix, mean over the 4 matched pairs. Immune to
                    cyclic 90 deg yaw rotations and any view permutation.

Both go through the same per-aggregate (conditional, penalized) story:
    conditional  — averaged over instances that have all 4 renders.
                   "If the script ran, how well did the geometry match?"
    penalized    — failed/missing renders count as 0; averaged over all 212.
                   "Overall fidelity, including failures."

Encoders (toggle with --encoder):
    siglip2  google/siglip2-so400m-patch16-naflex   image features
    dinov2   facebook/dinov2-large                  pooler_output
    dinov3   facebook/dinov3-vitl16-pretrain-lvd1689m  pooler_output (gated)

Usage:
    python eval/metrics/image_similarity.py \\
        --results-root results/image_to_3D \\
        --model gemini-3-flash-preview \\
        --encoder siglip2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import AutoImageProcessor, AutoModel, AutoProcessor


EVAL_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = EVAL_ROOT / "results" / "image_to_3D"
DEFAULT_DATA_ROOT = EVAL_ROOT / "data"

VIEWS = ["Image_005.png", "Image_015.png", "Image_025.png", "Image_035.png"]

ENCODERS = {
    "siglip2": {
        "model_id": "google/siglip2-so400m-patch16-naflex",
        "kind": "siglip",  # uses model.get_image_features
        "processor": "auto_processor",
    },
    "dinov2": {
        "model_id": "facebook/dinov2-large",
        "kind": "pooler",  # uses model(**inputs).pooler_output
        "processor": "image_processor",
    },
    "dinov3": {
        "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "kind": "pooler",
        "processor": "image_processor",
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--model", required=True, help="Inference model dir name under <results-root>/")
    p.add_argument("--encoder", choices=list(ENCODERS), default="siglip2")
    p.add_argument("--encoder-id", default=None, help="Override the HF model id for the chosen encoder.")
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--instances", nargs="*", default=None)
    return p.parse_args()


def load_encoder(encoder_name, override_id, device):
    spec = dict(ENCODERS[encoder_name])
    if override_id:
        spec["model_id"] = override_id
    print(f"Loading encoder: {encoder_name}  ->  {spec['model_id']}")
    t0 = time.time()

    # Try online first, then fall back to local cache (handles gated repos
    # like facebook/dinov3-vitl16-pretrain-lvd1689m where the weights are
    # cached but processor_config.json may not be reachable).
    def _load(cls, mid):
        try:
            return cls.from_pretrained(mid)
        except OSError:
            return cls.from_pretrained(mid, local_files_only=True)

    if spec["processor"] == "auto_processor":
        processor = _load(AutoProcessor, spec["model_id"])
    else:
        processor = _load(AutoImageProcessor, spec["model_id"])
    model = _load(AutoModel, spec["model_id"]).to(device).eval()
    print(f"  loaded in {time.time() - t0:.1f}s")
    return model, processor, spec


def encode_images(model, processor, spec, paths, batch_size, device):
    """Return (N, D) L2-normalized image embeddings."""
    embeds = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            if spec["kind"] == "siglip":
                out = model.get_image_features(**inputs)
                feats = out if isinstance(out, torch.Tensor) else out.pooler_output
            else:  # pooler (DINOv2 / DINOv3)
                feats = model(**inputs).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeds.append(feats.cpu())
    return torch.cat(embeds, 0) if embeds else torch.zeros(0)


def main():
    args = parse_args()
    model_dir = args.results_root / args.model
    if not model_dir.exists():
        raise SystemExit(f"No such model dir: {model_dir}")

    instances = sorted(d for d in model_dir.iterdir() if d.is_dir() and not d.name.startswith("_"))
    if args.instances:
        keep = set(args.instances)
        instances = [d for d in instances if d.name in keep]

    # Per-instance: collect rendered + reference paths, view-aligned.
    rows = []
    for inst in instances:
        renders_dir = inst / "renders"
        ref_dir = args.data_root / inst.name / "images"

        rendered = [(renders_dir / v) if (renders_dir / v).exists() else None for v in VIEWS]
        reference = [(ref_dir / v) if (ref_dir / v).exists() else None for v in VIEWS]
        rows.append(
            {
                "instance": inst.name,
                "rendered": rendered,
                "reference": reference,
            }
        )
    if not rows:
        raise SystemExit("No instances found.")

    # Flat list of all images to encode, with back-pointers.
    flat_paths = []
    flat_idx = []  # (row_idx, slot, kind)  kind in {"r","g"}
    for ri, r in enumerate(rows):
        for slot, p in enumerate(r["rendered"]):
            if p is not None:
                flat_paths.append(p)
                flat_idx.append((ri, slot, "r"))
        for slot, p in enumerate(r["reference"]):
            if p is not None:
                flat_paths.append(p)
                flat_idx.append((ri, slot, "g"))

    model, processor, spec = load_encoder(args.encoder, args.encoder_id, args.device)
    print(f"Encoding {len(flat_paths)} images (rendered + reference)…")
    t0 = time.time()
    embeds = encode_images(model, processor, spec, flat_paths, args.batch_size, args.device)
    print(f"  done in {time.time() - t0:.1f}s, shape={tuple(embeds.shape)}")

    # Slot the encoded vectors back into per-instance, per-view tensors.
    n = len(rows)
    rend_embed = [[None] * 4 for _ in range(n)]
    ref_embed = [[None] * 4 for _ in range(n)]
    for k, (ri, slot, kind) in enumerate(flat_idx):
        if kind == "r":
            rend_embed[ri][slot] = embeds[k]
        else:
            ref_embed[ri][slot] = embeds[k]

    # Per instance: build full 4x4 cosine matrix (rendered_i vs reference_j),
    # then derive view-paired (diagonal) and best-assignment (Hungarian).
    per_instance = []
    for ri, r in enumerate(rows):
        rend = rend_embed[ri]
        ref = ref_embed[ri]

        cos_matrix = [[None] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                if rend[i] is not None and ref[j] is not None:
                    cos_matrix[i][j] = float((rend[i] * ref[j]).sum())

        # View-paired (diagonal of the matrix).
        per_view = [cos_matrix[v][v] for v in range(4)]
        paired = [s for s in per_view if s is not None]
        score_paired = float(sum(paired) / len(paired)) if paired else None

        # Best-assignment via Hungarian on the dense submatrix of valid views.
        valid_rend = [i for i in range(4) if rend[i] is not None]
        valid_ref = [j for j in range(4) if ref[j] is not None]
        if valid_rend and valid_ref:
            sub = np.array([[cos_matrix[i][j] for j in valid_ref] for i in valid_rend])
            row_ind, col_ind = linear_sum_assignment(-sub)
            matched = [float(sub[i, j]) for i, j in zip(row_ind, col_ind)]
            score_assigned = sum(matched) / len(matched)
            assignment = [None] * 4  # rendered_v -> ref_v
            for il, jl in zip(row_ind, col_ind):
                assignment[valid_rend[il]] = valid_ref[jl]
        else:
            score_assigned = None
            assignment = [None] * 4

        per_instance.append(
            {
                "instance": r["instance"],
                "n_paired_views": len(paired),
                "score_mean": score_paired,  # view-paired (diagonal)
                "score_assigned": score_assigned,  # Hungarian best-assignment
                "per_view": per_view,
                "assignment": assignment,
                "cos_matrix": cos_matrix,  # 4x4, None for missing views
            }
        )

    # Aggregates over both metrics.
    full = [pi for pi in per_instance if pi["n_paired_views"] == 4]

    def agg(field):
        cond = float(sum(pi[field] for pi in full) / len(full)) if full else None
        pen = float(sum((pi[field] or 0.0) for pi in per_instance) / n)
        return {"conditional_mean": cond, "penalized_mean": pen}

    summary = {
        "model": args.model,
        "encoder": args.encoder,
        "encoder_id": spec["model_id"],
        "n_instances": n,
        "n_full_paired": len(full),
        "view_paired": agg("score_mean"),
        "best_assignment": agg("score_assigned"),
        "per_instance": per_instance,
    }

    out_dir = model_dir / "_metrics"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"image_similarity_{args.encoder}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else "  --  "

    vp, ba = summary["view_paired"], summary["best_assignment"]
    print()
    print(f"=== {args.encoder} image-image similarity — {args.model} ===")
    print(f"  Instances:                 {n}")
    print(f"  With all 4 paired views:   {len(full)}")
    print(f"  view-paired       conditional={fmt(vp['conditional_mean'])}  penalized={fmt(vp['penalized_mean'])}")
    print(f"  best-assignment   conditional={fmt(ba['conditional_mean'])}  penalized={fmt(ba['penalized_mean'])}")
    delta_c = (
        (ba["conditional_mean"] - vp["conditional_mean"])
        if (ba["conditional_mean"] and vp["conditional_mean"])
        else None
    )
    delta_p = ba["penalized_mean"] - vp["penalized_mean"]
    print(f"  delta (best - paired)         conditional={fmt(delta_c)}  penalized={fmt(delta_p)}")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
