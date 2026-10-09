#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3DCodeBench; see third_party/NOTICE.md for provenance.
"""Executability metric — aggregates per-instance render_log.json files.

Definition (per user spec): a generated script "executes" iff
    (1) Blender 5.0 runs the script via `--background --python`
        without raising an exception, AND
    (2) at least one mesh object exists in the scene afterwards.

This metric reads `results/<model>/<inst>/renders/render_log.json`
files (produced by `eval/utils/render.py`) and reports:

    - Pass rate (status == OK in the render log, which already encodes
      both criteria above plus successful 4-view render)
    - Breakdown by failure stage (ERR_EXEC, ERR_NO_MESH, ERR_RENDER,
      ERR_TIMEOUT, ERR_NOLOG, MISSING)
    - Top recurring error fingerprints (first line of `error` field)

Output: prints a summary to stdout AND writes a JSON file at
    `results/<model>/_metrics/executability.json`

Usage:
    python eval/metrics/executability.py --model gemini-3-flash-preview
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = EVAL_ROOT / "results"


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--model", required=True)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--top-errors", type=int, default=8)
    return p.parse_args()


_PY_FRAME_RE = re.compile(r'File ".*?\.py", line \d+')


def _err_fingerprint(err: str) -> str:
    """Compact error signature: first non-traceback line."""
    if not err:
        return ""
    for line in err.splitlines():
        line = line.strip()
        if line and not line.startswith("Traceback") and not line.startswith("File "):
            # Truncate noisy bits
            line = _PY_FRAME_RE.sub("File <...>", line)
            return line[:200]
    return err.splitlines()[0][:200]


def main():
    args = parse_args()
    model_dir = args.results_root / args.model
    if not model_dir.exists():
        raise SystemExit(f"No such model dir: {model_dir}")

    instances = sorted(d for d in model_dir.iterdir() if d.is_dir() and not d.name.startswith("_"))

    rows = []
    for inst in instances:
        log = inst / "renders" / "render_log.json"
        if not log.exists():
            rows.append(
                {
                    "instance": inst.name,
                    "status": "MISSING",
                    "error": "no render_log.json",
                    "n_meshes": 0,
                    "n_views_rendered": 0,
                    "latency_s": 0,
                }
            )
            continue
        try:
            rec = json.loads(log.read_text())
        except Exception as e:
            rows.append(
                {
                    "instance": inst.name,
                    "status": "BAD_LOG",
                    "error": f"{type(e).__name__}: {e}",
                    "n_meshes": 0,
                    "n_views_rendered": 0,
                    "latency_s": 0,
                }
            )
            continue
        rows.append(
            {
                "instance": inst.name,
                "status": rec.get("status"),
                "error": rec.get("error"),
                "n_meshes": rec.get("n_meshes", 0) or 0,
                "n_views_rendered": rec.get("n_views_rendered", 0) or 0,
                "latency_s": rec.get("latency_s", 0) or 0,
            }
        )

    n = len(rows)
    by_status = Counter(r["status"] for r in rows)
    n_ok = by_status.get("OK", 0)
    pass_rate = n_ok / n if n else 0.0

    # Top error fingerprints
    err_fps = Counter(_err_fingerprint(r["error"]) for r in rows if r["status"] != "OK" and r["error"])

    # Aggregate
    summary = {
        "model": args.model,
        "n_total": n,
        "n_pass": n_ok,
        "pass_rate": round(pass_rate, 4),
        "by_status": dict(by_status.most_common()),
        "top_errors": [{"count": c, "fingerprint": fp} for fp, c in err_fps.most_common(args.top_errors)],
        "failed_instances": [r["instance"] for r in rows if r["status"] != "OK"],
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / max(n, 1), 2),
    }

    # Write
    out_dir = model_dir / "_metrics"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "executability.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # Print
    print(f"=== Executability — {args.model} ===")
    print(f"Total instances:  {n}")
    print(f"Pass:             {n_ok} / {n}  ({100 * pass_rate:.1f}%)")
    print()
    print("Status breakdown:")
    for s, c in by_status.most_common():
        print(f"  {s:<14} {c:>4}  ({100 * c / n:.1f}%)")
    if err_fps:
        print()
        print("Top error fingerprints:")
        for fp, c in err_fps.most_common(args.top_errors):
            print(f"  ×{c:>3}  {fp}")
    print()
    print(f"Mean latency: {summary['mean_latency_s']}s")
    print(f"Saved:        {out_path}")


if __name__ == "__main__":
    main()
