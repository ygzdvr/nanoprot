#!/usr/bin/env python
"""
Aggregate the nanoprot release-suite results into a table + CSV.

Scans a release directory for completed cells (`meta.step == num_iterations`),
reads each checkpoint's self-describing metadata, and groups by (arch, scale)
to report the headline metric as mean ± std across seeds — the same numbers the
model cards and the scaling-curve figure use. Safe to run mid-training: it
simply reports whatever has finished so far.

Usage:
  python -m scripts.aggregate_results --release-root $NANOPROT_BASE_DIR/release
  python -m scripts.aggregate_results --release-root DIR --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NAME_RE = re.compile(r"^nanoprot-(?P<arch>gpt2|esm2|mamba)-(?P<scale>XS|S|M|L)-s(?P<seed>\d+)$")
_SCALE_ORDER = {"XS": 0, "S": 1, "M": 2, "L": 3}


def _final_meta(ckpt_dir: Path) -> Optional[Dict[str, Any]]:
    metas = sorted(ckpt_dir.glob("meta_*.json"))
    best = None
    for m in metas:
        try:
            d = json.loads(m.read_text())
        except Exception:
            continue
        if best is None or int(d.get("step", -1)) > int(best.get("step", -1)):
            best = d
    return best


def collect_cells(release_root: Path, *, only_done: bool = True) -> List[Dict[str, Any]]:
    """One record per (arch, scale, seed) checkpoint found under release_root."""
    cells: List[Dict[str, Any]] = []
    for d in sorted(release_root.glob("nanoprot-*-s*")):
        if not d.is_dir():
            continue
        m = NAME_RE.match(d.name)
        if not m:
            continue
        meta = _final_meta(d)
        if meta is None:
            continue
        done = meta.get("step") == meta.get("num_iterations")
        if only_done and not done:
            continue
        train = (meta.get("config") or {}).get("training", {})
        cells.append({
            "arch": m["arch"], "scale": m["scale"], "seed": int(m["seed"]),
            "objective": train.get("objective"),
            "n_params": meta.get("n_params"),
            "total_residues": meta.get("total_residues"),
            "total_flops": meta.get("total_flops"),
            "train_time_sec": meta.get("train_time_sec"),
            "val_bpr": meta.get("last_val_bpr"),
            "best_val_bpr": meta.get("best_val_bpr"),
            "step": meta.get("step"), "num_iterations": meta.get("num_iterations"),
            "done": done,
        })
    return cells


def _mean_std(xs: List[float]) -> tuple[Optional[float], Optional[float]]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return None, None
    mu = sum(xs) / len(xs)
    if len(xs) == 1:
        return mu, 0.0
    return mu, math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def aggregate(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group cells by (arch, scale) -> mean +/- std bpr across seeds."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for c in cells:
        groups.setdefault((c["arch"], c["scale"]), []).append(c)
    out: List[Dict[str, Any]] = []
    for (arch, scale), gs in groups.items():
        mu, sd = _mean_std([g["val_bpr"] for g in gs])
        out.append({
            "arch": arch, "scale": scale,
            "objective": gs[0]["objective"],
            "n_params": gs[0]["n_params"],
            "total_residues": gs[0]["total_residues"],
            "total_flops": gs[0]["total_flops"],
            "n_seeds": len(gs),
            "val_bpr_mean": mu, "val_bpr_std": sd,
            "seeds_done": sorted(g["seed"] for g in gs),
        })
    out.sort(key=lambda r: (r["arch"], _SCALE_ORDER.get(r["scale"], 9)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--csv", type=Path, default=None, help="Also write per-cell rows here.")
    ap.add_argument("--all", action="store_true", help="Include partial (not-yet-done) cells.")
    args = ap.parse_args()

    cells = collect_cells(args.release_root, only_done=not args.all)
    if not cells:
        print(f"No completed cells under {args.release_root} yet.")
        return 0
    agg = aggregate(cells)

    # AR models (gpt2/mamba) share bits-per-residue; esm2 is a different metric.
    print(f"\n  nanoprot release results  ({len(cells)} cells done)\n")
    print(f"  {'arch':6} {'scale':5} {'params':>9} {'residues':>9} "
          f"{'metric (mean±std)':>20} {'seeds':>7}")
    print("  " + "-" * 64)
    for r in agg:
        p = r["n_params"]
        params = f"{p/1e6:.0f}M" if p and p < 1e9 else (f"{p/1e9:.2f}B" if p else "?")
        res = r["total_residues"]
        residues = f"{res/1e9:.2f}B" if res else "?"
        mu, sd = r["val_bpr_mean"], r["val_bpr_std"]
        metric = f"{mu:.4f} ± {sd:.4f}" if mu is not None else "n/a"
        lbl = "bpr" if r["objective"] == "ar" else "mCE"
        print(f"  {r['arch']:6} {r['scale']:5} {params:>9} {residues:>9} "
              f"{metric:>16} {lbl:>3} {r['n_seeds']}/3")
    print()
    print("  bpr = bits-per-residue (AR: gpt2, mamba — directly comparable).")
    print("  mCE = masked cross-entropy bits (esm2 MLM — NOT comparable to bpr).\n")

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
            w.writeheader()
            w.writerows(cells)
        print(f"  wrote per-cell CSV: {args.csv}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
