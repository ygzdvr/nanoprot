#!/usr/bin/env python
"""
Emergence-metric analysis for the trajectory experiment (Pillar 2).

Turns the per-checkpoint probe curves (from ``run_trajectory_probes``) into the
operational emergence metrics the PI review asks for, rather than just "score goes up":

  * t_onset : first checkpoint where Delta = learned - baseline exceeds a PRE-REGISTERED
              epsilon (0.02 macro-F1 / 0.01 R^2 by default).
  * t50/t90 : checkpoint reaching 50% / 90% of the FINAL Delta (acquisition speed).
  * AUEC    : area under the emergence curve over log(step) — distinguishes two models
              with the same final Delta but different acquisition speed.
  * layer migration : whether the layer of peak decodability moves deeper over training.

Aggregates over seeds and contrasts gpt2 vs mamba per (scale, task): does secondary
structure become decodable EARLIER in attention than in the SSM? Reads the flat
``trajectory_results/<task>.csv`` (best-by-val per checkpoint) + the layer x checkpoint
JSON sidecars. Emits a markdown report + a metrics CSV. (Matched-loss vs validation loss
is drawn by ``plot_emergence.py``, which joins these curves with history.jsonl.)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ARCHES = ["gpt2", "mamba", "esm2"]
SCALES = ["XS", "S", "M", "L"]
DEFAULT_EPS = {"macro_f1": 0.02, "accuracy": 0.02, "r2": 0.01, "spearman": 0.02}


# ---------------------------------------------------------------------------
# Pure emergence metrics (unit-tested)
# ---------------------------------------------------------------------------

def emergence_times(steps: List[int], deltas: List[float], *, eps: float,
                    fracs: Tuple[float, ...] = (0.5, 0.9)) -> Dict[str, Optional[float]]:
    """Onset (first Delta > eps) and the steps reaching given fractions of the FINAL Delta.

    Returns step numbers (or ``None`` if never reached). ``delta_final`` is the last point.
    """
    pairs = sorted((s, d) for s, d in zip(steps, deltas))
    if not pairs:
        return {}
    s_sorted = [s for s, _ in pairs]
    d_sorted = [d for _, d in pairs]
    final = d_sorted[-1]

    def first_at(thresh: float) -> Optional[float]:
        for s, d in zip(s_sorted, d_sorted):
            if d >= thresh:
                return s
        return None

    out: Dict[str, Optional[float]] = {"delta_final": final, "t_onset": first_at(eps)}
    for f in fracs:
        out[f"t{int(round(f * 100))}"] = first_at(f * final) if final > 0 else None
    return out


def auec(steps: List[int], deltas: List[float]) -> float:
    """Area under the emergence curve over log(step) (trapezoidal; ignores step<=0)."""
    pairs = sorted((s, d) for s, d in zip(steps, deltas) if s > 0)
    if len(pairs) < 2:
        return float("nan")
    area = 0.0
    for (s0, d0), (s1, d1) in zip(pairs, pairs[1:]):
        area += 0.5 * (d0 + d1) * (math.log(s1) - math.log(s0))
    return area


def layer_star_over_time(sidecar: dict) -> List[Tuple[int, float]]:
    """[(step, rel_depth of the peak-Delta layer)] — does peak decodability move deeper?"""
    out: List[Tuple[int, float]] = []
    for c in sidecar.get("checkpoints", []):
        pl = c.get("per_layer") or []
        if pl:
            best = max(pl, key=lambda l: (l.get("delta") if l.get("delta") is not None else -9))
            out.append((c["step"], best.get("rel_depth")))
    return out


# ---------------------------------------------------------------------------
# IO + aggregation
# ---------------------------------------------------------------------------

def load_curves(results_dir: Path) -> Dict[tuple, List[Tuple[int, float]]]:
    """(concept, source, arch, scale, seed) -> sorted [(step, delta)] from the flat CSVs."""
    curves: Dict[tuple, List[Tuple[int, float]]] = defaultdict(list)
    metric: Dict[tuple, str] = {}
    for f in sorted(glob.glob(str(results_dir / "*.csv"))):
        reader = csv.DictReader(open(f))
        if not reader.fieldnames or "concept" not in reader.fieldnames:
            continue   # skip non-probe CSVs in the same dir (e.g. val_loss.csv)
        for r in reader:
            key = (r["concept"], r["source"], r["arch"], r["scale"], r["seed"])
            curves[key].append((int(r["step"]), float(r["delta"])))
            metric[(r["concept"], r["source"])] = r.get("metric", "macro_f1")
    for k in curves:
        curves[k].sort()
    return curves, metric


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def render(curves: Dict[tuple, List[Tuple[int, float]]], metric: Dict[tuple, str]) -> str:
    # per-cell metrics
    rows: Dict[tuple, dict] = {}
    for (concept, source, arch, scale, seed), curve in curves.items():
        steps = [s for s, _ in curve]
        deltas = [d for _, d in curve]
        eps = DEFAULT_EPS.get(metric.get((concept, source), "macro_f1"), 0.02)
        et = emergence_times(steps, deltas, eps=eps)
        rows[(concept, source, arch, scale, seed)] = {**et, "auec": auec(steps, deltas)}

    tasks = sorted({(c, s) for (c, s, *_rest) in curves}, key=lambda t: t)
    scales = sorted({k[3] for k in curves}, key=lambda s: SCALES.index(s) if s in SCALES else 9)
    L: List[str] = ["# Trajectory emergence report (Pillar 2)", "",
                    "Δ = learned − fixed-baseline; t_* in training steps (seed-mean). "
                    "AUEC = ∫Δ d·log(step). gpt2−mamba<0 in t_onset ⇒ gpt2 emerges earlier.", ""]
    for (concept, source) in tasks:
        L += [f"## {concept} / {source}  (metric={metric.get((concept, source), '?')})", "",
              "| arch | scale | t_onset | t50 | t90 | Δ_final | AUEC |",
              "|------|-------|---------|-----|-----|---------|------|"]
        for arch in ARCHES:
            for scale in scales:
                seeds = [rows[k] for k in rows if k[:4] == (concept, source, arch, scale)]
                if not seeds:
                    continue
                def mm(field):
                    v = _mean([s.get(field) for s in seeds])
                    return f"{v:.0f}" if (v is not None and field.startswith("t")) else (
                        f"{v:+.3f}" if v is not None else "—")
                L.append(f"| {arch} | {scale} | {mm('t_onset')} | {mm('t50')} | {mm('t90')} | "
                         f"{mm('delta_final')} | {mm('auec')} |")
        # gpt2 vs mamba onset contrast
        for scale in scales:
            g = _mean([rows[k]["t_onset"] for k in rows
                       if k[:4] == (concept, source, "gpt2", scale) and rows[k]["t_onset"] is not None])
            m = _mean([rows[k]["t_onset"] for k in rows
                       if k[:4] == (concept, source, "mamba", scale) and rows[k]["t_onset"] is not None])
            if g is not None and m is not None:
                who = "gpt2 earlier" if g < m else ("mamba earlier" if m < g else "tie")
                L.append(f"|  ↳ gpt2−mamba onset @ {scale} | | {g - m:+.0f} steps ({who}) | | | | |")
        L.append("")
    return "\n".join(L)


def write_csv(curves, metric, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["concept", "source", "arch", "scale", "seed", "t_onset", "t50", "t90",
                    "delta_final", "auec"])
        for (concept, source, arch, scale, seed), curve in sorted(curves.items()):
            steps = [s for s, _ in curve]; deltas = [d for _, d in curve]
            eps = DEFAULT_EPS.get(metric.get((concept, source), "macro_f1"), 0.02)
            et = emergence_times(steps, deltas, eps=eps)
            w.writerow([concept, source, arch, scale, seed, et.get("t_onset"), et.get("t50"),
                        et.get("t90"), f"{et.get('delta_final', float('nan')):.6f}",
                        f"{auec(steps, deltas):.6f}"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="trajectory_results dir (run_trajectory_probes CSVs + sidecars).")
    ap.add_argument("--out", type=Path, default=Path("docs/emergence_report.md"))
    args = ap.parse_args()

    curves, metric = load_curves(args.results_dir)
    if not curves:
        print(f"No trajectory CSVs in {args.results_dir}")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(curves, metric) + "\n")
    write_csv(curves, metric, args.out.with_suffix(".csv"))
    n_cells = len(curves)
    print(f"  wrote {args.out} + {args.out.with_suffix('.csv')}  ({n_cells} trajectory cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
