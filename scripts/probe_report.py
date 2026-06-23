#!/usr/bin/env python
"""
Tabular report for the cross-architecture probing result (Pillar 3) — the numbers the
"lock the result" step needs and ``plot_probes`` does not emit:

  1. Per (task x arch x scale): absolute probe score, the random-init baseline, and
     learned-minus-baseline, each as mean +/- std over seeds.
  2. The PAIRED gpt2-mamba delta: per (task x scale), Delta = (gpt2 learned-baseline) -
     (mamba learned-baseline) computed WITHIN each seed then averaged, with a seed-level
     sign check (did gpt2 win in every seed?). Pairing within seed removes seed variance
     and is the rigorous "attention out-scales the tested Mamba" statement.
  3. Per-class F1 (e.g. helix / strand / coil) at the best layer, from the sidecars.

A "task" is a (concept, source) pair — e.g. ss3/netsurfp, ss8/netsurfp, rsa/netsurfp,
ss3/swissprot — because all three NetSurfP concepts share source="netsurfp" and must NOT
be merged (different class sets, and rsa uses r2 not macro_f1).

Reads the ``run_probes`` CSVs (one per source) + per-layer JSON sidecars. Emits a
markdown report (paste-ready for RESULTS.md) and a flat CSV of the aggregated stats.
Safe on partial results — it reports only the cells that are present.

Usage:
  python -m scripts.probe_report --results-dir $NANOPROT_BASE_DIR/probe_results \
      --out docs/probe_report.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

SCALES = ["XS", "S", "M", "L"]
ARCHES = ["gpt2", "mamba", "esm2"]
SRC_ORDER = {"netsurfp": 0, "swissprot": 1, "dssp": 2}
CONCEPT_ORDER = {"ss3": 0, "ss8": 1, "rsa": 2}

# A task is the (concept, source) pair; this is the real comparison axis.
Task = Tuple[str, str]


def _task(r: dict) -> Task:
    return (r["concept"], r["source"])


def _task_label(t: Task) -> str:
    return f"{t[0]} / {t[1]}"


def _task_sort(t: Task):
    return (CONCEPT_ORDER.get(t[0], 9), SRC_ORDER.get(t[1], 9), t)


def _mean_sd(xs: List[float]) -> Tuple[float, float, int]:
    if not xs:
        return (float("nan"), float("nan"), 0)
    mu = sum(xs) / len(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return (mu, sd, len(xs))


def load_rows(results_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for f in sorted(glob.glob(str(results_dir / "*.csv"))):
        with open(f) as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def load_per_class(results_dir: Path) -> Tuple[Dict[tuple, List[float]], Dict[Task, List[str]]]:
    """(concept, source, arch, scale, seed) -> learned per_class_f1 @ best layer; + task->class_names."""
    pc: Dict[tuple, List[float]] = {}
    names: Dict[Task, List[str]] = {}
    for p in sorted(glob.glob(str(results_dir / "*.json"))):
        try:
            d = json.loads(Path(p).read_text())
        except Exception:
            continue
        row, best = d.get("row", {}), d.get("best", {})
        lt = best.get("learned_test", {}) if isinstance(best, dict) else {}
        if isinstance(lt, dict) and "per_class_f1" in lt:
            t = _task(row)
            pc[(t[0], t[1], row["arch"], row["scale"], row["seed"])] = lt["per_class_f1"]
            if d.get("class_names"):
                names[t] = d["class_names"]
    return pc, names


def summary(rows: List[dict]):
    """(task, arch, scale) -> {col: [values over seeds]}, plus task->metric."""
    g: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    metric: Dict[Task, str] = {}
    for r in rows:
        k = (_task(r), r["arch"], r["scale"])
        for col in ("learned_test", "baseline_test", "learned_minus_baseline"):
            if r.get(col):
                g[k][col].append(float(r[col]))
        metric[_task(r)] = r.get("metric", "?")
    return g, metric


def paired_gpt2_mamba(rows: List[dict]) -> Dict[tuple, tuple]:
    """(task, scale) -> (mean, sd, n_seeds, gpt2_won_every_seed). Delta paired within seed."""
    by: Dict[tuple, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["arch"] in ("gpt2", "mamba") and r.get("learned_minus_baseline"):
            by[(_task(r), r["scale"])][r["seed"]][r["arch"]] = float(r["learned_minus_baseline"])
    out: Dict[tuple, tuple] = {}
    for k, seeds in by.items():
        diffs = [v["gpt2"] - v["mamba"] for v in seeds.values() if "gpt2" in v and "mamba" in v]
        if diffs:
            mu, sd, n = _mean_sd(diffs)
            out[k] = (mu, sd, n, all(d > 0 for d in diffs))
    return out


def _tasks(rows: List[dict]) -> List[Task]:
    return sorted({_task(r) for r in rows}, key=_task_sort)


def render(rows: List[dict], pc: Dict[tuple, List[float]], names: Dict[Task, List[str]]) -> str:
    g, metric = summary(rows)
    paired = paired_gpt2_mamba(rows)
    L: List[str] = ["# Cross-architecture probing report (Pillar 3)", ""]

    # 1. absolute + learned-minus-baseline, per task
    for t in _tasks(rows):
        L += [f"## {_task_label(t)}  (metric = {metric.get(t, '?')})", "",
              "| arch | scale | learned | baseline | learned−baseline | n |",
              "|------|-------|---------|----------|------------------|---|"]
        for arch in ARCHES:
            for sc in SCALES:
                cell = g.get((t, arch, sc))
                if not cell:
                    continue
                lm, ls, _ = _mean_sd(cell.get("learned_test", []))
                bm, bs, _ = _mean_sd(cell.get("baseline_test", []))
                dm, ds, n = _mean_sd(cell.get("learned_minus_baseline", []))
                L.append(f"| {arch} | {sc} | {lm:.3f}±{ls:.3f} | {bm:.3f}±{bs:.3f} | "
                         f"**{dm:+.3f}**±{ds:.3f} | {n} |")
        L.append("")

    # 2. paired gpt2 - mamba (within-seed)
    L += ["## Paired gpt2 − mamba  (Δ-of-Δ within seed; >0 ⇒ gpt2 more decodable)", "",
          "| task | scale | mean Δ | sd | n | gpt2 won every seed |",
          "|------|-------|--------|----|----|---------------------|"]
    for t in _tasks(rows):
        for sc in SCALES:
            v = paired.get((t, sc))
            if not v:
                continue
            mu, sd, n, allwin = v
            L.append(f"| {_task_label(t)} | {sc} | **{mu:+.3f}** | {sd:.3f} | {n} | "
                     f"{'✓' if allwin else '—'} |")
    L.append("")

    # 3. per-class F1 at best layer (classification tasks w/ sidecars)
    for t in _tasks(rows):
        cls = names.get(t)
        if not cls:
            continue
        L += [f"## Per-class F1 @ best layer — {_task_label(t)}  ({' / '.join(cls)})", "",
              "| arch | scale | " + " | ".join(cls) + " |",
              "|------|-------|" + "|".join(["-----"] * len(cls)) + "|"]
        for arch in ARCHES:
            for sc in SCALES:
                vals = [pc[k] for k in pc if k[:4] == (t[0], t[1], arch, sc)]
                if not vals:
                    continue
                means = [sum(c) / len(c) for c in zip(*vals)]   # mean over seeds, per class
                L.append(f"| {arch} | {sc} | " + " | ".join(f"{x:.3f}" for x in means) + " |")
        L.append("")
    return "\n".join(L)


def write_csv(rows: List[dict], out_csv: Path) -> None:
    g, metric = summary(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["concept", "source", "arch", "scale", "metric", "learned_mean",
                    "learned_sd", "baseline_mean", "baseline_sd", "delta_mean", "delta_sd",
                    "n_seeds"])
        for (t, arch, sc), cell in sorted(g.items(), key=lambda kv: (_task_sort(kv[0][0]),)):
            lm, ls, _ = _mean_sd(cell.get("learned_test", []))
            bm, bs, _ = _mean_sd(cell.get("baseline_test", []))
            dm, ds, n = _mean_sd(cell.get("learned_minus_baseline", []))
            w.writerow([t[0], t[1], arch, sc, metric.get(t, "?"),
                        f"{lm:.6f}", f"{ls:.6f}", f"{bm:.6f}", f"{bs:.6f}",
                        f"{dm:.6f}", f"{ds:.6f}", n])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Dir with run_probes <source>.csv files + sidecar JSONs.")
    ap.add_argument("--out", type=Path, default=Path("docs/probe_report.md"))
    args = ap.parse_args()

    rows = load_rows(args.results_dir)
    if not rows:
        print(f"No result CSVs in {args.results_dir}")
        return 0
    pc, names = load_per_class(args.results_dir)
    report = render(rows, pc, names)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n")
    write_csv(rows, args.out.with_suffix(".csv"))
    n_cells = len({(_task(r), r["arch"], r["scale"], r["seed"]) for r in rows})
    print(f"  wrote {args.out} + {args.out.with_suffix('.csv')}  "
          f"({len(rows)} rows, {n_cells} cells, {len(_tasks(rows))} tasks, "
          f"{len(pc)} per-class sidecars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
