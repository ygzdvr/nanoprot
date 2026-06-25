#!/usr/bin/env python
"""
Figures for the trajectory-emergence experiment (Pillar 2).

Four families (PI review S8), each safe to draw on partial results:

  1. Delta(t) vs training step    — emergence curves, seed mean +/- range, per task.
  2. Delta(t) vs validation loss  — the MATCHED-LOSS view (joins history.jsonl): is gpt2's
                                    edge mere optimization speed, or more decodable biology
                                    at the SAME language-modeling competence?
  3. layer x checkpoint heatmaps  — Delta_layer(t), the Pythia-style developmental view.
  4. emergence-time summary       — t_onset / t50 / t90 as gpt2-vs-mamba dumbbells.

Reads ``trajectory_results/<task>.csv`` + ``traj_*.json`` sidecars from
``run_trajectory_probes``, and (for family 2) ``history.jsonl`` under
``release_traj/<model>/``. Aesthetic follows the probe figures: sans-serif, no rainbow
(colour = arch, line style = scale), de-spined, panel letters.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analyze_emergence import (  # noqa: E402
    DEFAULT_EPS, auec, emergence_times, load_curves,
)

_COLOR = {"gpt2": "#1f4e79", "mamba": "#b5471f", "esm2": "#3f7f3f"}
_LS = {"XS": ":", "S": "--", "M": "-", "L": "-."}
_ARCHES = ["gpt2", "mamba", "esm2"]
_SCALES = ["XS", "S", "M", "L"]
_SRC_ORDER = {"netsurfp": 0, "swissprot": 1, "dssp": 2}
_CONCEPT_ORDER = {"ss3": 0, "ss8": 1, "rsa": 2}


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "figure.dpi": 150,
    })


def _tasks(curves) -> List[Tuple[str, str]]:
    ts = {(c, s) for (c, s, *_r) in curves}
    return sorted(ts, key=lambda t: (_CONCEPT_ORDER.get(t[0], 9), _SRC_ORDER.get(t[1], 9)))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _by_arch_scale(curves, concept, source) -> Dict[tuple, Dict[int, List[float]]]:
    agg: Dict[tuple, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (c, s, arch, scale, seed), curve in curves.items():
        if (c, s) != (concept, source):
            continue
        for step, delta in curve:
            agg[(arch, scale)][step].append(delta)
    return agg


# ---------------------------------------------------------------------------
# Loaders for the sidecars + training history (family 2 + 3)
# ---------------------------------------------------------------------------

def load_sidecars(results_dir: Path) -> List[dict]:
    out = []
    for p in sorted(glob.glob(str(results_dir / "traj_*.json"))):
        try:
            out.append(json.loads(Path(p).read_text()))
        except Exception:
            continue
    return out


def load_val_loss_csv(results_dir: Path) -> Dict[tuple, float]:
    """(arch, scale, seed, step) -> val_loss, from posthoc_val_loss_eval's val_loss.csv.

    Exact per-checkpoint val_loss (the post-hoc eval scores every saved step), so the
    matched-loss panel joins by step with no interpolation.
    """
    out: Dict[tuple, float] = {}
    p = Path(results_dir) / "val_loss.csv"
    if not p.exists():
        return out
    for r in csv.DictReader(open(p)):
        out[(r["arch"], r["scale"], r["seed"], int(r["step"]))] = float(r["val_loss"])
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_vs_step(curves, out: Path) -> None:
    """Family 1: Delta vs training step, one panel per task; colour=arch, style=scale."""
    _style()
    tasks = _tasks(curves)
    if not tasks:
        return
    ncols = min(len(tasks), 3)
    nrows = -(-len(tasks) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows),
                             constrained_layout=True, squeeze=False)
    flat = [a for row in axes for a in row]
    for i, (concept, source) in enumerate(tasks):
        ax = flat[i]
        for (arch, scale), bystep in sorted(_by_arch_scale(curves, concept, source).items()):
            steps = sorted(bystep)
            mean = [_mean(bystep[s]) for s in steps]
            lo = [min(bystep[s]) for s in steps]
            hi = [max(bystep[s]) for s in steps]
            ax.plot(steps, mean, color=_COLOR.get(arch, "0.3"), ls=_LS.get(scale, "-"),
                    lw=1.4, label=f"{arch}-{scale}")
            ax.fill_between(steps, lo, hi, color=_COLOR.get(arch, "0.3"), alpha=0.13, lw=0)
        ax.axhline(0, color="0.6", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("training step")
        if i == 0:
            ax.set_ylabel("Δ  (learned − baseline)")
        ax.set_title(f"{concept} / {source}")
        ax.grid(True, color="0.93", lw=0.6)
        ax.text(-0.16, 1.04, "abcdefgh"[i], transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.legend(frameon=False, fontsize=6)
    for ax in flat[len(tasks):]:
        ax.set_visible(False)
    _save(fig, out)


def fig_vs_valloss(curves, valloss: Dict[tuple, float], out: Path) -> None:
    """Family 2 (matched-loss): Delta vs validation loss, exact per-checkpoint join.

    The decisive figure: at the SAME validation loss, does gpt2 expose more decodable
    structure than mamba (representational inductive bias), or do the curves overlap
    (a pure optimization-speed difference)?
    """
    if not valloss:
        return
    _style()
    tasks = _tasks(curves)
    drew = False
    ncols = min(len(tasks), 3)
    nrows = -(-len(tasks) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows),
                             constrained_layout=True, squeeze=False)
    flat = [a for row in axes for a in row]
    for i, (concept, source) in enumerate(tasks):
        ax = flat[i]
        for (c, s, arch, scale, seed), curve in curves.items():
            if (c, s) != (concept, source):
                continue
            pts = sorted((valloss[(arch, scale, seed, st)], d) for st, d in curve
                         if (arch, scale, seed, st) in valloss)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=_COLOR.get(arch, "0.3"),
                        ls=_LS.get(scale, "-"), lw=1.0, alpha=0.7)
                drew = True
        ax.invert_xaxis()            # training progresses as val loss falls
        ax.set_xlabel("validation loss")
        if i == 0:
            ax.set_ylabel("Δ  (learned − baseline)")
        ax.set_title(f"{concept} / {source}")
        ax.grid(True, color="0.93", lw=0.6)
        ax.text(-0.16, 1.04, "abcdefgh"[i], transform=ax.transAxes, fontsize=10, fontweight="bold")
    if drew:
        for ax in flat[len(tasks):]:
            ax.set_visible(False)
        _save(fig, out)
    else:
        plt.close(fig)


def fig_layer_heatmaps(sidecars: List[dict], out: Path, *, concept="ss3", source="netsurfp") -> None:
    """Family 3: Delta_layer(t) heatmaps for the primary task, one per (arch,scale)."""
    sel = [d for d in sidecars if d.get("concept") == concept and d.get("source") == source]
    if not sel:
        return
    _style()
    sel = sorted(sel, key=lambda d: (_ARCHES.index(d.get("arch", "gpt2")) if d.get("arch") in _ARCHES else 9,
                                     _SCALES.index(d.get("scale", "S")) if d.get("scale") in _SCALES else 9,
                                     d.get("seed", 0)))
    n = len(sel)
    ncols = min(n, 4)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows),
                             constrained_layout=True, squeeze=False)
    flat = [a for row in axes for a in row]
    vmax = max((c["delta"] for d in sel for ck in d["checkpoints"] for c in ck["per_layer"]
                if c.get("delta") is not None), default=0.1)
    for i, d in enumerate(sel):
        ax = flat[i]
        cks = d["checkpoints"]
        steps = [ck["step"] for ck in cks]
        n_layers = len(cks[0]["per_layer"]) if cks else 0
        mat = [[ck["per_layer"][l]["delta"] for ck in cks] for l in range(n_layers)]
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=vmax,
                       extent=[0, len(steps), 0, 1])
        ax.set_title(f"{d.get('arch')}-{d.get('scale')} s{d.get('seed')}", fontsize=8)
        ax.set_xlabel("checkpoint →");
        if i % ncols == 0:
            ax.set_ylabel("relative depth")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Δ")
    for ax in flat[n:]:
        ax.set_visible(False)
    _save(fig, out)


def fig_emergence_times(curves, metric_by_task, out: Path) -> None:
    """Family 4: gpt2-vs-mamba t_onset/t50/t90 dumbbells per (task, scale)."""
    _style()
    rows = []
    for (concept, source, arch, scale, seed), curve in curves.items():
        steps = [s for s, _ in curve]; deltas = [d for _, d in curve]
        eps = DEFAULT_EPS.get(metric_by_task.get((concept, source), "macro_f1"), 0.02)
        et = emergence_times(steps, deltas, eps=eps)
        rows.append((concept, source, arch, scale, et))
    labels, gpt2_t, mamba_t = [], [], []
    for (concept, source) in _tasks(curves):
        for scale in _SCALES:
            g = _mean([r[4].get("t50") for r in rows if r[:4] == (concept, source, "gpt2", scale)])
            m = _mean([r[4].get("t50") for r in rows if r[:4] == (concept, source, "mamba", scale)])
            if g is not None and m is not None:
                labels.append(f"{concept}/{source[:4]} {scale}")
                gpt2_t.append(g); mamba_t.append(m)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(4.2, 0.4 * len(labels) + 1.2), constrained_layout=True)
    y = range(len(labels))
    for yi, g, m in zip(y, gpt2_t, mamba_t):
        ax.plot([g, m], [yi, yi], color="0.7", lw=1.0, zorder=1)
    ax.scatter(gpt2_t, list(y), color=_COLOR["gpt2"], label="gpt2", zorder=2, s=22)
    ax.scatter(mamba_t, list(y), color=_COLOR["mamba"], label="mamba", zorder=2, s=22)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.set_xscale("log"); ax.set_xlabel("t₅₀  (training step to half-final Δ)")
    ax.set_title("emergence half-time: gpt2 vs mamba")
    ax.legend(frameon=False)
    _save(fig, out)


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.with_suffix('.png')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="trajectory_results dir (run_trajectory_probes CSVs + traj_*.json).")
    ap.add_argument("--out", type=Path, default=Path("docs/figures/emergence"))
    args = ap.parse_args()

    curves, metric = load_curves(args.results_dir)
    if not curves:
        print(f"No trajectory CSVs in {args.results_dir}")
        return 0
    sidecars = load_sidecars(args.results_dir)
    valloss = load_val_loss_csv(args.results_dir)
    fig_vs_step(curves, Path(f"{args.out}_vs_step"))
    fig_vs_valloss(curves, valloss, Path(f"{args.out}_vs_valloss"))
    fig_layer_heatmaps(sidecars, Path(f"{args.out}_heatmaps"))
    fig_emergence_times(curves, metric, Path(f"{args.out}_times"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
