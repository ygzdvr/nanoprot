#!/usr/bin/env python
"""Hierarchical (partial-pooling) posterior for the rank-reversal (strategy-doc priority #3).

Replaces the naive 3-seed bootstrap P(reversal) with a defensible Bayesian posterior. The bootstrap over
3 seeds has ~a handful of distinct resamples and grossly under-estimates tails, so "P(reversal)=1.0" is a
low evidential bar. Here each cell's per-seed early/final margins are noisy observations of a cell-level
mean, the cell means share a modality-level hyperprior (partial pooling shrinks noisy 3-seed cells toward
the group and widens their intervals honestly), and

    P(reversal | data)  =  E_posterior[ 1( theta_early[cell] < 0  AND  theta_final[cell] > 0 ) ]

is computed by integrating over the hierarchy (Gibbs draws). Margins are extracted with the SAME loaders
and interpolation as rank_reversal.py, so the point estimates match; only the uncertainty model changes.

Model (per modality, early and final margins, conjugate Normal hierarchy):
    y[c,r] ~ N(theta[c], sigma^2)      theta[c] ~ N(mu, tau^2)
    mu ~ N(0, 100)    tau^2, sigma^2 ~ InvGamma(1, 0.01)   (weakly informative)
theta_early[c] and theta_final[c] are a-posteriori independent (separate likelihoods/hyperparams), so
the reversal indicator is AND-ed over paired independent draws.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from scripts.rank_reversal import AR, interp, load_meta, load_task_curves

_RNG = np.random.default_rng(0)


def cell_margins(task_curves, scale, early_frac=0.01):
    """Per-seed seed-paired early/final margins for one (concept, scale) cell — replicates
    rank_reversal.analyze() exactly. Returns (de, df) lists or (None, None)."""
    g = {s for (a, sc, s) in task_curves if a == AR[0] and sc == scale}
    m = {s for (a, sc, s) in task_curves if a == AR[1] and sc == scale}
    pairs = []
    for s in sorted(g & m):
        cg, cm = task_curves[(AR[0], scale, s)], task_curves[(AR[1], scale, s)]
        if min(cg[-1][0], cm[-1][0]) > max(cg[0][0], cm[0][0]):
            pairs.append((cg, cm))
    if len(pairs) < 2:
        return None, None
    LO = max(max(cg[0][0], cm[0][0]) for cg, cm in pairs)
    HI = min(min(cg[-1][0], cm[-1][0]) for cg, cm in pairs)
    if HI <= LO:
        return None, None
    lce = max(LO, HI + math.log(early_frac))
    de = [interp(cg, lce) - interp(cm, lce) for cg, cm in pairs]
    df = [interp(cg, HI) - interp(cm, HI) for cg, cm in pairs]
    return de, df


def gibbs_hier(Y, cells, *, n_iter=8000, burn=2000, seed=0):
    """Conjugate Gibbs for the Normal hierarchy above. Y: cell -> list of per-seed values.
    Returns theta draws [n_draw, C]."""
    rng = np.random.default_rng(seed)
    ybar = np.array([np.mean(Y[c]) for c in cells])
    nc = np.array([len(Y[c]) for c in cells], float)
    C = len(cells); N = nc.sum()
    theta = ybar.copy(); mu = float(theta.mean()); tau2 = float(theta.var() + 1e-2); sig2 = 0.01
    a0, b0, v0 = 1.0, 0.01, 100.0
    out = []
    for it in range(n_iter):
        prec = nc / sig2 + 1.0 / tau2                       # theta[c] | .
        mean = (nc * ybar / sig2 + mu / tau2) / prec
        theta = rng.normal(mean, np.sqrt(1.0 / prec))
        vpost = 1.0 / (1.0 / v0 + C / tau2)                 # mu | .
        mu = rng.normal(vpost * (theta.sum() / tau2), math.sqrt(vpost))
        tau2 = 1.0 / rng.gamma(a0 + C / 2.0, 1.0 / (b0 + 0.5 * float(np.sum((theta - mu) ** 2))))
        sse = sum(float(np.sum((np.asarray(Y[c]) - theta[i]) ** 2)) for i, c in enumerate(cells))
        sig2 = 1.0 / rng.gamma(a0 + N / 2.0, 1.0 / (b0 + 0.5 * sse))
        if it >= burn:
            out.append(theta.copy())
    return np.array(out)


def analyze_modality(results_dir: Path, scales, early_frac, label):
    meta = load_meta(results_dir)
    curves = load_task_curves(results_dir, meta)
    cells, Ye, Yf, boot = [], {}, {}, {}
    for concept in sorted(curves):
        if concept == "val_bpr":
            continue
        for sc in scales:
            de, df = cell_margins(curves[concept], sc, early_frac)
            if de is None:
                continue
            c = f"{concept}/{sc}"
            cells.append(c); Ye[c] = de; Yf[c] = df
            # old bootstrap P(reversal) on the same margins, for side-by-side
            de_a, df_a = np.array(de), np.array(df); rev = 0
            for _ in range(4000):
                idx = _RNG.integers(0, len(de_a), len(de_a))
                rev += (de_a[idx].mean() < 0) and (df_a[idx].mean() > 0)
            boot[c] = rev / 4000.0
    if len(cells) < 2:
        print(f"  [{label}] <2 cells; skipping"); return
    de_draws = gibbs_hier(Ye, cells, seed=1)
    df_draws = gibbs_hier(Yf, cells, seed=2)
    print(f"\n===== {label} : hierarchical posterior (partial pooling over {len(cells)} cells) =====")
    print(f"  {'cell':16s}{'n':>3s}{'d_early':>9s}{'bootP':>7s}{'hierP(rev)':>11s}{'theta_early 95% CrI':>22s}")
    reslines = []
    for i, c in enumerate(cells):
        te, tf = de_draws[:, i], df_draws[:, i]
        p_rev = float(np.mean((te < 0) & (tf > 0)))
        lo, hi = np.percentile(te, [2.5, 97.5])
        star = "  <-- headline" if c.split("/")[0] in ("ss3", "ss8", "fold", "frame") else ""
        line = (f"  {c:16s}{len(Ye[c]):3d}{np.mean(Ye[c]):+9.3f}{boot[c]:7.2f}{p_rev:11.2f}"
                f"   [{lo:+.3f},{hi:+.3f}]{star}")
        print(line); reslines.append((c, p_rev, boot[c]))
    # population-level "does the reversal hold on average" (hypermean of early / final margins)
    # recover mu draws cheaply: mu ~ mean of theta across cells per draw (pooled center)
    mu_e = de_draws.mean(1); mu_f = df_draws.mean(1)
    print(f"  population early-margin hypermean: {mu_e.mean():+.3f} (95% CrI "
          f"[{np.percentile(mu_e,2.5):+.3f},{np.percentile(mu_e,97.5):+.3f}]); "
          f"P(mu_early<0)={float(np.mean(mu_e<0)):.2f}")

    # forest plot: theta_early 95% CrI per cell, colored by hierarchical P(reversal)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.8})
    order = sorted(range(len(cells)), key=lambda i: de_draws[:, i].mean())
    fig, ax = plt.subplots(figsize=(5.4, 0.32 * len(cells) + 0.7))
    for row, i in enumerate(order):
        te = de_draws[:, i]; lo, hi = np.percentile(te, [2.5, 97.5])
        p = float(np.mean((te < 0) & (df_draws[:, i] > 0)))
        col = "#C0473B" if p >= 0.9 else ("#E08214" if p >= 0.5 else "#3B6FB6")
        ax.plot([lo, hi], [row, row], color=col, lw=2.2, solid_capstyle="round")
        ax.plot(te.mean(), row, "o", color=col, ms=4, zorder=3)
        hl = cells[i].split("/")[0] in ("ss3", "ss8", "fold", "frame")
        ax.text(max(hi, 0) + 0.006, row, f"{cells[i]}  n={len(Ye[cells[i]])}  P={p:.2f}", va="center",
                fontsize=6.2, fontweight="bold" if hl else "normal", color="0.15" if hl else "0.4")
    ax.axvline(0, color="0.5", ls="--", lw=0.9); ax.set_yticks([]); ax.set_ylim(-0.7, len(cells) - 0.3)
    ax.set_xlabel(r"early margin $\theta_{\mathrm{early}}=\mathrm{gpt2}-\mathrm{mamba}$  (95% posterior CrI)")
    ns = [len(Ye[c]) for c in cells]; nrange = f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}–{max(ns)}"
    ax.set_title(f"{label}: hierarchical reversal posterior\n(partial pooling, {nrange} seeds/cell; "
                 f"red $P{{\\geq}}0.9$, orange $\\geq0.5$, blue $<0.5$)", fontsize=7.5)
    fig.tight_layout()
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"docs/figures/hier_reversal_{label.lower()}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"docs/figures/hier_reversal_{label.lower()}.pdf", bbox_inches="tight")
    print(f"  wrote docs/figures/hier_reversal_{label.lower()}.png/.pdf")
    return reslines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protein-dir", type=Path, default=Path(".cache/nanoprot/trajectory_results"))
    ap.add_argument("--genome-dir", type=Path, default=Path(".cache/nanoprot/trajectory_results_genome"))
    ap.add_argument("--scales", nargs="+", default=["S", "M"])
    ap.add_argument("--early-frac", type=float, default=0.01)
    args = ap.parse_args()
    if args.protein_dir.exists():
        analyze_modality(args.protein_dir, args.scales, args.early_frac, "PROTEIN")
    if args.genome_dir.exists():
        analyze_modality(args.genome_dir, args.scales, args.early_frac, "GENOME")
    print("\n  Interpretation: hierP(rev) is the honest reversal probability; where it falls well below the"
          "\n  bootstrap, the 3-seed bootstrap was over-confident (the fix the strategy doc calls for).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
