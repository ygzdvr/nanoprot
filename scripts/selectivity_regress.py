#!/usr/bin/env python
"""Selectivity law: regress the signed EARLY architecture margin on the data-only sequential-structure
scalar, and break the signal-strength confound.

Inputs: selectivity_scalars.csv (Pi, Omega per property; from selectivity_law.py), the rank_reversal
CSVs (d_early per property x scale, tagged by modality), and the trajectory dirs (converged decodability
= signal strength s_k). Reports Pearson + Spearman (permutation p), the PARTIAL correlation of d_early
with sigma controlling for s_k (the confound break), and writes a two-panel figure: (a) d_early vs sigma
(the law), (b) d_early vs s_k (signal strength alone does NOT predict reversal).
"""
from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

_RNG = np.random.default_rng(0)
# short display names + which properties are the story's anchors / the fold outlier
NICE = {"pfam_family": "family", "ec_class": "EC", "ss3": "ss3", "ss8": "ss8", "rsa": "rsa",
        "disorder": "disorder", "active": "active", "subcellular": "loc", "fold": "fold",
        "frame": "frame", "gc": "GC", "splice": "splice", "exon": "exon"}


def zscore(d):
    v = np.array(list(d.values()), float); m, s = v.mean(), v.std() + 1e-9
    return {k: (val - m) / s for k, val in d.items()}


def perm_p(x, y, stat, n=20000):
    obs = abs(stat(x, y)); c = 1
    for _ in range(n):
        if abs(stat(x, _RNG.permutation(y))) >= obs:
            c += 1
    return c / (n + 1)


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return pearson(rx.astype(float), ry.astype(float))


def partial(x, y, z):
    """corr(x,y | z) = (r_xy - r_xz r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))."""
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    den = np.sqrt(max(1e-9, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return (rxy - rxz * ryz) / den


def converged_decodability(traj_dirs):
    s = {}
    for d in traj_dirs:
        for f in glob.glob(str(Path(d) / "*.csv")):
            if Path(f).name == "val_loss.csv":
                continue
            rows = list(csv.DictReader(open(f)))
            if not rows or "concept" not in rows[0]:
                continue
            best = {}
            for r in rows:
                k = (r["arch"], r["scale"], r["seed"]); st = int(r["step"])
                if k not in best or st > best[k][0]:
                    best[k] = (st, float(r["learned_test"]))
            vals = [v for _, v in best.values()]
            if vals:
                s[rows[0]["concept"]] = float(np.mean(vals))
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scalars", type=Path, default=Path("docs/selectivity_scalars.csv"))
    ap.add_argument("--protein-rr", type=Path, default=Path("docs/rank_reversal_protein.csv"))
    ap.add_argument("--genome-rr", type=Path, default=Path("docs/rank_reversal_genome.csv"))
    ap.add_argument("--protein-traj", type=Path, default=Path(".cache/nanoprot/trajectory_results"))
    ap.add_argument("--genome-traj", type=Path, default=Path(".cache/nanoprot/trajectory_results_genome"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures/selectivity_law"))
    args = ap.parse_args()

    sc = {r["property"]: r for r in csv.DictReader(open(args.scalars))}
    Pi = {k: float(r["Pi"]) for k, r in sc.items()}
    Om = {k: float(r["Omega"]) for k, r in sc.items()}
    zPi, zOm = zscore(Pi), zscore(Om)
    sigma = {k: zPi[k] + zOm[k] for k in sc}
    strength = converged_decodability([args.protein_traj, args.genome_traj])

    # points at the (property, scale) grain — the mechanism predicts the SIGNED early margin per cell
    pts = []   # (concept, scale, modality, d_early)
    for f, mod in ((args.protein_rr, "protein"), (args.genome_rr, "genome")):
        for r in csv.DictReader(open(f)):
            c = r["task"]
            if c == "val_bpr" or c not in sigma:
                continue
            pts.append((c, r["scale"], mod, float(r["d_early"])))
    concepts_all = sorted({c for c, *_ in pts})
    xO = np.array([zOm[c] for c, *_ in pts])                 # order-sensitivity (the predictor)
    xS = np.array([sigma[c] for c, *_ in pts])               # Pi+Omega composite (weaker)
    y = np.array([d for *_, d in pts])
    s = np.array([strength.get(c, np.nan) for c, *_ in pts])
    keepfold = np.array([c != "fold" for c, *_ in pts])      # fold: k-mer probe can't see deep folds
    ok = ~np.isnan(s)

    print(f"  n = {len(pts)} (property x scale) cells, {len(concepts_all)} properties, 2 modalities")
    print(f"  d_early ~ Omega (order-sens.)      : r={pearson(xO, y):+.3f}  perm p={perm_p(xO, y, pearson):.4f}"
          f"   rho={spearman(xO, y):+.3f}")
    print(f"  d_early ~ Omega  (excl. fold)      : r={pearson(xO[keepfold], y[keepfold]):+.3f}  "
          f"perm p={perm_p(xO[keepfold], y[keepfold], pearson):.4f}")
    print(f"  d_early ~ sigma=Pi+Omega           : r={pearson(xS, y):+.3f}  (Pi alone r="
          f"{pearson(np.array([zPi[c] for c, *_ in pts]), y):+.3f}; Pi fires only on frame)")
    m = ok & keepfold
    print(f"  CONFOUND BREAK: partial(d~Omega | strength) = {partial(xO[m], y[m], s[m]):+.3f}   "
          f"raw d~strength r = {pearson(s[ok], y[ok]):+.3f}  (strength does NOT predict)")

    # ---- figure: (a) the law on Omega, (b) the null on signal strength ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as ml
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.8, "figure.dpi": 150})
    col = {"protein": "#3B6FB6", "genome": "#C0473B"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.4))
    seen = set()
    for (c, sca, mod, d), xo, xs_ in zip(pts, xO, xS):
        face = "none" if c == "fold" else col[mod]
        a.scatter(xo, d, s=32, facecolor=face, edgecolor=col[mod], linewidth=1.0, zorder=3)
        b.scatter(strength.get(c, np.nan), d, s=32, facecolor=face, edgecolor=col[mod],
                  linewidth=1.0, zorder=3)
        if c not in seen:
            a.annotate(NICE.get(c, c), (xo, d), fontsize=6.2, xytext=(3, 2), textcoords="offset points",
                       color=col[mod]); seen.add(c)
    for ax in (a, b):
        ax.axhline(0, color="0.6", lw=0.8, ls="--")
        ax.set_ylabel(r"early margin  $\Delta_{@1\%}=\mathrm{attn}-\mathrm{ssm}$")
    mm = np.polyfit(xO, y, 1); xs = np.linspace(xO.min(), xO.max(), 40)
    a.plot(xs, np.polyval(mm, xs), color="0.3", lw=1.2, zorder=1)
    a.set_xlabel(r"order-sensitivity  $z(\Omega_k)$  (data-only)")
    a.set_title(f"reversal tracks sequential structure\nr={pearson(xO, y):+.2f} (p={perm_p(xO, y, pearson):.3f}), "
                f"excl. fold r={pearson(xO[keepfold], y[keepfold]):+.2f}", fontsize=7.5)
    b.set_xlabel(r"converged decodability  $s_k$  (signal strength)")
    b.set_title(f"strength alone does NOT predict\nreversal (r={pearson(s[ok], y[ok]):+.2f})", fontsize=7.5)
    a.legend(handles=[ml.Line2D([], [], marker="o", ls="", color=col["protein"], label="protein"),
                      ml.Line2D([], [], marker="o", ls="", color=col["genome"], label="genome"),
                      ml.Line2D([], [], marker="o", ls="", mfc="none", mec="0.4", label="fold (k-mer blind)")],
             fontsize=6.3, loc="lower left", frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", bbox_inches="tight"); fig.savefig(str(args.out) + ".pdf", bbox_inches="tight")
    print(f"\n  wrote {args.out}.png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
