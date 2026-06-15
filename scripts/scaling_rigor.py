#!/usr/bin/env python
"""
Statistical hardening of nanoprot's cross-architecture scaling result.

The headline ("gpt2 out-scales mamba; gap widens with scale") is reported in
scaling_laws.py as point estimates. This script turns it into a defensible claim,
using the PER-SEED checkpoints (3 seeds x 4 scales per arch):

 1. Bootstrap CIs on the scaling exponent. alpha is the log-log slope of L vs N
    (the robust 2-parameter exponent, which always converges under resampling); a
    *stratified seed bootstrap* (seeds resampled within each scale, preserving the
    experimental structure) gives its 95% CI per architecture.

 2. The cross-architecture test. From the bootstrap, the distribution of
    d_alpha = alpha_gpt2 - alpha_mamba and P(alpha_gpt2 > alpha_mamba); PLUS the
    fit-free per-scale loss gap (mamba - gpt2) with 95% CIs, showing it is
    significantly positive at every scale and grows from XS to L.

 3. Iso-FLOP frontier. L vs training FLOPs (C = total_flops) per arch, so the gpt2
    win is shown at matched COMPUTE, not only at matched parameters.

gpt2/mamba share bits-per-residue and are compared directly; esm2 (masked-CE) is a
different metric, so its alpha is reported on its own and never cross-compared.
Deterministic: the bootstrap RNG is seeded, so reruns reproduce exactly.

Usage:
  python -m scripts.scaling_rigor --release-root $NANOPROT_BASE_DIR/release \
      --out figures/scaling_rigor [--bootstrap 4000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.aggregate_results import collect_cells  # noqa: E402

_SCALE_ORDER = {"XS": 0, "S": 1, "M": 2, "L": 3}
_COLOR = {"gpt2": "#1f4e79", "mamba": "#b5471f", "esm2": "#3f7f3f"}
_MARK = {"gpt2": "o", "mamba": "s", "esm2": "D"}


# ---------------------------------------------------------------------------
# Data shaping
# ---------------------------------------------------------------------------

def per_scale(cells: List[dict], arch: str) -> List[dict]:
    """Group an architecture's cells by scale (ascending N).

    Returns a list of {scale, N, C, Ls} where Ls is the list of per-seed
    val_bpr values at that scale, N the parameter count, C the mean total_flops.
    """
    by_scale: Dict[str, dict] = {}
    for c in cells:
        if c["arch"] != arch or c["val_bpr"] is None or not c["n_params"]:
            continue
        s = by_scale.setdefault(c["scale"], {"scale": c["scale"], "N": c["n_params"],
                                             "C": [], "Ls": []})
        s["Ls"].append(float(c["val_bpr"]))
        if c.get("total_flops"):
            s["C"].append(float(c["total_flops"]))
    rows = sorted(by_scale.values(), key=lambda r: _SCALE_ORDER.get(r["scale"], 9))
    for r in rows:
        r["C"] = float(np.mean(r["C"])) if r["C"] else None
    return rows


def loglog_alpha(N: np.ndarray, L: np.ndarray) -> float:
    """Robust scaling exponent: the negative slope of log L vs log N."""
    return float(-np.polyfit(np.log(N), np.log(L), 1)[0])


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_alpha(rows: List[dict], rng: np.random.Generator, B: int = 4000) -> np.ndarray:
    """Stratified seed bootstrap of the log-log exponent.

    Each iteration resamples the seeds *within* each scale (with replacement),
    takes the per-scale mean, and refits the slope across scales. Returns B alphas.
    Needs >= 2 scales; scales with their own seed lists are resampled independently.
    """
    N = np.array([r["N"] for r in rows], dtype=float)
    seed_arrays = [np.array(r["Ls"], dtype=float) for r in rows]
    out = np.empty(B, dtype=float)
    for b in range(B):
        L = np.array([sa[rng.integers(0, len(sa), len(sa))].mean() for sa in seed_arrays])
        out[b] = loglog_alpha(N, L)
    return out


def ci(x: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> Tuple[float, float]:
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def gap_per_scale(cells: List[dict], a_hi: str, a_lo: str, rng: np.random.Generator,
                  B: int = 4000) -> List[dict]:
    """Per-scale loss gap (mean a_hi - mean a_lo) with a 95% CI, fit-free.

    Bootstraps the two architectures' seeds independently at each shared scale.
    Positive gap => a_hi has higher loss (i.e. a_lo wins).
    """
    hi = {r["scale"]: r["Ls"] for r in per_scale(cells, a_hi)}
    lo = {r["scale"]: r["Ls"] for r in per_scale(cells, a_lo)}
    Ns = {r["scale"]: r["N"] for r in per_scale(cells, a_lo)}
    out = []
    for scale in sorted(set(hi) & set(lo), key=lambda s: _SCALE_ORDER.get(s, 9)):
        h, l = np.array(hi[scale], float), np.array(lo[scale], float)
        point = float(h.mean() - l.mean())
        draws = np.array([h[rng.integers(0, len(h), len(h))].mean()
                          - l[rng.integers(0, len(l), len(l))].mean() for _ in range(B)])
        clo, chi = ci(draws)
        out.append({"scale": scale, "N": Ns[scale], "gap": point, "ci": (clo, chi),
                    "p_pos": float((draws > 0).mean())})
    return out


# ---------------------------------------------------------------------------
# Iso-FLOP
# ---------------------------------------------------------------------------

def isoflop_points(cells: List[dict], arch: str) -> Tuple[np.ndarray, np.ndarray]:
    rows = [r for r in per_scale(cells, arch) if r["C"]]
    C = np.array([r["C"] for r in rows], dtype=float)
    L = np.array([float(np.mean(r["Ls"])) for r in rows], dtype=float)
    return C, L


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def analyze(cells: List[dict], B: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    result: dict = {"alpha": {}, "gap": None, "delta": None}
    for arch in ("gpt2", "mamba", "esm2"):
        rows = per_scale(cells, arch)
        if len(rows) < 2:
            continue
        N = np.array([r["N"] for r in rows], dtype=float)
        L = np.array([float(np.mean(r["Ls"])) for r in rows], dtype=float)
        boots = bootstrap_alpha(rows, rng, B)
        result["alpha"][arch] = {"point": loglog_alpha(N, L), "ci": ci(boots),
                                 "boots": boots, "n_scales": len(rows)}
    # cross-arch: d_alpha and per-scale gap (AR archs only)
    if "gpt2" in result["alpha"] and "mamba" in result["alpha"]:
        dg = result["alpha"]["gpt2"]["boots"] - result["alpha"]["mamba"]["boots"]
        result["delta"] = {"point": result["alpha"]["gpt2"]["point"]
                           - result["alpha"]["mamba"]["point"],
                           "ci": ci(dg), "p_gpt2_steeper": float((dg > 0).mean())}
        result["gap"] = gap_per_scale(cells, "mamba", "gpt2", rng, B)
    return result


def _print(result: dict) -> None:
    print("\n  Scaling exponent  alpha = -d(log L)/d(log N)   [95% CI, stratified seed bootstrap]\n")
    print(f"  {'arch':6} {'scales':>6} {'alpha':>8}   {'95% CI':>18}   metric")
    print("  " + "-" * 56)
    for arch in ("gpt2", "mamba", "esm2"):
        a = result["alpha"].get(arch)
        if not a:
            continue
        lo, hi = a["ci"]
        metric = "bpr" if arch in ("gpt2", "mamba") else "mCE (separate)"
        print(f"  {arch:6} {a['n_scales']:>6} {a['point']:>8.4f}   "
              f"[{lo:.4f}, {hi:.4f}]   {metric}")
    if result["delta"]:
        d = result["delta"]
        lo, hi = d["ci"]
        print(f"\n  gpt2 vs mamba:  d_alpha = {d['point']:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]"
              f"   P(gpt2 steeper) = {d['p_gpt2_steeper']:.3f}")
        sig = "non-overlapping with 0 -> gpt2 scales significantly faster" if lo > 0 \
            else "CI includes 0 -> not separable at this seed count"
        print(f"  => {sig}")
    if result["gap"]:
        print("\n  Per-scale loss gap  (mamba bpr - gpt2 bpr)   [95% CI; positive => gpt2 wins]\n")
        print(f"  {'scale':5} {'params':>9} {'gap':>9}   {'95% CI':>20}   {'P(gap>0)':>8}")
        print("  " + "-" * 60)
        for g in result["gap"]:
            lo, hi = g["ci"]
            p = g["N"]
            params = f"{p/1e6:.0f}M" if p < 1e9 else f"{p/1e9:.2f}B"
            print(f"  {g['scale']:5} {params:>9} {g['gap']:>+9.4f}   "
                  f"[{lo:+.4f}, {hi:+.4f}]   {g['p_pos']:>8.3f}")
    print()


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.4, "figure.dpi": 150,
    })


def _figure(cells: List[dict], result: dict, out: Path) -> None:
    _style()
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(9.6, 3.1), constrained_layout=True)

    # (a) iso-FLOP frontier: L vs C
    for arch in ("gpt2", "mamba"):
        C, L = isoflop_points(cells, arch)
        if len(C):
            axa.plot(C, L, marker=_MARK[arch], color=_COLOR[arch], lw=1.3, label=arch)
    axa.set_xscale("log"); axa.set_xlabel("training FLOPs (C = 6ND)")
    axa.set_ylabel("val bits-per-residue"); axa.set_title("iso-FLOP frontier")
    axa.grid(True, color="0.92", lw=0.6); axa.legend(frameon=False)
    axa.text(-0.2, 1.03, "a", transform=axa.transAxes, fontsize=10, fontweight="bold")

    # (b) bootstrap alpha distributions (violin, AR archs) -> separation
    data, labels, colors = [], [], []
    for arch in ("gpt2", "mamba"):
        a = result["alpha"].get(arch)
        if a is not None:
            data.append(a["boots"]); labels.append(arch); colors.append(_COLOR[arch])
    if data:
        vp = axb.violinplot(data, showmeans=True, showextrema=False)
        for body, col in zip(vp["bodies"], colors):
            body.set_facecolor(col); body.set_alpha(0.5); body.set_edgecolor(col)
        if "cmeans" in vp:
            vp["cmeans"].set_color("0.2")
        axb.set_xticks(range(1, len(labels) + 1)); axb.set_xticklabels(labels)
    axb.set_ylabel(r"scaling exponent $\alpha$")
    axb.set_title("bootstrap $\\alpha$ (95% CI)")
    axb.grid(True, axis="y", color="0.92", lw=0.6)
    axb.text(-0.2, 1.03, "b", transform=axb.transAxes, fontsize=10, fontweight="bold")

    # (c) per-scale loss gap with CI error bars (NOT bars)
    if result["gap"]:
        xs = [g["N"] for g in result["gap"]]
        ys = [g["gap"] for g in result["gap"]]
        lo = [g["gap"] - g["ci"][0] for g in result["gap"]]
        hi = [g["ci"][1] - g["gap"] for g in result["gap"]]
        axc.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color="#444", ecolor="#888",
                     capsize=3, lw=1.2, zorder=3)
        axc.axhline(0, color="0.6", lw=0.8, ls="--")
        axc.set_xscale("log")
    axc.set_xlabel("parameters"); axc.set_ylabel("mamba − gpt2  (bpr)")
    axc.set_title("loss gap vs scale"); axc.grid(True, color="0.92", lw=0.6)
    axc.text(-0.2, 1.03, "c", transform=axc.transAxes, fontsize=10, fontweight="bold")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.with_suffix('.png')} and .pdf\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("figures/scaling_rigor"))
    ap.add_argument("--bootstrap", type=int, default=4000, help="Bootstrap resamples (default 4000).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    cells = collect_cells(args.release_root, only_done=True)
    if not cells:
        print(f"No completed cells under {args.release_root} yet.")
        return 0
    result = analyze(cells, B=args.bootstrap, seed=args.seed)
    _print(result)
    if not args.no_figure:
        _figure(cells, result, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
