#!/usr/bin/env python
"""
Fit nanoprot's compute-optimal scaling laws and plot them.

Each (arch, scale) cell is one fully-LR-decayed point (N=params, D=residues,
C=6ND FLOPs, L=val metric) on the compute-optimal line (D = ratio * N). For each
architecture we fit the Chinchilla-style form

    L(N) = E + A * N^(-alpha)

(E = irreducible loss, alpha = scaling exponent), and also the robust 2-parameter
pure power law L = A * N^(-alpha). AR archs (gpt2, mamba) share bits-per-residue
and are directly comparable; esm2 (masked-LM) is fit separately on its own metric.

Because the grid is compute-optimal, alpha here is the *combined* compute-optimal
exponent (N and D co-vary); it is NOT the isolated parameter-exponent of a full
L(N, D) surface. See the module docstring of the release scope for the caveat.

Safe to run mid-training (fits whatever cells have finished; needs >= 3 scales
for a 3-parameter fit, >= 2 for the pure power law).

Usage:
  python -m scripts.scaling_laws --release-root $NANOPROT_BASE_DIR/release \
      --out figures/scaling_laws
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

from scripts.aggregate_results import collect_cells, aggregate  # noqa: E402

_COLOR = {"gpt2": "#1f4e79", "mamba": "#b5471f", "esm2": "#3f7f3f"}
_MARK = {"gpt2": "o", "mamba": "s", "esm2": "D"}


def _points(agg, arch) -> Tuple[np.ndarray, np.ndarray]:
    rows = [r for r in agg if r["arch"] == arch and r["val_bpr_mean"] is not None
            and r["n_params"]]
    rows.sort(key=lambda r: r["n_params"])
    N = np.array([r["n_params"] for r in rows], dtype=float)
    L = np.array([r["val_bpr_mean"] for r in rows], dtype=float)
    return N, L


def fit_power_law(N: np.ndarray, L: np.ndarray) -> Optional[Dict[str, float]]:
    """Fit L = E + A*N^-alpha (3-param) and L = A*N^-alpha (2-param)."""
    from scipy.optimize import curve_fit
    out: Dict[str, float] = {}
    # 2-param pure power law (log-linear, robust with few points)
    if len(N) >= 2:
        coef = np.polyfit(np.log(N), np.log(L), 1)
        out["pl_alpha"] = float(-coef[0])
        out["pl_A"] = float(np.exp(coef[1]))
        pred = out["pl_A"] * N ** (-out["pl_alpha"])
        out["pl_r2"] = float(1 - np.sum((L - pred) ** 2) / np.sum((L - L.mean()) ** 2))
    # 3-param Chinchilla form (needs >= 3 points; E weakly constrained at 4)
    if len(N) >= 3:
        def f(n, E, A, alpha):
            return E + A * n ** (-alpha)
        try:
            p0 = [min(L) * 0.8, out.get("pl_A", 10.0), out.get("pl_alpha", 0.05)]
            popt, _ = curve_fit(f, N, L, p0=p0, maxfev=20000,
                                bounds=([0, 0, 0], [min(L), np.inf, 1.0]))
            out["E"], out["A"], out["alpha"] = map(float, popt)
            pred = f(N, *popt)
            out["r2"] = float(1 - np.sum((L - pred) ** 2) / np.sum((L - L.mean()) ** 2))
        except Exception:
            pass
    return out or None


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.4, "lines.markersize": 5,
        "figure.dpi": 150,
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("figures/scaling_laws"))
    args = ap.parse_args()

    agg = aggregate(collect_cells(args.release_root, only_done=True))
    if not agg:
        print("No completed cells yet.")
        return 0

    fits: Dict[str, Dict[str, float]] = {}
    print("\n  Compute-optimal scaling-law fits  (L = E + A·N^-alpha)\n")
    print(f"  {'arch':6} {'n_scales':>8} {'alpha':>8} {'E(irred)':>9} {'R^2':>6}   (pure PL: alpha, R^2)")
    print("  " + "-" * 72)
    for arch in ("gpt2", "mamba", "esm2"):
        N, L = _points(agg, arch)
        if len(N) < 2:
            continue
        fit = fit_power_law(N, L)
        if not fit:
            continue
        fits[arch] = fit
        a3 = fit.get("alpha"); e3 = fit.get("E"); r3 = fit.get("r2")
        a3s = f"{a3:.4f}" if a3 is not None else "  --  "
        e3s = f"{e3:.4f}" if e3 is not None else "  --  "
        r3s = f"{r3:.4f}" if r3 is not None else "  --  "
        print(f"  {arch:6} {len(N):>8} {a3s:>8} {e3s:>9} {r3s:>6}   "
              f"(alpha={fit['pl_alpha']:.4f}, R^2={fit['pl_r2']:.4f})")
    print("\n  alpha = compute-optimal scaling exponent (N and D co-vary; not the "
          "isolated\n  parameter-exponent). gpt2/mamba comparable (bpr); esm2 separate (mCE).\n")

    # ---- figure: L vs N (log-log) with fitted curves ----
    _style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.0), constrained_layout=True)
    for ax, archs, ylabel, title, tag in [
        (axa, ["gpt2", "mamba"], "val bits-per-residue", "autoregressive", "a"),
        (axb, ["esm2"], "val masked-CE (bits)", "masked LM", "b"),
    ]:
        any_drawn = False
        for arch in archs:
            N, L = _points(agg, arch)
            if len(N) == 0:
                continue
            any_drawn = True
            ax.scatter(N, L, marker=_MARK[arch], color=_COLOR[arch], zorder=3, label=arch)
            fit = fits.get(arch)
            if fit and "alpha" in fit:
                xs = np.geomspace(N.min(), N.max(), 100)
                ys = fit["E"] + fit["A"] * xs ** (-fit["alpha"])
                ax.plot(xs, ys, color=_COLOR[arch], lw=1.2, alpha=0.8,
                        label=f"  α={fit['alpha']:.3f}")
            elif fit:
                xs = np.geomspace(N.min(), N.max(), 100)
                ys = fit["pl_A"] * xs ** (-fit["pl_alpha"])
                ax.plot(xs, ys, color=_COLOR[arch], lw=1.2, ls="--", alpha=0.8,
                        label=f"  α≈{fit['pl_alpha']:.3f}")
        ax.set_xscale("log")
        ax.set_xlabel("parameters")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.text(-0.18, 1.03, tag, transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.grid(True, color="0.92", lw=0.6, zorder=0)
        if any_drawn:
            ax.legend(frameon=False, loc="upper right")
        else:
            ax.text(0.5, 0.5, "(no cells yet)", transform=ax.transAxes,
                    ha="center", va="center", color="0.6")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {args.out.with_suffix('.png')} and .pdf\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
