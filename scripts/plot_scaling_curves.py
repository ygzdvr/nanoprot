#!/usr/bin/env python
"""
Plot the nanoprot suite's scaling curves: validation metric vs. parameters.

Two panels, because the metrics are not interchangeable:
  (a) autoregressive models (gpt2, mamba) — bits-per-residue, directly comparable;
  (b) esm2 (masked LM) — masked cross-entropy, a different quantity.

Reads completed cells from a release directory (via scripts.aggregate_results),
so it is safe to run mid-training — it draws whatever has finished. Seed spread
is shown as error bars (typically tiny — the seeds are tight).

Usage:
  python -m scripts.plot_scaling_curves --release-root $NANOPROT_BASE_DIR/release \
      --out figures/scaling_curves
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.aggregate_results import collect_cells, aggregate, _SCALE_ORDER  # noqa: E402

# Restrained, distinct, non-rainbow palette (dark, print-safe).
_COLOR = {"gpt2": "#1f4e79", "mamba": "#b5471f", "esm2": "#3f7f3f"}
_MARK = {"gpt2": "o", "mamba": "s", "esm2": "D"}


def _series(agg, arch):
    """(params, mean, std, scale-labels) for one arch, sorted by params."""
    rows = [r for r in agg if r["arch"] == arch and r["val_bpr_mean"] is not None]
    rows.sort(key=lambda r: _SCALE_ORDER.get(r["scale"], 9))
    xs = [r["n_params"] for r in rows]
    ys = [r["val_bpr_mean"] for r in rows]
    es = [r["val_bpr_std"] or 0.0 for r in rows]
    labels = [r["scale"] for r in rows]
    return xs, ys, es, labels


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.4, "lines.markersize": 5,
        "figure.dpi": 150,
    })


def _panel(ax, agg, archs, ylabel, title, tag):
    drawn = False
    for arch in archs:
        xs, ys, es, labels = _series(agg, arch)
        if not xs:
            continue
        drawn = True
        ax.errorbar(xs, ys, yerr=es, marker=_MARK[arch], color=_COLOR[arch],
                    capsize=2, elinewidth=0.8, label=arch, zorder=3)
        # annotate the largest point with its scale ladder rung
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4),
                        fontsize=6, color=_COLOR[arch])
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="center")
    ax.text(-0.18, 1.02, tag, transform=ax.transAxes, fontsize=10, fontweight="bold")
    if drawn:
        ax.legend(frameon=False, loc="upper right")
    else:
        ax.text(0.5, 0.5, "(no completed cells yet)", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="0.6")
    ax.grid(True, which="major", axis="both", color="0.9", linewidth=0.6, zorder=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("figures/scaling_curves"))
    args = ap.parse_args()

    cells = collect_cells(args.release_root, only_done=True)
    if not cells:
        print(f"No completed cells under {args.release_root} yet.")
        return 0
    agg = aggregate(cells)

    _style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 2.9), constrained_layout=True)
    _panel(axa, agg, ["gpt2", "mamba"], "val bits-per-residue",
           "autoregressive (gpt2, mamba)", "a")
    _panel(axb, agg, ["esm2"], "val masked-CE (bits)",
           "masked LM (esm2)", "b")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = args.out.with_suffix(".png"), args.out.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    n_done = len(cells)
    print(f"  wrote {png} and {pdf}  ({n_done} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
