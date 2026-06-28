#!/usr/bin/env python3
"""Figure for P3.2 — early-proxy rank reversal.

Per task, the seed-paired margin Δ(C) = score(gpt2) − score(mamba) (higher ⇒ gpt2 ahead) as a
function of iso-FLOP compute C(t)=train_flops·step/train_residues, for scales S and M. A curve
that starts below 0 and crosses above is a rank reversal (mamba early winner → gpt2 converged
winner). The shaded band y<0 is the "mamba-ahead" region. Reuses the tested analysis loaders.
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import statistics as st  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rank_reversal import load_meta, load_task_curves, interp  # noqa: E402

TASKS = ["val_bpr", "ss8", "ss3", "active", "rsa"]   # reversing + non-reversing for contrast
SCALE_COLOR = {"S": "#4C72B0", "M": "#C44E52"}


def delta_grid(curves, scale, n=60):
    gs = {s for (a, sc, s) in curves if a == "gpt2" and sc == scale}
    ms = {s for (a, sc, s) in curves if a == "mamba" and sc == scale}
    pairs = []
    for s in sorted(gs & ms):
        g, m = curves[("gpt2", scale, s)], curves[("mamba", scale, s)]
        lo, hi = max(g[0][0], m[0][0]), min(g[-1][0], m[-1][0])
        if hi > lo:
            pairs.append((g, m, lo, hi))
    if not pairs:
        return None
    LO = max(p[2] for p in pairs); HI = min(p[3] for p in pairs)
    if HI <= LO:
        return None
    grid = [LO + (HI - LO) * i / (n - 1) for i in range(n)]
    perseed = [[interp(g, x) - interp(m, x) for x in grid] for g, m, _, _ in pairs]
    return grid, perseed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path(".cache/nanoprot/trajectory_results"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures/rank_reversal"))
    a = ap.parse_args()

    meta = load_meta(a.results_dir)
    tasks = load_task_curves(a.results_dir, meta)
    present = [t for t in TASKS if t in tasks]
    if not present:
        print("no tasks found"); return 1

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(present), figsize=(2.3 * len(present), 2.7), squeeze=False)
    axes = axes[0]
    for i, (ax, task) in enumerate(zip(axes, present)):
        ymin = 0.0
        for scale in ("S", "M"):
            dg = delta_grid(tasks[task], scale)
            if not dg:
                continue
            grid, perseed = dg
            x = [g / math.log(10) for g in grid]
            mean = [st.mean(col) for col in zip(*perseed)]
            ymin = min(ymin, min(min(ps) for ps in perseed))
            for ps in perseed:
                ax.plot(x, ps, color=SCALE_COLOR[scale], lw=0.5, alpha=0.30)
            ax.plot(x, mean, color=SCALE_COLOR[scale], lw=1.7, label=scale)
        ax.axhline(0, color="0.35", lw=0.8, ls="--")
        ax.axhspan(min(ymin * 1.1, -1e-4), 0, color="0.93", zorder=0)  # mamba-ahead region
        ax.set_title(task, fontsize=8)
        ax.set_xlabel("log₁₀ compute (a.u.)")
        ax.text(-0.08, 1.03, chr(97 + i), transform=ax.transAxes, fontweight="bold", fontsize=10)
        ax.margins(x=0.02)
    axes[0].set_ylabel("Δ = gpt2 − mamba  (>0 ⇒ gpt2 ahead)")
    axes[0].legend(title="scale", frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(a.out) + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(str(a.out) + ".pdf", bbox_inches="tight")
    print("wrote", str(a.out) + ".png / .pdf  (tasks:", ", ".join(present) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
