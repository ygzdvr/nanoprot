#!/usr/bin/env python3
"""Paper B, Fig 2 — the early-proxy mis-ranking window shrinks with scale.

Crossover fraction C×/C_f (compute fraction below which the state-space model leads, i.e. the
width of the window in which the early proxy mis-ranks) vs model scale, for the cleanly-reversing
secondary-structure tasks. Reads docs/rank_reversal.csv (produced by rank_reversal.py).
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCALE_X = {"XS": 0, "S": 1, "M": 2, "L": 3}
# Only the tasks with a clear, significant reversal (P(rev)=1.0 at S/M); others are transient/tied.
TASK_STYLE = {
    "ss8": ("#C44E52", "o", "8-state secondary structure"),
    "ss3": ("#4C72B0", "s", "3-state secondary structure"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("docs/rank_reversal.csv"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures/crossover_scaling"))
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(4.3, 3.3))
    for task, (color, marker, label) in TASK_STYLE.items():
        pts = sorted(((SCALE_X[r["scale"]], float(r["cross_over_frac"]))
                      for r in rows if r["task"] == task and r["scale"] in SCALE_X
                      and r.get("cross_over_frac") not in ("", "None", None)),
                     key=lambda t: t[0])
        if not pts:
            continue
        ax.plot([x for x, _ in pts], [y for _, y in pts], color=color, marker=marker,
                lw=1.8, ms=7, label=label)
    ax.set_yscale("log")
    ax.set_xticks(list(SCALE_X.values()))
    ax.set_xticklabels(list(SCALE_X.keys()))
    ax.set_xlim(0.7, 3.3)
    ax.set_xlabel("model scale")
    ax.set_ylabel("crossover fraction  C×/C_f\n(width of the mis-ranking window)")
    ax.legend(frameon=False, fontsize=8, title="capability probe")
    ax.grid(axis="y", which="both", color="0.92", lw=0.6)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(a.out) + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(str(a.out) + ".pdf", bbox_inches="tight")
    print("wrote", str(a.out) + ".png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
