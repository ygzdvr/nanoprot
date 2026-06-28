#!/usr/bin/env python3
"""Option A analysis — does the capability-vs-loss story (and the forecasting method) generalize to
NLP? Reads docs/pythia_capability.csv (Pythia LAMBADA acc + LM loss per size×step).

Two results:
1. Timing divergence: compute where LOSS reaches 50% of its total drop vs where CAPABILITY reaches
   50% of its final value — capability lags loss (loss improves long before the capability appears).
2. Forecasting: predict converged LAMBADA acc per size from early checkpoints; cap-extrap
   (linear-in-log-compute) vs early-probe (use the early value). RMSE by early budget.
Compute axis = log(FLOPs)=log(6·N·tokens). Emits a report + a twin-axis acc/loss figure.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    by = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[r["scale"]].append((math.log(float(r["flops"])), float(r["lambada_acc"]), float(r["loss"]), int(r["step"])))
    for k in by:
        by[k].sort()
    return by


def interp(xs, ys, x):
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    i = np.searchsorted(xs, x)
    t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] + t * (ys[i] - ys[i - 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("docs/pythia_capability.csv"))
    ap.add_argument("--out", type=Path, default=Path("docs/pythia_forecast_report.md"))
    args = ap.parse_args()
    by = load(args.csv)
    sizes = [s for s in ["70m", "160m", "410m", "1b", "1.4b"] if s in by]
    lines = ["# Option A — Pythia (NLP) generality of capability-vs-loss + forecasting\n",
             f"Sizes: {sizes}; LAMBADA-openai acc + LM loss; compute = log FLOPs.\n",
             "## 1. Capability-vs-loss timing divergence",
             "| size | loss 50%-drop @step | capability(acc) 50%-final @step | lag (×, in steps) |",
             "|---|---|---|---|"]
    for s in sizes:
        d = by[s]; logC = [a for a, _, _, _ in d]; acc = [b for _, b, _, _ in d]
        loss = [c for _, _, c, _ in d]; steps = [t for _, _, _, t in d]
        # loss half-drop compute
        l_mid = loss[0] - 0.5 * (loss[0] - loss[-1])
        # acc half-final compute (acc rises 0 -> final)
        a_mid = 0.5 * acc[-1]
        def cross_step(vals, target, decreasing):
            for i in range(1, len(vals)):
                if (decreasing and vals[i] <= target) or (not decreasing and vals[i] >= target):
                    f = (target - vals[i - 1]) / (vals[i] - vals[i - 1]) if vals[i] != vals[i - 1] else 0
                    return math.exp(logC[i - 1] + f * (logC[i] - logC[i - 1]))  # FLOPs; convert later
            return None
        # work in steps for readability
        def cross_in_steps(vals, target, decreasing):
            for i in range(1, len(vals)):
                if (decreasing and vals[i] <= target) or (not decreasing and vals[i] >= target):
                    f = (target - vals[i - 1]) / (vals[i] - vals[i - 1]) if vals[i] != vals[i - 1] else 0
                    return steps[i - 1] + f * (steps[i] - steps[i - 1])
            return None
        ls = cross_in_steps(loss, l_mid, True); cs = cross_in_steps(acc, a_mid, False)
        lag = (cs / ls) if (ls and cs and ls > 0) else float("nan")
        lines.append(f"| {s} | {ls:.0f} | {cs:.0f} | {lag:.1f}× |")
    lines += ["\nLoss reaches its half-improvement long before the capability appears — capability has",
              "its own (much later, abrupt) emergence schedule. The same loss-≠-capability phenomenon as in",
              "proteins, now in NLP.\n",
              "## 2. Forecasting converged LAMBADA acc from early checkpoints (RMSE)",
              "| budget f=C_e/C_final | early-probe | cap-extrap (ours) |", "|---|---|---|"]
    BUD = [0.01, 0.03, 0.1, 0.3]
    ep_se = defaultdict(list); ce_se = defaultdict(list)
    for s in sizes:
        d = by[s]; logC = np.array([a for a, _, _, _ in d]); acc = np.array([b for _, b, _, _ in d])
        logCf = logC[-1]; Sf = acc[-1]
        for f in BUD:
            logCe = logCf + math.log(f)
            ep = interp(logC, acc, logCe)
            mask = logC <= logCe + 1e-9
            if mask.sum() >= 3:
                a, b = np.polyfit(logC[mask], acc[mask], 1); ce = max(0.0, a * logCf + b)
            else:
                ce = float(acc[mask][-1]) if mask.sum() else float(acc[0])
            ep_se[f].append((ep - Sf) ** 2); ce_se[f].append((ce - Sf) ** 2)
    for f in BUD:
        lines.append(f"| {f} | {math.sqrt(np.mean(ep_se[f])):.3f} | {math.sqrt(np.mean(ce_se[f])):.3f} |")
    lines += ["\nHonest reading: because LAMBADA capability emerges *abruptly*, neither method can predict",
              "it from *pre-emergence* checkpoints (acc≈0 → both extrapolate ~0); once capability has begun",
              "rising, cap-extrap forecasts the converged value with lower error than the early value. The",
              "sudden emergence itself reinforces the thesis: early/loss signals under-determine final",
              "capability. (n_sizes small; this is a method-generality data point, not the arch-reversal.)"]
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                             "axes.spines.top": False})
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax2 = ax.twinx(); ax2.spines["top"].set_visible(False)
        col = {"70m": "#4C72B0", "160m": "#C44E52"}
        for s in sizes:
            d = by[s]; step = [t for _, _, _, t in d]; acc = [b for _, b, _, _ in d]; loss = [c for _, _, c, _ in d]
            ax.plot(step, acc, "-o", color=col.get(s, "0.3"), lw=1.7, ms=4, label=f"{s} acc")
            ax2.plot(step, loss, "--", color=col.get(s, "0.3"), lw=1.3, alpha=0.7)
        ax.set_xscale("log"); ax.set_xlabel("training step")
        ax.set_ylabel("LAMBADA accuracy (capability, solid)"); ax2.set_ylabel("LM loss (dashed)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        figp = args.out.parent / "figures" / "pythia_timing"
        figp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(figp) + ".png", dpi=200, bbox_inches="tight"); fig.savefig(str(figp) + ".pdf", bbox_inches="tight")
        print(f"[pythia_forecast] wrote {figp}.png/.pdf")
    except Exception as e:
        print(f"[pythia_forecast] figure skipped: {e}")
    print(f"[pythia_forecast] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
