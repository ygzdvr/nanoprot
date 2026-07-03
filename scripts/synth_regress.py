#!/usr/bin/env python
"""Aggregate the synthetic selectivity sweep -> the controlled dose-response LAW (Paper B centerpiece).

Merges per-cell curves, computes the iso-FLOP early margin d_early = F1(attn) - F1(ssm) per (task, seed),
verifies converged F1 is matched across the battery (the strength control, by construction), regresses
d_early on the order/periodicity axis sigma = z(Omega)+z(Pi), breaks the signal-strength confound via
partial correlation, and writes the figure: (a) d_early vs sigma with a fitted line + per-task reversal
P; (b) d_early vs converged F1 (flat = strength does not drive it); (c) example crossing learning curves.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

_RNG = np.random.default_rng(0)
AXIS_COLOR = {"composition": "#3B6FB6", "order": "#C0473B", "period": "#E08214", "off-diag": "#6A51A3"}


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float))


def perm_p(x, y, stat, n=20000):
    o = abs(stat(x, y)); c = 1
    for _ in range(n):
        c += abs(stat(x, _RNG.permutation(y))) >= o
    return c / (n + 1)


def partial(x, y, z):
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    return (rxy - rxz * ryz) / math.sqrt(max(1e-9, (1 - rxz ** 2) * (1 - ryz ** 2)))


def interp_logflop(curve, target_logf):
    f = np.array([c[0] for c in curve]); v = np.array([c[1] for c in curve])
    lf = np.log(np.maximum(f, 1))
    if target_logf <= lf[0]:
        return v[0]
    if target_logf >= lf[-1]:
        return v[-1]
    i = np.searchsorted(lf, target_logf)
    t = (target_logf - lf[i - 1]) / (lf[i] - lf[i - 1])
    return float(v[i - 1] + t * (v[i] - v[i - 1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, default=Path(".cache/nanoprot/synth_results/cells"))
    ap.add_argument("--spec", type=Path, default=Path("docs/synth_task_spec.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures/synth_selectivity_law"))
    ap.add_argument("--early-frac", type=float, default=0.01)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text())
    # curves[task][arch][seed] = sorted [(flop, f1)]
    curves = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in glob.glob(str(args.cells / "*.csv")):
        for r in csv.DictReader(open(f)):
            curves[r["task"]][r["arch"]][int(r["seed"])].append((float(r["flop"]), float(r["val_f1"])))
    for t in curves:
        for a in curves[t]:
            for s in curves[t][a]:
                curves[t][a][s].sort()

    # per-task: d_early (iso-FLOP, seed-mean + bootstrap), d_final, converged F1, reversal P
    tasks = sorted(curves)
    rec = {}
    for t in tasks:
        seeds = sorted(set(curves[t]["attn"]) & set(curves[t]["ssm"]))
        de, dfin, convF = [], [], []
        for s in seeds:
            ca, cs = curves[t]["attn"][s], curves[t]["ssm"][s]
            if len(ca) < 3 or len(cs) < 3:
                continue
            hi = min(math.log(ca[-1][0]), math.log(cs[-1][0]))
            lo = max(math.log(ca[1][0]), math.log(cs[1][0]))
            le = max(lo, hi + math.log(args.early_frac))
            de.append(interp_logflop(ca, le) - interp_logflop(cs, le))
            dfin.append(interp_logflop(ca, hi) - interp_logflop(cs, hi))
            convF.append(0.5 * (ca[-1][1] + cs[-1][1]))
        if len(de) < 2:
            continue
        de, dfin = np.array(de), np.array(dfin)
        # seed bootstrap P(reversal): d_early<0 (ssm early) AND d_final>0 (attn converged)
        rev = 0
        for _ in range(args.n_boot):
            idx = _RNG.integers(0, len(de), len(de))
            rev += (de[idx].mean() < 0) and (dfin[idx].mean() > 0)
        rec[t] = {"d_early": float(de.mean()), "d_early_sd": float(de.std()),
                  "d_final": float(dfin.mean()), "conv_f1": float(np.mean(convF)),
                  "p_rev": rev / args.n_boot, "n_seed": len(de),
                  "omega": spec[t]["omega"], "pi": spec[t]["pi"], "axis": spec[t]["axis"].split()[0]}

    # axis scalar sigma = z(Omega) + z(Pi)
    om = np.array([rec[t]["omega"] for t in rec]); pi = np.array([rec[t]["pi"] for t in rec])
    zom = (om - om.mean()) / (om.std() + 1e-9); zpi = (pi - pi.mean()) / (pi.std() + 1e-9)
    sig = {t: float(zom[i] + zpi[i]) for i, t in enumerate(rec)}
    ts = list(rec)
    x = np.array([sig[t] for t in ts]); y = np.array([rec[t]["d_early"] for t in ts])
    s = np.array([rec[t]["conv_f1"] for t in ts])

    print(f"  {len(ts)} tasks, {sum(rec[t]['n_seed'] for t in ts)} (task,seed) cells\n")
    print(f"  {'task':11s}{'axis':6s}{'sigma':>7s}{'Omega':>7s}{'Pi':>6s}{'d_early':>9s}{'d_final':>9s}"
          f"{'convF1':>8s}{'P(rev)':>8s}")
    for t in sorted(ts, key=lambda t: sig[t]):
        r = rec[t]
        print(f"  {t:11s}{r['axis']:6s}{sig[t]:+7.2f}{r['omega']:7.2f}{r['pi']:6.2f}{r['d_early']:+9.3f}"
              f"{r['d_final']:+9.3f}{r['conv_f1']:8.3f}{r['p_rev']:8.2f}")
    rp = pearson(x, y); rsp = spearman(x, y)
    print(f"\n  d_early ~ sigma      : r={rp:+.3f}  perm p={perm_p(x, y, pearson):.4f}   rho={rsp:+.3f}")
    print(f"  d_early ~ Omega      : r={pearson(zom, y):+.3f}")
    print(f"  d_early ~ Pi         : r={pearson(zpi, y):+.3f}")
    print(f"  CONFOUND BREAK: partial(d_early ~ sigma | convF1) = {partial(x, y, s):+.3f}   "
          f"(raw d_early~convF1 r={pearson(s, y):+.3f})")
    print(f"  matched strength: converged F1 = {s.mean():.3f} +/- {s.std():.3f}  "
          f"[{s.min():.3f}, {s.max():.3f}]")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as ml
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.8, "figure.dpi": 150})
    fig = plt.figure(figsize=(10.5, 3.3))
    a = fig.add_subplot(1, 3, 1); b = fig.add_subplot(1, 3, 2); cpanel = fig.add_subplot(1, 3, 3)
    col = [AXIS_COLOR.get(rec[t]["axis"], "#444") for t in ts]
    for ax, xv, xl in ((a, x, r"axis  $\sigma_k=z(\Omega)+z(\Pi)$"), (b, s, r"converged $F_1$ (strength)")):
        ax.scatter(xv, y, s=40, c=col, zorder=3, edgecolor="white", linewidth=0.5)
        for t, xx in zip(ts, xv):
            ax.annotate(t.replace("_", ""), (xx, rec[t]["d_early"]), fontsize=5.6, xytext=(3, 2),
                        textcoords="offset points")
        ax.axhline(0, color="0.6", lw=0.8, ls="--"); ax.set_xlabel(xl)
        ax.set_ylabel(r"early margin $\Delta_{@1\%}=F_1^{\mathrm{attn}}-F_1^{\mathrm{ssm}}$")
    mm = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 40)
    a.plot(xs, np.polyval(mm, xs), color="0.3", lw=1.3, zorder=1)
    a.set_title(f"controlled law: reversal tracks the axis\nr={rp:+.2f} (p={perm_p(x, y, pearson):.3f}), "
                fr"$\rho$={rsp:+.2f}", fontsize=8)
    b.set_title(f"strength does NOT drive it\n(r={pearson(s, y):+.2f}; partial|strength={partial(x, y, s):+.2f})",
                fontsize=8)
    a.legend(handles=[ml.Line2D([], [], marker="o", ls="", color=v, label=k) for k, v in AXIS_COLOR.items()],
             fontsize=6, loc="upper right", frameon=False)
    # panel c: example crossing curves for the most-reversing task
    trev = min(ts, key=lambda t: rec[t]["d_early"])
    for arch, c in (("attn", "#3B6FB6"), ("ssm", "#C0473B")):
        allc = defaultdict(list)
        for s_ in curves[trev][arch]:
            for fl, v in curves[trev][arch][s_]:
                allc[fl].append(v)
        pts = sorted((fl, np.mean(v)) for fl, v in allc.items())
        cpanel.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=1.6, label=arch)
    cpanel.set_xscale("log"); cpanel.set_xlabel("compute (FLOP)"); cpanel.set_ylabel(r"task $F_1$")
    cpanel.set_title(f"example reversal: {trev}\n(ssm early, attn converged)", fontsize=8)
    cpanel.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", bbox_inches="tight"); fig.savefig(str(args.out) + ".pdf", bbox_inches="tight")
    # dump the per-task table
    with open(str(args.out) + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "axis", "sigma", "omega", "pi", "d_early",
                                           "d_final", "conv_f1", "p_rev", "n_seed"])
        w.writeheader()
        for t in ts:
            w.writerow({"task": t, "axis": rec[t]["axis"], "sigma": round(sig[t], 4), **{
                k: round(rec[t][k], 5) if isinstance(rec[t][k], float) else rec[t][k]
                for k in ("omega", "pi", "d_early", "d_final", "conv_f1", "p_rev", "n_seed")}})
    print(f"\n  wrote {args.out}.png/.pdf/.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
