#!/usr/bin/env python
"""Aggregate the synthetic selectivity sweep -> the controlled dose-response LAW (Paper B centerpiece).

MATCHED STEP is the primary comparison (the iso-FLOP axis has a uniform pro-SSM bias -- the SSM is
cheaper per step -- which manufactures spurious reversals). The early margin is measured at the
ACQUISITION step (where learning actually happens: the first step the mean curve reaches acq_frac of the
way from chance to converged), NOT at a fixed 1% that can fall in the pre-learning noise floor.

d_early = F1(attn) - F1(ssm) at the matched acquisition step; d_final at convergence. Regress d_early on
the axis sigma=z(Omega)+z(Pi); break the signal-strength confound via partial correlation on converged
F1; write the figure. Reversal (per task, seed bootstrap): d_early<0 (ssm early) & d_final>0 (attn late).
"""
from __future__ import annotations

import argparse
import csv
import glob
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


def load_curves(cells):
    """[task][arch][seed] = {step: f1}."""
    c = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for f in glob.glob(str(cells / "*.csv")):
        for r in csv.DictReader(open(f)):
            c[r["task"]][r["arch"]][int(r["seed"])][int(r["step"])] = float(r["val_f1"])
    return c


def seed_mean(perseed):
    """{seed:{step:f1}} -> {step: mean over seeds present}."""
    allsteps = sorted({st for s in perseed for st in perseed[s]})
    out = {}
    for st in allsteps:
        vals = [perseed[s][st] for s in perseed if st in perseed[s]]
        if vals:
            out[st] = float(np.mean(vals))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, default=Path(".cache/nanoprot/synth_results/cells"))
    ap.add_argument("--spec", type=Path, default=Path("docs/synth_task_spec.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures/synth_selectivity_law"))
    ap.add_argument("--acq-frac", type=float, default=0.5, help="acquisition threshold (chance->converged)")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--archs", nargs=2, default=["attn", "ssm"],
                    help="the (late-winner, early-winner-hypothesis) arch pair; d = A0 - A1")
    args = ap.parse_args()
    A0, A1 = args.archs

    import json
    spec = json.loads(args.spec.read_text())
    C = load_curves(args.cells)
    tasks = sorted(C)
    rec = {}
    for t in tasks:
        if A0 not in C[t] or A1 not in C[t]:
            continue
        A, S = seed_mean(C[t][A0]), seed_mean(C[t][A1])
        steps = sorted(set(A) & set(S))
        if len(steps) < 4:
            continue
        chance = 0.5 * (A[steps[0]] + S[steps[0]]); fin = 0.5 * (A[steps[-1]] + S[steps[-1]])
        thr = chance + args.acq_frac * (fin - chance)
        t_acq = next((st for st in steps if 0.5 * (A[st] + S[st]) >= thr), steps[len(steps) // 3])
        seeds = sorted(set(C[t][A0]) & set(C[t][A1]))
        de, dfin, convF = [], [], []
        for s in seeds:
            ca, cs = C[t][A0][s], C[t][A1][s]
            if len(ca) < 4 or len(cs) < 4:
                continue
            ka = min(ca, key=lambda k: abs(k - t_acq)); ks = min(cs, key=lambda k: abs(k - t_acq))
            la, ls = max(ca), max(cs)
            de.append(ca[ka] - cs[ks]); dfin.append(ca[la] - cs[ls]); convF.append(0.5 * (ca[la] + cs[ls]))
        if len(de) < 2:
            continue
        de, dfin = np.array(de), np.array(dfin)
        rev = sum((de[i].mean() < 0) and (dfin[i].mean() > 0)
                  for i in [_RNG.integers(0, len(de), len(de)) for _ in range(args.n_boot)])
        rec[t] = {"d_early": float(de.mean()), "d_final": float(dfin.mean()),
                  "conv_f1": float(np.mean(convF)), "p_rev": rev / args.n_boot, "n_seed": len(de),
                  "t_acq": t_acq, "omega": spec[t]["omega"], "pi": spec[t]["pi"],
                  "axis": spec[t]["axis"].split()[0]}

    ts = list(rec)
    om = np.array([rec[t]["omega"] for t in ts]); pi = np.array([rec[t]["pi"] for t in ts])
    zom = (om - om.mean()) / (om.std() + 1e-9); zpi = (pi - pi.mean()) / (pi.std() + 1e-9)
    sig = {t: float(zom[i] + zpi[i]) for i, t in enumerate(ts)}
    x = np.array([sig[t] for t in ts]); y = np.array([rec[t]["d_early"] for t in ts])
    s = np.array([rec[t]["conv_f1"] for t in ts])

    print(f"  {len(ts)} tasks, {sum(rec[t]['n_seed'] for t in ts)} (task,seed) cells; "
          f"MATCHED-STEP margin at the acquisition step (acq_frac={args.acq_frac})\n")
    print(f"  {'task':11s}{'axis':6s}{'sigma':>7s}{'Om':>6s}{'Pi':>6s}{'t_acq':>6s}{'d_early':>9s}"
          f"{'d_final':>9s}{'convF1':>8s}{'P(rev)':>8s}")
    for t in sorted(ts, key=lambda t: sig[t]):
        r = rec[t]
        print(f"  {t:11s}{r['axis']:6s}{sig[t]:+7.2f}{r['omega']:6.2f}{r['pi']:6.2f}{r['t_acq']:6d}"
              f"{r['d_early']:+9.3f}{r['d_final']:+9.3f}{r['conv_f1']:8.3f}{r['p_rev']:8.2f}")
    rp = pearson(x, y); rsp = spearman(x, y)
    print(f"\n  d_early ~ sigma : r={rp:+.3f}  perm p={perm_p(x, y, pearson):.4f}   rho={rsp:+.3f}")
    print(f"  d_early ~ Omega : r={pearson(zom, y):+.3f}      d_early ~ Pi : r={pearson(zpi, y):+.3f}")
    print(f"  CONFOUND BREAK : partial(d_early~sigma | convF1)={partial(x, y, s):+.3f}   "
          f"(raw d_early~convF1 r={pearson(s, y):+.3f})")
    print(f"  matched strength: converged F1 = {s.mean():.3f} +/- {s.std():.3f}  [{s.min():.3f},{s.max():.3f}]")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as ml
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.8, "figure.dpi": 150})
    fig = plt.figure(figsize=(10.5, 3.3))
    a, b, cp = (fig.add_subplot(1, 3, i) for i in (1, 2, 3))
    col = [AXIS_COLOR.get(rec[t]["axis"], "#444") for t in ts]
    for ax, xv, xl in ((a, x, r"axis  $\sigma=z(\Omega)+z(\Pi)$"), (b, s, r"converged $F_1$ (strength)")):
        ax.scatter(xv, y, s=40, c=col, zorder=3, edgecolor="white", linewidth=0.5)
        for t, xx in zip(ts, xv):
            ax.annotate(t.replace("_", ""), (xx, rec[t]["d_early"]), fontsize=5.6, xytext=(3, 2),
                        textcoords="offset points")
        ax.axhline(0, color="0.6", lw=0.8, ls="--"); ax.set_xlabel(xl)
        ax.set_ylabel(rf"early margin $\Delta=F_1^{{\mathrm{{{A0}}}}}-F_1^{{\mathrm{{{A1}}}}}$ (<0: {A1} early)")
    mm = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 40)
    a.plot(xs, np.polyval(mm, xs), color="0.3", lw=1.3, zorder=1)
    a.set_title(f"controlled law (matched step)\nr={rp:+.2f} (p={perm_p(x, y, pearson):.3f}), "
                fr"$\rho$={rsp:+.2f}", fontsize=8)
    b.set_title(f"strength does NOT drive it\n(r={pearson(s, y):+.2f}; partial|str={partial(x, y, s):+.2f})",
                fontsize=8)
    a.legend(handles=[ml.Line2D([], [], marker="o", ls="", color=v, label=k) for k, v in AXIS_COLOR.items()],
             fontsize=6, loc="best", frameon=False)
    trev = min(ts, key=lambda t: rec[t]["d_early"])
    for arch, c in ((A0, "#3B6FB6"), (A1, "#C0473B")):
        m = seed_mean(C[trev][arch]); pts = sorted(m.items())
        cp.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=1.6, label=arch, marker="o", ms=2)
    cp.axvline(rec[trev]["t_acq"], color="0.6", ls=":", lw=0.8)
    cp.set_xlabel("training step"); cp.set_ylabel(r"task $F_1$")
    cp.set_title(f"most-reversing: {trev}\n(ssm early, attn converged)", fontsize=8)
    cp.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".png", bbox_inches="tight"); fig.savefig(str(args.out) + ".pdf", bbox_inches="tight")
    with open(str(args.out) + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "axis", "sigma", "omega", "pi", "d_early", "d_final",
                                           "conv_f1", "p_rev", "n_seed"])
        w.writeheader()
        for t in ts:
            w.writerow({"task": t, "axis": rec[t]["axis"], "sigma": round(sig[t], 4),
                        **{k: round(rec[t][k], 5) if isinstance(rec[t][k], float) else rec[t][k]
                           for k in ("omega", "pi", "d_early", "d_final", "conv_f1", "p_rev", "n_seed")}})
    print(f"\n  wrote {args.out}.png/.pdf/.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
