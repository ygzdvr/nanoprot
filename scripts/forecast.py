#!/usr/bin/env python3
"""P3.4 (v2) — forecast converged capability + architecture ranking from EARLY checkpoints.

Fixes two issues found in v1:
  (1) FAIR early-probe baseline — use the last checkpoint AT/BEFORE the budget C_e (v1 interpolated
      between the bracketing checkpoints, peeking at one just past C_e → unfair to our method).
  (2) BOUNDED/SATURATING forecaster S(C)=S_inf·σ(k(logC−logC0)) — linear-in-log-compute overshoots
      saturating capability curves (e.g., LAMBADA), so we add a logistic-in-log-compute fit.

Methods compared at each early budget f=C_e/C_final:
  - early-probe (fair): last value at/before C_e.
  - loss-rank: rank by linearly-extrapolated final loss (ranking only).
  - cap-linear: linear-in-logC capability extrapolation (v1).
  - cap-sat (ours): logistic-in-logC capability extrapolation (v2).
Runs on PROTEINS (multi-arch → ranking accuracy + value RMSE) and PYTHIA (NLP → value RMSE).
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
import importlib.util

import numpy as np
from scipy.optimize import least_squares

_spec = importlib.util.spec_from_file_location("rr", str(Path(__file__).resolve().parent / "rank_reversal.py"))
rr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rr)

PROT_CONCEPTS = ["ss3", "ss8", "rsa", "active", "disorder"]
PROT_SCALES = ["S", "M", "L"]
BUDGETS = [0.01, 0.03, 0.1, 0.3]


def avail(curve, logCe):
    return [(lc, s) for lc, s in curve if lc <= logCe + 1e-9]


def fair_early(curve, logCe, logCf):
    a = avail(curve, logCe)
    return a[-1][1] if a else curve[0][1]


def cap_linear(curve, logCe, logCf, clip=(0.0, 1.0)):
    a = avail(curve, logCe)
    if len(a) < 3:
        return fair_early(curve, logCe, logCf)
    x = np.array([p[0] for p in a]); y = np.array([p[1] for p in a])
    m, b = np.polyfit(x, y, 1)
    v = m * logCf + b
    return float(v if clip is None else np.clip(v, clip[0], clip[1]))  # clip OK for capability∈[0,1]; pass None for loss (−bpr<0)


def cap_sat(curve, logCe, logCf):
    """Logistic-in-log-compute: S_inf·sigmoid(k·(logC−logC0)). Robust (soft_l1) fit on ≤C_e points."""
    a = avail(curve, logCe)
    if len(a) < 4:
        return fair_early(curve, logCe, logCf)
    x = np.array([p[0] for p in a]); y = np.array([p[1] for p in a])
    def model(p, x): Si, k, x0 = p; return Si / (1.0 + np.exp(-np.clip(k * (x - x0), -60, 60)))
    def resid(p, x, y): return model(p, x) - y
    p0 = [max(float(y.max()), 0.05), 1.0, float(x.mean())]
    lo = [0.0, 0.05, float(x.min()) - 10]; hi = [1.2, 30.0, float(x.max()) + 15]
    try:
        r = least_squares(resid, p0, args=(x, y), bounds=(lo, hi), loss="soft_l1",
                          f_scale=0.05, max_nfev=10000)
        return float(np.clip(model(r.x, np.array([logCf]))[0], 0.0, 1.0))
    except Exception:
        return cap_linear(curve, logCe, logCf)


METHODS = {"early-probe": fair_early, "cap-linear": cap_linear, "cap-sat": cap_sat}


def eval_proteins(rd):
    meta = rr.load_meta(rd); tasks = rr.load_task_curves(rd, meta)
    rank_ok = defaultdict(list); val_se = defaultdict(list); cells = 0
    for concept in PROT_CONCEPTS:
        if concept not in tasks:
            continue
        for scale in PROT_SCALES:
            ac = {a: [tasks[concept][(a, scale, s)] for (aa, sc, s) in tasks[concept]
                      if aa == a and sc == scale] for a in ("gpt2", "mamba")}
            if not ac["gpt2"] or not ac["mamba"]:
                continue
            logCf = min(min(c[-1][0] for c in ac[a]) for a in ("gpt2", "mamba"))
            trueS = {a: float(np.mean([rr.interp(c, logCf) for c in ac[a]])) for a in ac}
            tr = 1 if trueS["gpt2"] > trueS["mamba"] else -1
            cells += 1
            lc = {a: [tasks["val_bpr"][(a, scale, s)] for (aa, sc, s) in tasks["val_bpr"]
                      if aa == a and sc == scale] for a in ("gpt2", "mamba")}
            for f in BUDGETS:
                logCe = logCf + math.log(f)
                for mname, mfn in METHODS.items():
                    pred = {a: float(np.mean([mfn(c, logCe, logCf) for c in ac[a]])) for a in ac}
                    rank_ok[(mname, f)].append((1 if pred["gpt2"] > pred["mamba"] else -1) == tr)
                    val_se[(mname, f)] += [(pred[a] - trueS[a]) ** 2 for a in ac]
                if lc["gpt2"] and lc["mamba"]:
                    pl = {a: float(np.mean([cap_linear(c, logCe, logCf, clip=None) for c in lc[a]])) for a in lc}
                    rank_ok[("loss-rank", f)].append((1 if pl["gpt2"] > pl["mamba"] else -1) == tr)
    return rank_ok, val_se, cells


def eval_pythia(csv_path):
    by = defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        by[r["scale"]].append((math.log(float(r["flops"])), float(r["lambada_acc"])))
    for k in by:
        by[k].sort()
    val_se = defaultdict(list)
    for s, curve in by.items():
        logCf = curve[-1][0]; trueS = curve[-1][1]
        for f in BUDGETS:
            logCe = logCf + math.log(f)
            for mname, mfn in METHODS.items():
                val_se[(mname, f)].append((mfn(curve, logCe, logCf) - trueS) ** 2)
    return val_se, len(by)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path(".cache/nanoprot/trajectory_results"))
    ap.add_argument("--pythia-csv", type=Path, default=Path("docs/pythia_capability.csv"))
    ap.add_argument("--out", type=Path, default=Path("docs/forecast_v2_report.md"))
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    rank_ok, val_se, cells = eval_proteins(a.results_dir)
    def accci(m, f):
        v = np.array(rank_ok[(m, f)], float)
        if not len(v): return (float("nan"),) * 3
        bt = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(a.n_boot)])
        return v.mean(), np.quantile(bt, .025), np.quantile(bt, .975)
    def rmse(d, m, f): return math.sqrt(float(np.mean(d[(m, f)]))) if d[(m, f)] else float("nan")

    L = [f"# P3.4 v2 — fair baseline + saturating forecaster (proteins + Pythia)\n",
         f"## Proteins — ranking accuracy (gpt2 vs mamba), {cells} cells, mean [95% CI] by budget",
         "| method | " + " | ".join(f"f={f}" for f in BUDGETS) + " |", "|" + "---|" * (len(BUDGETS) + 1)]
    for m in ["early-probe", "loss-rank", "cap-linear", "cap-sat"]:
        L.append(f"| {m} | " + " | ".join(
            (lambda t: f"{t[0]:.2f} [{t[1]:.2f},{t[2]:.2f}]")(accci(m, f)) for f in BUDGETS) + " |")
    L += ["\n## Proteins — converged-value RMSE by budget",
          "| method | " + " | ".join(f"f={f}" for f in BUDGETS) + " |", "|" + "---|" * (len(BUDGETS) + 1)]
    for m in ["early-probe", "cap-linear", "cap-sat"]:
        L.append(f"| {m} | " + " | ".join(f"{rmse(val_se, m, f):.3f}" for f in BUDGETS) + " |")

    pse, nsz = eval_pythia(a.pythia_csv)
    L += [f"\n## Pythia (NLP, {nsz} sizes) — converged-value RMSE by budget (generality)",
          "| method | " + " | ".join(f"f={f}" for f in BUDGETS) + " |", "|" + "---|" * (len(BUDGETS) + 1)]
    for m in ["early-probe", "cap-linear", "cap-sat"]:
        L.append(f"| {m} | " + " | ".join(f"{rmse(pse, m, f):.3f}" for f in BUDGETS) + " |")
    L += ["\nFair early-probe = last checkpoint at/before C_e (no peeking). cap-sat = logistic-in-log-",
          "compute (bounded). Proteins are gradual (linear ok); Pythia saturates (linear overshoots →",
          "saturating form needed). v1 seed bootstrap on protein ranking; Pythia n_sizes small."]
    a.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.3))
        col = {"early-probe": "#999999", "loss-rank": "#55A868", "cap-linear": "#4C72B0", "cap-sat": "#C44E52"}
        for m in ["early-probe", "loss-rank", "cap-linear", "cap-sat"]:
            axL.plot(BUDGETS, [float(np.mean(rank_ok[(m, f)])) for f in BUDGETS], "-o",
                     color=col[m], lw=1.7, ms=5, label=m)
        axL.set_xscale("log"); axL.axhline(0.5, color="0.6", ls=":", lw=0.8); axL.set_ylim(0.4, 1.03)
        axL.set_xlabel("early budget  f=C_e/C_final"); axL.set_ylabel("ranking accuracy")
        axL.set_title("Proteins: gpt2 vs mamba ranking", fontsize=9)
        axL.legend(frameon=False, fontsize=7.5, loc="lower right")
        for m in ["early-probe", "cap-linear", "cap-sat"]:
            axR.plot(BUDGETS, [rmse(pse, m, f) for f in BUDGETS], "-o", color=col[m], lw=1.7, ms=5, label=m)
        axR.set_xscale("log"); axR.set_xlabel("early budget  f=C_e/C_final")
        axR.set_ylabel("converged-capability RMSE"); axR.set_title("Pythia / NLP: generality", fontsize=9)
        axR.legend(frameon=False, fontsize=7.5)
        for i, ax in enumerate((axL, axR)):
            ax.text(-0.08, 1.04, chr(97 + i), transform=ax.transAxes, fontweight="bold", fontsize=10)
        fig.tight_layout(); figp = a.out.parent / "figures" / "forecast_v2"
        figp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(figp) + ".png", dpi=200, bbox_inches="tight"); fig.savefig(str(figp) + ".pdf", bbox_inches="tight")
        print(f"[forecast v2] wrote {figp}.png/.pdf")
    except Exception as e:
        print(f"[forecast v2] figure skipped: {e}")
    print(f"[forecast v2] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
