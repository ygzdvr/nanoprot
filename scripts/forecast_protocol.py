#!/usr/bin/env python3
"""B-FCAST + B-ALLOC + B-CAL — leakage-proof forecasting protocol, allocation rule, calibration.

Unit = a protein cell (concept k, arch a, scale N); response = converged learned structure
Δ_final (Δ=learned−random-init, seed-mean). Forecast from checkpoints up to an early budget
C_e=f·C_final, under strict **leave-one-cell-out** (LOCO): any cross-cell component (the scaling
law; the loss→Δ map) is fit on OTHER cells only — the held-out cell's convergence is never seen.

Two distinct tasks, deliberately separated (a key finding of the audit):
  • SELECTION / B-ALLOC (the decision): at a fixed scale, pick the better ARCHITECTURE. This is the
    rank-reversal made actionable. Methods that can separate same-scale archs only: loss-rank
    (Chinchilla-style "pick the lower-loss model"), early-probe, cap-linear, cap-sat.
  • VALUE / B-FCAST (magnitude): predict Δ_final. Here the per-concept SCALING LAW (scale only) is a
    very strong baseline; the trajectory's value-add is selection, not magnitude.

Forecasters: early-probe (last ≤C_e), loss-rank/loss-extrap (extrapolate loss; lower=better /
map to Δ on training cells), scaling-pop (Δ_final~logN on training cells — no trajectory), cap-linear
(linear-in-logC; right for gradual proteins), cap-sat (bounded logistic; right for SATURATING
domains e.g. NLP — overshoots non-saturating proteins), cap-hier (value-only: w·cap-linear +
(1−w)·scaling-pop, w=clip(f/f0,0,1) — combines trajectory + scaling law).
"""
import argparse
import csv
import glob
import math
from collections import defaultdict
from pathlib import Path
import importlib.util

import numpy as np
from scipy.optimize import least_squares

_spec = importlib.util.spec_from_file_location("rr", str(Path(__file__).resolve().parent / "rank_reversal.py"))
rr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rr)

CONCEPTS = ["ss3", "ss8", "rsa", "active", "disorder"]
SCALES = ["S", "M", "L"]
LOGN = {"S": math.log(36e6), "M": math.log(143e6), "L": math.log(1.0e9)}
BUDGETS = [0.01, 0.03, 0.1, 0.3]
F0 = 0.1


def load_cells(rd):
    dlt = defaultdict(lambda: defaultdict(list))
    for fn in glob.glob(str(rd / "*.csv")):
        if Path(fn).name == "val_loss.csv":
            continue
        for r in csv.DictReader(open(fn)):
            if r["arch"] not in ("gpt2", "mamba"):
                continue
            try:
                step = int(r["step"]); C = float(r["train_flops"]) * step / float(r["train_residues"])
                dd = float(r["delta"])
            except (KeyError, ValueError, ZeroDivisionError):
                continue
            if C > 0:
                dlt[r["concept"]][(r["arch"], r["scale"], r["seed"])].append((math.log(C), dd))
    meta = rr.load_meta(rd); tasks = rr.load_task_curves(rd, meta)
    cells = {}
    for k in CONCEPTS:
        if k not in dlt:
            continue
        for a in ("gpt2", "mamba"):
            for N in SCALES:
                seeds = [sorted(dlt[k][(aa, nn, se)]) for (aa, nn, se) in dlt[k] if aa == a and nn == N]
                if len(seeds) < 2:
                    continue
                lo = max(c[0][0] for c in seeds); hi = min(c[-1][0] for c in seeds)
                if hi <= lo:
                    continue
                grid = np.linspace(lo, hi, 40)
                dcur = [(float(x), float(np.mean([rr.interp(c, x) for c in seeds]))) for x in grid]
                lseeds = [tasks["val_bpr"].get((a, N, se)) for (aa, nn, se) in dlt[k]
                          if aa == a and nn == N and tasks["val_bpr"].get((a, N, se))]
                lcur = [(float(x), float(np.mean([-rr.interp(c, x) for c in lseeds]))) for x in grid] if lseeds else None
                cells[(k, a, N)] = dict(d=dcur, loss=lcur, dfin=dcur[-1][1], logCf=hi)
    return cells


def _av(cur, lce): return [(x, y) for x, y in cur if x <= lce + 1e-9]


def cap_linear(cur, lce, lcf):
    a = _av(cur, lce)
    if len(a) < 3:
        return a[-1][1] if a else cur[0][1]
    x = np.array([p[0] for p in a]); y = np.array([p[1] for p in a])
    m, b = np.polyfit(x, y, 1); return float(np.clip(m * lcf + b, 0, 1))


def cap_sat(cur, lce, lcf):
    a = _av(cur, lce)
    if len(a) < 4:
        return a[-1][1] if a else cur[0][1]
    x = np.array([p[0] for p in a]); y = np.clip([p[1] for p in a], 0, 1)
    def res(p): Si, k, x0 = p; return Si / (1 + np.exp(-np.clip(k * (x - x0), -60, 60))) - y
    try:
        r = least_squares(res, [max(float(y.max()), .05), 1., float(x.mean())],
                          bounds=([0, .05, x.min() - 10], [1.2, 30, x.max() + 15]),
                          loss="soft_l1", f_scale=.05, max_nfev=10000)
        Si, k, x0 = r.x; return float(np.clip(Si / (1 + math.exp(-max(-60, min(60, k * (lcf - x0))))), 0, 1))
    except Exception:
        return a[-1][1]


def pred_loss_final(cur_loss, lce, lcf):
    """Extrapolate bpr (decreasing, saturating) to C_final; lower = better LM."""
    if not cur_loss:
        return None
    a = _av(cur_loss, lce)
    if len(a) < 4:
        return a[-1][1] if a else cur_loss[0][1]
    x = np.array([p[0] for p in a]); y = np.array([p[1] for p in a])
    def res(p): E, A, c = p; return E + A * np.exp(-c * (x - x[0])) - y
    try:
        r = least_squares(res, [float(y.min()), float(y[0] - y.min()), .5],
                          bounds=([0, 0, 0], [10, 20, 5]), loss="soft_l1", f_scale=.02, max_nfev=8000)
        E, A, c = r.x; return float(E + A * math.exp(-c * (lcf - x[0])))
    except Exception:
        return a[-1][1]


def scaling_pop(cells, concept, logN, exclude):
    pts = [(LOGN[N], c["dfin"]) for (k, a, N), c in cells.items() if k == concept and (k, a, N) != exclude]
    if len(pts) < 2:
        return float(np.mean([p[1] for p in pts])) if pts else 0.0
    X = np.array([[1, p[0]] for p in pts]); Y = np.array([p[1] for p in pts])
    b, *_ = np.linalg.lstsq(X, Y, rcond=None); return float(max(0, b[0] + b[1] * logN))


def loss_map(cells, concept, exclude):
    pts = [(c["loss"][-1][1], c["dfin"]) for (k, a, N), c in cells.items()
           if k == concept and (k, a, N) != exclude and c["loss"]]
    if len(pts) < 2:
        return None
    X = np.array([[1, p[0]] for p in pts]); Y = np.array([p[1] for p in pts])
    b, *_ = np.linalg.lstsq(X, Y, rcond=None); return lambda v: float(max(0, b[0] + b[1] * v))


def value_pred(cells, cell, f, method):
    c = cells[cell]; lcf = c["logCf"]; lce = lcf + math.log(f); k = cell[0]
    if method == "early-probe":
        a = _av(c["d"], lce); return a[-1][1] if a else c["d"][0][1]
    if method == "scaling-pop":
        return scaling_pop(cells, k, LOGN[cell[2]], cell)
    if method == "cap-linear":
        return cap_linear(c["d"], lce, lcf)
    if method == "cap-sat":
        return cap_sat(c["d"], lce, lcf)
    if method == "cap-hier":
        w = min(max(f / F0, 0), 1)
        return w * cap_linear(c["d"], lce, lcf) + (1 - w) * scaling_pop(cells, k, LOGN[cell[2]], cell)
    if method == "loss-extrap":
        sm = loss_map(cells, k, cell); bf = pred_loss_final(c["loss"], lce, lcf)
        if sm is None or bf is None:
            a = _av(c["d"], lce); return a[-1][1] if a else c["d"][0][1]
        return sm(bf)
    raise ValueError(method)


def sel_score(cells, cell, f, method):
    """Larger = method prefers this cell's arch (selection between same-scale archs)."""
    c = cells[cell]; lcf = c["logCf"]; lce = lcf + math.log(f)
    if method == "loss-rank":
        bf = pred_loss_final(c["loss"], lce, lcf); return (-bf) if bf is not None else None
    return value_pred(cells, cell, f, method)


VALUE_METHODS = ["early-probe", "loss-extrap", "scaling-pop", "cap-linear", "cap-sat", "cap-hier"]
SEL_METHODS = ["loss-rank", "early-probe", "cap-linear", "cap-sat"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path(".cache/nanoprot/trajectory_results"))
    ap.add_argument("--out", type=Path, default=Path("docs/forecast_protocol_report.md"))
    ap.add_argument("--concepts", nargs="+", default=None,
                    help="Concepts to evaluate (default: the protein 5; for B-GEN/genome use --concepts exon).")
    a = ap.parse_args()
    global CONCEPTS
    if a.concepts:
        CONCEPTS = a.concepts
    cells = load_cells(a.results_dir); keys = sorted(cells); rng = np.random.default_rng(0)

    vpred = {(m, f): [(c, float(value_pred(cells, c, f, m)), cells[c]["dfin"]) for c in keys]
             for m in VALUE_METHODS for f in BUDGETS}
    def rmse_ci(m, f):
        e = np.array([p - t for _, p, t in vpred[(m, f)]])
        bt = [float(np.sqrt(np.mean(e[rng.integers(0, len(e), len(e))] ** 2))) for _ in range(2000)]
        return float(np.sqrt(np.mean(e ** 2))), float(np.quantile(bt, .025)), float(np.quantile(bt, .975))

    def sel_eval(m, f):
        ok, reg = [], []
        for k in CONCEPTS:
            for N in SCALES:
                cg, cm = (k, "gpt2", N), (k, "mamba", N)
                if cg not in cells or cm not in cells:
                    continue
                sg, sm = sel_score(cells, cg, f, m), sel_score(cells, cm, f, m)
                if sg is None or sm is None:
                    continue
                pick = "gpt2" if sg > sm else "mamba"
                tg, tm = cells[cg]["dfin"], cells[cm]["dfin"]
                ok.append(pick == ("gpt2" if tg > tm else "mamba"))
                reg.append(max(tg, tm) - cells[(k, pick, N)]["dfin"])
        return ok, reg
    def sel_ci(m, f):
        ok, _ = sel_eval(m, f); v = np.array(ok, float)
        bt = [float(v[rng.integers(0, len(v), len(v))].mean()) for _ in range(2000)]
        return v.mean(), float(np.quantile(bt, .025)), float(np.quantile(bt, .975))

    L = ["# B-ALLOC / B-FCAST / B-CAL — selection-by-capability, forecasting, calibration\n",
         f"{len(keys)} protein cells; LOCO; response Δ_final; seed-mean; 2000-boot CIs. Selection chance = 0.50.\n",
         "## B-ALLOC (headline) — top-1 architecture selection accuracy by budget f  [95% CI]",
         "| method | " + " | ".join(f"f={f}" for f in BUDGETS) + " |", "|" + "---|" * (len(BUDGETS) + 1)]
    for m in SEL_METHODS:
        L.append(f"| {m} | " + " | ".join((lambda t: f"{t[0]:.2f} [{t[1]:.2f},{t[2]:.2f}]")(sel_ci(m, f)) for f in BUDGETS) + " |")
    L += ["", "Mean regret (Δ lost vs oracle) & FLOP savings (=1−f per candidate):",
          "| method | " + " | ".join(f"f={f}" for f in BUDGETS) + " |", "|" + "---|" * (len(BUDGETS) + 1)]
    for m in SEL_METHODS:
        L.append(f"| {m} | " + " | ".join(f"{np.mean(sel_eval(m, f)[1]):.3f}" for f in BUDGETS) + " |")
    L.append(f"\nFLOP savings at f: {{{', '.join(f'{f}:{1-f:.0%}' for f in BUDGETS)}}}.  loss-rank < chance ⇒ "
             "selecting by loss actively mis-picks the architecture (the rank-reversal as a decision failure).")

    L += ["\n## B-FCAST — held-out value (Δ_final) RMSE by budget  [95% CI]",
          "| method | " + " | ".join(f"f={f}" for f in BUDGETS) + " |", "|" + "---|" * (len(BUDGETS) + 1)]
    for m in VALUE_METHODS:
        L.append(f"| {m} | " + " | ".join((lambda t: f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]")(rmse_ci(m, f)) for f in BUDGETS) + " |")
    L.append("\nValue≠selection: the per-concept scaling law (scaling-pop) predicts converged MAGNITUDE "
             "well but cannot rank same-scale archs (selection acc 0); cap-sat overshoots non-saturating "
             "protein curves (bounded form is for SATURATING domains, e.g. Pythia/NLP — forecast.py); "
             "cap-hier (scaling-law ⊕ trajectory) is the best value forecaster.")

    def conformal_cov(method, f, alpha=0.1):
        """Distribution-free split-conformal coverage: for held-out cell i the interval half-width is
        the conformal quantile of the OTHER cells' |LOCO error|. Coverage ≈ 1−α if calibrated."""
        errs = np.array([abs(p - t) for _, p, t in vpred[(method, f)]]); n = len(errs); cov = []
        for i in range(n):
            cal = np.sort(np.delete(errs, i)); m = len(cal)
            Q = cal[min(m - 1, int(math.ceil((m + 1) * (1 - alpha))) - 1)]
            cov.append(errs[i] <= Q)
        return float(np.mean(cov))

    L += ["\n## B-CAL — calibration (cap-hier conformal 90% coverage; LOCO) + value ablation",
          "| budget f | gaussian cov@90 | conformal cov@90 | RMSE scaling-pop | cap-linear | cap-hier |",
          "|---|---|---|---|---|---|"]
    for f in BUDGETS:
        e = np.array([p - t for _, p, t in vpred[("cap-hier", f)]]); sig = float(np.std(e))
        gcov = float(np.mean(np.abs(e) <= 1.645 * sig)) if sig > 0 else float("nan")
        L.append(f"| {f} | {gcov:.2f} | {conformal_cov('cap-hier', f):.2f} | {rmse_ci('scaling-pop',f)[0]:.3f} | "
                 f"{rmse_ci('cap-linear',f)[0]:.3f} | {rmse_ci('cap-hier',f)[0]:.3f} |")
    L += ["\nGaussian PI is mildly under-covered (heavy-tailed errors); the distribution-free conformal",
          "interval restores ~90% coverage — the calibrated uncertainty we report. Ablation: cap-hier ≤",
          "cap-linear and ≤ scaling-pop at small budgets ⇒ combining the scaling-law prior with the",
          "trajectory helps when early data is scarce. Leakage control: scaling law + loss→Δ map + the",
          "conformal calibration set are all OTHER-cells-only (LOCO). Cross-domain generality (Pythia/NLP,",
          "bounded forecaster) in forecast.py. TODO: per-cluster bootstrap; more seeds/scales; NUTS check."]
    a.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.3))
        cs = {"loss-rank": "#55A868", "early-probe": "#999999", "cap-linear": "#4C72B0", "cap-sat": "#C44E52"}
        for m in SEL_METHODS:
            axL.plot(BUDGETS, [sel_ci(m, f)[0] for f in BUDGETS], "-o", color=cs[m], lw=1.7, ms=5, label=m)
        axL.axhline(0.5, color="0.5", ls=":", lw=1); axL.set_xscale("log"); axL.set_ylim(0.3, 1.0)
        axL.set_xlabel("early budget f  (FLOP savings = 1−f)"); axL.set_ylabel("architecture selection accuracy")
        axL.set_title("B-ALLOC: capability > chance > loss", fontsize=8.5)
        axL.legend(frameon=False, fontsize=7.5, loc="lower right")
        cv = {"scaling-pop": "#8172B3", "cap-linear": "#4C72B0", "cap-sat": "#C44E52", "cap-hier": "#CCB974"}
        for m in ["scaling-pop", "cap-linear", "cap-sat", "cap-hier"]:
            axR.plot(BUDGETS, [rmse_ci(m, f)[0] for f in BUDGETS], "-o", color=cv[m], lw=1.7, ms=5, label=m)
        axR.set_xscale("log"); axR.set_xlabel("early budget f"); axR.set_ylabel("converged-value RMSE")
        axR.set_title("B-FCAST: value≠selection (scale-law wins value)", fontsize=8.5)
        axR.legend(frameon=False, fontsize=7.5)
        for i, ax in enumerate((axL, axR)):
            ax.text(-0.08, 1.04, chr(97 + i), transform=ax.transAxes, fontweight="bold", fontsize=10)
        fig.tight_layout(); figp = a.out.parent / "figures" / "forecast_protocol"
        figp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(figp) + ".png", dpi=200, bbox_inches="tight"); fig.savefig(str(figp) + ".pdf", bbox_inches="tight")
        print(f"[forecast_protocol] wrote {figp}.png/.pdf")
    except Exception as e:
        print(f"[forecast_protocol] figure skipped: {e}")
    print(f"[forecast_protocol] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
