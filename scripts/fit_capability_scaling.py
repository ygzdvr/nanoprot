#!/usr/bin/env python3
"""P3.3 — hierarchical capability-scaling: emergence-time N-scaling + the inductive-bias coefficient.

Corrected after a sanity audit of a first (logistic) attempt that failed twice:
  - The logistic asymptote is UNIDENTIFIABLE for protein curves that are still rising at the end
    (not saturated) → c0/S_inf are nonsense for parameter extraction. (The logistic is kept only
    for FORECASTING in forecast.py, where it is validated and the target is the extrapolated value,
    not the parameters.) Here we use ROBUST, model-light observed quantities instead.
  - The capability response must be Δ = learned − random-init baseline (LEARNED structure), not
    absolute decodability: absolute conflates the baseline (mamba's random-init reps are more
    decodable), which flipped the inductive-bias sign. We use Δ ("delta" column) throughout.

Components (scipy/numpy; reproducible):
  (1–2) Per (concept, arch, scale): emergence time t50 = log-compute where Δ reaches ½·Δ_final
        (first crossing, interpolated) and asymptote Δ_final. Seed mean + seed-bootstrap CI.
        Regress t50 and Δ_final on logN per (concept,arch) → emergence-time & asymptote scaling.
  (3)   Inductive bias: per concept, OLS  Δ ~ β0 + β_L·(−bpr) + scale-FE + γ_arch·[gpt2]  over all
        AR-pair checkpoints. γ_arch>0 with H_full≻H_loss (ΔAIC) ⇒ gpt2 exposes more LEARNED
        structure at matched (measured) loss & scale — slice-robust. Cluster bootstrap (scale×seed)
        gives the γ_arch CI. Forest figure.
"""
import argparse
import csv
import glob
import math
from collections import defaultdict
from pathlib import Path
import importlib.util

import numpy as np

_spec = importlib.util.spec_from_file_location("rr", str(Path(__file__).resolve().parent / "rank_reversal.py"))
rr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rr)

CONCEPTS = ["ss3", "ss8", "rsa", "active", "disorder"]
SCALES = ["S", "M", "L"]
LOGN = {"S": math.log(36e6), "M": math.log(143e6), "L": math.log(1.0e9)}


def load_delta(results_dir):
    """(concept) -> (arch,scale,seed) -> sorted [(logC, delta)] from the concept CSVs."""
    out = defaultdict(lambda: defaultdict(list))
    for fn in glob.glob(str(results_dir / "*.csv")):
        if Path(fn).name == "val_loss.csv":
            continue
        with open(fn) as f:
            for r in csv.DictReader(f):
                if r["arch"] not in ("gpt2", "mamba"):
                    continue
                try:
                    step = int(r["step"]); C = float(r["train_flops"]) * step / float(r["train_residues"])
                    d = float(r["delta"])
                except (KeyError, ValueError, ZeroDivisionError):
                    continue
                if C > 0:
                    out[r["concept"]][(r["arch"], r["scale"], r["seed"])].append((math.log(C), d))
    for k in out:
        for key in out[k]:
            out[k][key].sort()
    return out


def t50_logC(curve, frac=0.5):
    """First log-compute at which Δ reaches frac·Δ_final (interpolated). Robust to non-saturation."""
    x = [a for a, _ in curve]; y = [b for _, b in curve]
    yf = float(np.mean(y[-2:])) if len(y) >= 2 else y[-1]
    if yf <= 0:
        return None
    tgt = frac * yf
    for i in range(1, len(y)):
        if y[i - 1] < tgt <= y[i] or (y[i] >= tgt and y[i - 1] >= tgt and i == 1):
            if y[i] == y[i - 1]:
                return x[i]
            t = (tgt - y[i - 1]) / (y[i] - y[i - 1])
            return x[i - 1] + t * (x[i] - x[i - 1])
    return x[-1]


def boot(vals, n=2000, seed=0):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bt = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return float(v.mean()), float(np.quantile(bt, .025)), float(np.quantile(bt, .975))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path(".cache/nanoprot/trajectory_results"))
    ap.add_argument("--out", type=Path, default=Path("docs/capability_scaling_report.md"))
    a = ap.parse_args()
    delta = load_delta(a.results_dir)
    meta = rr.load_meta(a.results_dir); tasks = rr.load_task_curves(a.results_dir, meta)  # for val_bpr

    # ---- Stage 1-2: robust emergence time + asymptote, and N-scaling ----
    cell = {}
    for k in CONCEPTS:
        if k not in delta:
            continue
        for arch in ("gpt2", "mamba"):
            for sc in SCALES:
                seeds = [delta[k][(aa, s2, se)] for (aa, s2, se) in delta[k] if aa == arch and s2 == sc]
                if len(seeds) < 2:
                    continue
                t50 = [t50_logC(c) for c in seeds]; df = [float(np.mean([y for _, y in c][-2:])) for c in seeds]
                cell[(k, arch, sc)] = dict(t50=boot(t50), dfin=boot(df))

    L = ["# P3.3 — capability-scaling: emergence-time N-scaling + inductive-bias coefficient\n",
         "Δ = learned − random-init (learned structure); t50 = log-compute at ½·Δ_final (observed,",
         "no saturation assumption); LOGN(S,M,L)=(17.4,18.8,20.7). Seed-bootstrap CIs.\n",
         "## 1–2. Emergence time t50 and asymptote Δ_final by scale; slope vs logN",
         "concept · arch | t50 @S/M/L | dt50/dlogN | Δ_final @S/M/L | dΔ/dlogN"]
    for k in CONCEPTS:
        for arch in ("gpt2", "mamba"):
            row = [cell.get((k, arch, sc)) for sc in SCALES]
            if not all(row):
                continue
            t = [r["t50"][0] for r in row]; df = [r["dfin"][0] for r in row]
            xs = np.array([LOGN[sc] for sc in SCALES])
            dt = float(np.polyfit(xs, t, 1)[0]); dd = float(np.polyfit(xs, df, 1)[0])
            L.append(f"{k:>9} · {arch:5} | {t[0]:.1f}/{t[1]:.1f}/{t[2]:.1f} | {dt:+.2f} | "
                     f"{df[0]:.3f}/{df[1]:.3f}/{df[2]:.3f} | {dd:+.3f}")

    # ---- Stage 3: inductive-bias coefficient on Δ (matched loss & scale) ----
    L += ["\n## 3. Inductive bias: Δ ~ β0 + β_L·(−bpr) + scale-FE + γ_arch·[gpt2]  (AR pair, all ckpts)",
          "γ_arch>0 ⇒ gpt2 has MORE learned structure than mamba at matched measured-loss & scale.",
          "concept | γ_arch [95% CI] | β_L | ΔAIC(H_loss−H_full) | n | H_full wins?"]
    gamma_rows = []
    for k in CONCEPTS:
        if k not in delta:
            continue
        X = []; Y = []; clu = []
        for (aa, sc, se) in delta[k]:
            loss = tasks["val_bpr"].get((aa, sc, se))
            if not loss:
                continue
            for (lc, d) in delta[k][(aa, sc, se)]:
                Y.append(d)
                X.append([1.0, rr.interp(loss, lc), 1.0 if sc == "M" else 0.0,
                          1.0 if sc == "L" else 0.0, 1.0 if aa == "gpt2" else 0.0])
                clu.append((sc, se))
        if len(Y) < 20:
            continue
        X = np.array(X); Y = np.array(Y)
        def ols(Xm):
            b, *_ = np.linalg.lstsq(Xm, Y, rcond=None)
            r = Y - Xm @ b; rss = float(r @ r); n = len(Y)
            return b, n * math.log(rss / n) + 2 * Xm.shape[1]
        b_full, aic_full = ols(X)
        _, aic_loss = ols(X[:, [0, 1, 2, 3]])  # drop arch dummy
        ga = float(b_full[4]); bL = float(b_full[1])
        cls = sorted(set(clu)); by = defaultdict(list)
        for i, c in enumerate(clu):
            by[c].append(i)
        rng = np.random.default_rng(0); gb = []
        for _ in range(1000):
            ii = [i for j in rng.integers(0, len(cls), len(cls)) for i in by[cls[j]]]
            try:
                bb, *_ = np.linalg.lstsq(X[ii], Y[ii], rcond=None); gb.append(float(bb[4]))
            except Exception:
                pass
        glo, ghi = (float(np.quantile(gb, .025)), float(np.quantile(gb, .975))) if gb else (math.nan, math.nan)
        win = "yes" if (aic_full < aic_loss - 2 and (glo > 0 or ghi < 0)) else "no"
        gamma_rows.append((k, ga, glo, ghi))
        L.append(f"{k:>9} | {ga:+.3f} [{glo:+.3f},{ghi:+.3f}] | {bL:+.3f} | {aic_loss - aic_full:+.1f} | {len(Y)} | {win}")
    L += ["\nDriving by MEASURED loss makes γ_arch robust to the mamba replicated-slice offset. NUTS",
          "cross-check + the L(N,D) α/β surface (non-identifiable from the noisy sweep) remain TODO."]
    a.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                             "axes.spines.top": False, "axes.spines.right": False})
        gr = [g for g in gamma_rows if np.isfinite(g[2])]
        fig, ax = plt.subplots(figsize=(4.3, 0.5 * len(gr) + 1.1))
        for y, (k, g, lo, hi) in enumerate(gr):
            c = "#C44E52" if lo > 0 else ("#4C72B0" if hi < 0 else "0.6")
            ax.plot([lo, hi], [y, y], color=c, lw=2); ax.plot([g], [y], "o", color=c, ms=6)
        ax.axvline(0, color="0.4", ls="--", lw=0.8)
        ax.set_yticks(range(len(gr))); ax.set_yticklabels([g[0] for g in gr])
        ax.set_xlabel("γ_arch  (gpt2 learned-Δ advantage at matched loss & scale)")
        fig.tight_layout()
        figp = a.out.parent / "figures" / "inductive_bias_forest"
        figp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(figp) + ".png", dpi=200, bbox_inches="tight"); fig.savefig(str(figp) + ".pdf", bbox_inches="tight")
        print(f"[capability_scaling] wrote {figp}.png/.pdf")
    except Exception as e:
        print(f"[capability_scaling] figure skipped: {e}")
    print(f"[capability_scaling] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
