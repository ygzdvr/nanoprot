#!/usr/bin/env python3
"""P3.2 — Rank-reversal phenomenon (Paper B anchor).

Quantifies early-proxy mis-ranking of the AR pair (gpt2 vs mamba): at an EARLY iso-FLOP
budget the ranking can invert relative to the converged ranking. Comparisons are ISO-FLOP
(``train_flops``), never fixed-step, and seed-PAIRED (gpt2-seed-r vs mamba-seed-r) then
aggregated with a seed bootstrap.

For each task k in {val_bpr} u {capability concepts} and scale s:
  - score oriented higher=better (probe macro_f1/r2 as stored; loss -> -val_bpr)
  - per seed r, Delta_r(C) = score(gpt2,s,r,C) - score(mamba,s,r,C), each interpolated in
    log C onto the seed-pair's shared compute range
  - early budget C_e = max(min shared C, early_frac * C_f); final C_f = max shared C
  - reversal := sign(mean_r Delta_r(C_e)) != sign(mean_r Delta_r(C_f))
  - crossover C* = inf{C : mean Delta(C') > 0 for all C' >= C}, reported as C*/C_f
  - P(reversal): seed bootstrap (B draws, resample the r index with replacement)

v1 scope: within-scale AR-pair reversal on COMPLETE scales (default S,M). Deferred (noted,
not silently dropped): per-cluster block bootstrap (needs --dump-cluster-scores re-probe),
roster-level Kendall tau across scales, and the L scale (gated on P1.4 completion).

Output: docs/rank_reversal.csv + a printed headline. Every number traces to this script+seed.
"""
import argparse
import bisect
import csv
import glob
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

AR = ("gpt2", "mamba")


def load_meta(results_dir: Path) -> dict:
    """(arch, scale, seed) -> (train_flops_total, train_residues_total).

    NB: ``train_flops``/``train_residues`` are per-RUN totals repeated on every checkpoint row
    (not cumulative-at-checkpoint), so the per-checkpoint compute is recovered as
    C(t) = train_flops * step / train_residues. The common total-batch-size factor cancels,
    which is irrelevant to an iso-FLOP cross-arch comparison evaluated at a shared C.
    """
    meta = {}
    for fn in glob.glob(str(results_dir / "*.csv")):
        if Path(fn).name == "val_loss.csv":
            continue
        with open(fn) as f:
            for r in csv.DictReader(f):
                try:
                    meta[(r["arch"], r["scale"], r["seed"])] = (
                        float(r["train_flops"]), float(r["train_residues"]))
                except (KeyError, ValueError):
                    continue
    return meta


def load_task_curves(results_dir: Path, meta: dict) -> dict:
    """task -> (arch, scale, seed) -> sorted [(log C(t), score_higher_better)],
    with per-checkpoint compute C(t) = train_flops * step / train_residues (iso-FLOP)."""
    tasks = defaultdict(lambda: defaultdict(list))
    for fn in glob.glob(str(results_dir / "*.csv")):
        if Path(fn).name == "val_loss.csv":
            continue
        with open(fn) as f:
            for r in csv.DictReader(f):
                if r["arch"] not in AR:
                    continue
                try:
                    step = int(r["step"])
                    C = float(r["train_flops"]) * step / float(r["train_residues"])
                    v = float(r["learned_test"])
                except (KeyError, ValueError, ZeroDivisionError):
                    continue
                if C > 0:
                    tasks[r["concept"]][(r["arch"], r["scale"], r["seed"])].append((math.log(C), v))
    vp = results_dir / "val_loss.csv"
    if vp.exists():
        with open(vp) as f:
            for r in csv.DictReader(f):
                mkey = (r["arch"], r["scale"], r["seed"])
                if r["arch"] not in AR or mkey not in meta:
                    continue
                flops_tot, res_tot = meta[mkey]
                try:
                    step = int(r["step"])
                    C = flops_tot * step / res_tot
                    v = -float(r["val_bpr"])  # -bpr: higher=better
                except (KeyError, ValueError, ZeroDivisionError):
                    continue
                if C > 0:
                    tasks["val_bpr"][mkey].append((math.log(C), v))
    for t in tasks:
        for k in tasks[t]:
            tasks[t][k].sort()
    return tasks


def interp(curve, x: float) -> float:
    xs = [a for a, _ in curve]; ys = [b for _, b in curve]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_left(xs, x)
    t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] + t * (ys[i] - ys[i - 1])


def analyze(curves: dict, scale: str, *, early_frac: float, n_boot: int, rng: random.Random):
    """curves: (arch,scale,seed)->curve for one task. Returns reversal stats or None."""
    g_seeds = {s for (a, sc, s) in curves if a == "gpt2" and sc == scale}
    m_seeds = {s for (a, sc, s) in curves if a == "mamba" and sc == scale}
    pairs = []
    for s in sorted(g_seeds & m_seeds):
        g, m = curves[("gpt2", scale, s)], curves[("mamba", scale, s)]
        lo, hi = max(g[0][0], m[0][0]), min(g[-1][0], m[-1][0])
        if hi > lo:
            pairs.append((g, m, lo, hi))
    if not pairs:
        return None
    LO = max(p[2] for p in pairs); HI = min(p[3] for p in pairs)
    if HI <= LO:
        return None
    Cf = math.exp(HI)
    Ce = max(math.exp(LO), early_frac * Cf)
    logCe, logCf = math.log(Ce), HI
    de = [interp(g, logCe) - interp(m, logCe) for g, m, _, _ in pairs]
    df = [interp(g, logCf) - interp(m, logCf) for g, m, _, _ in pairs]
    mde, mdf = st.mean(de), st.mean(df)
    ew = "gpt2" if mde > 0 else "mamba"
    fw = "gpt2" if mdf > 0 else "mamba"
    # crossover on the seed-mean Delta, fine grid in log C
    grid = [LO + (HI - LO) * i / 200 for i in range(201)]
    dmean = [st.mean(interp(g, x) - interp(m, x) for g, m, _, _ in pairs) for x in grid]
    cross = next((math.exp(grid[i]) for i in range(len(grid))
                  if all(dmean[j] > 0 for j in range(i, len(grid)))), None)

    def boot(xs):
        out = []
        for _ in range(n_boot):
            idx = [rng.randrange(len(xs)) for _ in xs]
            out.append(st.mean(xs[i] for i in idx))
        return out

    be, bf = boot(de), boot(df)
    p_rev = sum(1 for a, b in zip(be, bf) if (a > 0) != (b > 0)) / n_boot
    be.sort(); bf.sort()
    q = lambda v, lo: v[int(lo * len(v))]
    return dict(scale=scale, n_seed=len(de), C_e=Ce, C_f=Cf,
                cross_over_frac=(cross / Cf if cross else None),
                d_early=mde, d_early_lo=q(be, .025), d_early_hi=q(be, .975),
                d_final=mdf, d_final_lo=q(bf, .025), d_final_hi=q(bf, .975),
                early_winner=ew, final_winner=fw,
                reversal=int(ew != fw), p_reversal=p_rev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path(".cache/nanoprot/trajectory_results"))
    ap.add_argument("--out", type=Path, default=Path("docs/rank_reversal.csv"))
    ap.add_argument("--scales", nargs="+", default=["S", "M"],
                    help="Complete scales only (L gated on P1.4).")
    ap.add_argument("--early-frac", type=float, default=0.03, help="C_e = early_frac * C_f.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    meta = load_meta(args.results_dir)
    tasks = load_task_curves(args.results_dir, meta)
    order = ["val_bpr", "ss3", "ss8", "rsa", "active_site", "disorder",
             "transmembrane", "signal_peptide"]
    task_keys = [t for t in order if t in tasks] + [t for t in tasks if t not in order]

    rows = []
    for task in task_keys:
        for scale in args.scales:
            res = analyze(tasks[task], scale, early_frac=args.early_frac,
                          n_boot=args.n_boot, rng=rng)
            if res:
                rows.append({"task": task, **res})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["task", "scale", "n_seed", "C_e", "C_f", "cross_over_frac",
            "d_early", "d_early_lo", "d_early_hi", "d_final", "d_final_lo", "d_final_hi",
            "early_winner", "final_winner", "reversal", "p_reversal"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})

    print(f"[rank_reversal] {len(rows)} (task,scale) cells -> {args.out}")
    print(f"{'task':14s}{'sc':>3}{'early':>7}{'final':>7}{'rev':>4}{'P(rev)':>8}{'Cx/Cf':>8}"
          f"  d_early[CI]            d_final[CI]")
    for r in rows:
        ce = f"[{r['d_early_lo']:+.3f},{r['d_early_hi']:+.3f}]"
        cf = f"[{r['d_final_lo']:+.3f},{r['d_final_hi']:+.3f}]"
        cx = f"{r['cross_over_frac']:.3f}" if r["cross_over_frac"] is not None else "  -  "
        print(f"{r['task']:14s}{r['scale']:>3}{r['early_winner']:>7}{r['final_winner']:>7}"
              f"{r['reversal']:>4}{r['p_reversal']:>8.3f}{cx:>8}  {r['d_early']:+.3f}{ce:18s}"
              f"  {r['d_final']:+.3f}{cf}")
    print("\nNOTE: v1 seed bootstrap (n_seed<=3). Deferred: per-cluster block bootstrap "
          "(--dump-cluster-scores re-probe), roster Kendall tau, L scale (gated on P1.4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
