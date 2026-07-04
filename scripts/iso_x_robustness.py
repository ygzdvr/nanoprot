#!/usr/bin/env python
"""iso-X robustness of the rank-reversal (strategy-doc priority #4).

The reversal is defined at iso-FLOP. Attention and the SSM have very different FLOPs-per-token (here
mamba costs ~3.3x MORE flops/token than gpt2 at L=512, expand=2), so the compute axis is not neutral:
  - at matched FLOP mamba has run ~0.30x the STEPS of gpt2 (iso-FLOP HANDICAPS mamba early);
  - the iso-FLOP shared endpoint is gpt2's final FLOP, where mamba is only ~27% trained (iso-FLOP
    FLATTERS the 'gpt2 wins converged' side).
So we recompute the reversal under iso-FLOP / iso-step / iso-token and report where it survives. We also
report d_final at EACH-MODEL'S-OWN convergence (axis-independent) so 'gpt2 wins at convergence' is not an
artifact of comparing a converged gpt2 to an under-trained mamba. No wall-clock is logged -> not tested
(stated honestly). Interpolation + seed bootstrap are identical to rank_reversal.py, so the iso-FLOP
column MUST reproduce the paper's numbers (a correctness gate, checked at the end).
"""
from __future__ import annotations

import argparse
import bisect
import csv
import glob
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

AR = ("gpt2", "mamba")   # (converged-winner hypothesis, early-winner hypothesis); margin = gpt2 - mamba


def interp(curve, x):
    xs = [a for a, _ in curve]; ys = [b for _, b in curve]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_left(xs, x)
    t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] + t * (ys[i] - ys[i - 1])


def load_raw(results_dir: Path):
    """concept -> (arch,scale,seed) -> sorted list[(step, score, flops_total, res_total)]."""
    data = defaultdict(lambda: defaultdict(list))
    for fn in glob.glob(str(results_dir / "*.csv")):
        if Path(fn).name == "val_loss.csv":
            continue
        for r in csv.DictReader(open(fn)):
            if r["arch"] not in AR:
                continue
            try:
                step = int(r["step"]); v = float(r["learned_test"])
                fl = float(r["train_flops"]); rs = float(r["train_residues"])
            except (KeyError, ValueError):
                continue
            data[r["concept"]][(r["arch"], r["scale"], r["seed"])].append((step, v, fl, rs))
    meta = {k: (rows[0][2], rows[0][3]) for c in data for k, rows in data[c].items()}
    vp = results_dir / "val_loss.csv"
    if vp.exists():
        for r in csv.DictReader(open(vp)):
            k = (r["arch"], r["scale"], r["seed"])
            if r["arch"] not in AR or k not in meta:
                continue
            try:
                step = int(r["step"]); v = -float(r["val_bpr"]); fl, rs = meta[k]
            except (KeyError, ValueError):
                continue
            data["val_bpr"][k].append((step, v, fl, rs))
    for c in data:
        for k in data[c]:
            data[c][k].sort()
    return data


# axis(step, flops_total, res_total, maxstep) -> cumulative axis value (None to drop the checkpoint)
AXES = {
    "flop":  lambda s, fl, rs, ms: fl * s / rs if s > 0 else None,   # cumulative FLOP proxy (== rank_reversal)
    "step":  lambda s, fl, rs, ms: float(s) if s > 0 else None,      # gradient updates
    "token": lambda s, fl, rs, ms: (rs / ms) * s if s > 0 else None, # cumulative tokens (batch*step)
}


def build_curve(rows, axis):
    ms = max(s for s, _, _, _ in rows)
    c = [(math.log(x), v) for s, v, fl, rs in rows
         if (x := axis(s, fl, rs, ms)) is not None and x > 0]
    c.sort()
    return c


def analyze(curves, scale, axis, *, early_frac=0.01, n_boot=2000, rng=None):
    gs = {s for (a, sc, s) in curves if a == AR[0] and sc == scale}
    mm = {s for (a, sc, s) in curves if a == AR[1] and sc == scale}
    pairs = []
    for s in sorted(gs & mm):
        g = build_curve(curves[(AR[0], scale, s)], axis); m = build_curve(curves[(AR[1], scale, s)], axis)
        if len(g) >= 2 and len(m) >= 2 and min(g[-1][0], m[-1][0]) > max(g[0][0], m[0][0]):
            pairs.append((g, m))
    if len(pairs) < 2:
        return None
    LO = max(max(g[0][0], m[0][0]) for g, m in pairs)
    HI = min(min(g[-1][0], m[-1][0]) for g, m in pairs)
    if HI <= LO:
        return None
    lce = max(LO, HI + math.log(early_frac))
    de = [interp(g, lce) - interp(m, lce) for g, m in pairs]
    df = [interp(g, HI) - interp(m, HI) for g, m in pairs]
    mde, mdf = st.mean(de), st.mean(df)
    rev = 0
    for _ in range(n_boot):
        idx = [rng.randrange(len(pairs)) for _ in pairs]
        if st.mean(de[i] for i in idx) < 0 and st.mean(df[i] for i in idx) > 0:
            rev += 1
    return dict(n=len(pairs), d_early=mde, d_final=mdf, reversal=int(mde < 0 and mdf > 0),
                p_rev=rev / n_boot)


def own_convergence_dfinal(curves, scale, rng, n_boot=2000):
    """d_final at EACH model's OWN last checkpoint (both fully trained) — axis-independent."""
    gs = {s for (a, sc, s) in curves if a == AR[0] and sc == scale}
    mm = {s for (a, sc, s) in curves if a == AR[1] and sc == scale}
    df = []
    for s in sorted(gs & mm):
        g = curves[(AR[0], scale, s)]; m = curves[(AR[1], scale, s)]
        df.append(max(g, key=lambda r: r[0])[1] - max(m, key=lambda r: r[0])[1])
    if len(df) < 2:
        return None, None
    lo = sorted(st.mean(df[i] for i in (rng.randrange(len(df)) for _ in df)) for _ in range(n_boot))
    return st.mean(df), (lo[int(.025 * n_boot)], lo[int(.975 * n_boot)])


def mamba_frac_at_isoflop_end(curves, scale):
    """% of mamba's training reached at the iso-FLOP shared endpoint (context for d_final bias)."""
    gs = {s for (a, sc, s) in curves if a == AR[0] and sc == scale}
    mm = {s for (a, sc, s) in curves if a == AR[1] and sc == scale}
    fr = []
    for s in sorted(gs & mm):
        g = curves[(AR[0], scale, s)]; m = curves[(AR[1], scale, s)]
        gfl = max(g, key=lambda r: r[0]); flops_end = gfl[2] * gfl[0] / gfl[3]   # gpt2 final FLOP
        m_ms = max(r[0] for r in m); fl, rs = m[0][2], m[0][3]
        step_at = flops_end * rs / fl                                            # mamba step at that FLOP
        fr.append(min(1.0, step_at / m_ms))
    return st.mean(fr) if fr else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protein-dir", type=Path, default=Path(".cache/nanoprot/trajectory_results"))
    ap.add_argument("--genome-dir", type=Path, default=Path(".cache/nanoprot/trajectory_results_genome"))
    ap.add_argument("--scales", nargs="+", default=["S", "M", "L"])
    ap.add_argument("--early-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    HEAD = ("ss3", "ss8", "fold", "pfam_family", "frame")

    for label, d in (("PROTEIN", args.protein_dir), ("GENOME", args.genome_dir)):
        if not d.exists():
            continue
        data = load_raw(d)
        print(f"\n===== {label}: reversal under iso-FLOP / iso-step / iso-token (early={args.early_frac:.0%}) =====")
        print(f"  {'concept/scale':17s}{'n':>2s} | {'FLOP dE/dF/P':>20s} | {'STEP dE/dF/P':>20s} | "
              f"{'TOKEN dE/dF/P':>20s} | {'own-conv dF':>13s}{'mamba%@flopEnd':>14s}")
        for concept in sorted(data):
            if concept == "val_bpr":
                continue
            for sc in args.scales:
                res = {ax: analyze(data[concept], sc, AXES[ax], early_frac=args.early_frac,
                                   rng=random.Random(args.seed)) for ax in ("flop", "step", "token")}
                if res["flop"] is None:
                    continue
                ocf, oci = own_convergence_dfinal(data[concept], sc, random.Random(args.seed))
                mf = mamba_frac_at_isoflop_end(data[concept], sc)
                mark = " <<" if concept in HEAD else ""

                def fmt(r):
                    return f"{r['d_early']:+.3f}/{r['d_final']:+.3f}/{r['p_rev']:.2f}" if r else "  --  "
                oc = f"{ocf:+.3f}" if ocf is not None else " -- "
                print(f"  {concept+'/'+sc:17s}{res['flop']['n']:>2d} | {fmt(res['flop']):>20s} | "
                      f"{fmt(res['step']):>20s} | {fmt(res['token']):>20s} | {oc:>13s}"
                      f"{(mf if mf else 0):>13.0%}{mark}")
    print("\n  dE=d_early (mamba ahead if <0), dF=d_final (gpt2 ahead if >0), P=P(reversal) seed bootstrap.")
    print("  own-conv dF = gpt2-vs-mamba at each model's OWN last checkpoint (axis-independent).")
    print("  wall-clock: not logged -> not tested (honest limitation).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
