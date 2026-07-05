#!/usr/bin/env python
"""Recompute the rank-reversal on CORRECTED FLOPs (fixing the mamba scan seq_len bug).

The logged train_flops used mamba.estimate_flops() with a spurious seq_len factor in scan_flops, inflating
mamba's FLOPs/token ~2.6x. Here we recompute each mamba cell's corrected FLOPs/token from its checkpoint
meta -- corrected_fpt = fpt_buggy - scan_bug*(L-1)/L, scan_bug = 6*(expand*d_model)*d_state*L*n_layer --
with a CORRECTNESS GATE (6*n_params + scan_bug must reproduce the logged flops_per_token). gpt2 is
unchanged (its attention L-factor is legitimate). We then re-run the reversal under three compute axes:
  buggy-FLOP (== the paper's iso-FLOP), corrected-FLOP (the fix), and step (FLOP-independent ground truth).
Prediction: corrected-FLOP ~= step (mamba~=gpt2 in cost), so the reversal picture becomes the honest one.
Interpolation + seed bootstrap are identical to rank_reversal.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

from scripts.iso_x_robustness import AR, interp, load_raw


def corrected_fpt_lookup(release_dir: Path):
    """(arch,scale,seed) -> (fpt_buggy, fpt_corrected). Gate-checked against the logged flops_per_token.

    Handles BOTH the protein release ('nanoprot-<arch>-<scale>-s<seed>') and the genome training dir
    ('genome-...'), because mamba's corrected FLOPs/token are context-dependent only weakly (dominated by
    6*n_params) while gpt2's are strongly context-dependent (attention grows with seq_len) -- so the
    genome axis MUST use genome configs, not protein ones. It also handles a dir that MIXES pre-fix metas
    (buggy fpt with the scan seq_len factor) and post-fix metas (already-corrected fpt emitted by the
    patched mamba.estimate_flops): a per-meta guard (fpt > 1.5*6*n_params <=> the huge O(L) scan term is
    still present) decides whether to apply the correction, so new corrected seeds are never re-corrected.
    """
    lut, gate_max_err = {}, 0.0
    for prefix in ("nanoprot-", "genome-"):
        for d in sorted(glob.glob(str(release_dir / (prefix + "*-*-s*")))):
            name = Path(d).name.split(prefix, 1)[1]            # e.g. mamba-M-s0
            parts = name.rsplit("-", 2)
            if len(parts) != 3:
                continue
            arch, scale, seedtok = parts
            if arch not in AR or scale not in ("XS", "S", "M", "L"):
                continue
            seed = seedtok.lstrip("s")
            metas = sorted(glob.glob(str(Path(d) / "meta_*.json")))
            if not metas:
                continue
            m = json.load(open(metas[-1]))
            fpt = float(m["flops_per_token"]); cfg = m["model_config"]; npar = int(m["n_params"])
            if arch == "mamba":
                L = int(cfg["max_seq_len"]); d_inner = cfg["expand"] * cfg["d_model"]
                scan_bug = 6 * d_inner * cfg["d_state"] * L * cfg["depth"]
                if fpt > 1.5 * 6 * npar:                        # pre-fix meta: O(L) scan term still present
                    recon = 6 * npar + scan_bug                 # gate (~fpt within non_matmul slack)
                    gate_max_err = max(gate_max_err, abs(recon - fpt) / fpt)
                    fpt_buggy, fpt_corr = fpt, fpt - scan_bug * (L - 1) / L
                else:                                           # post-fix meta: fpt already corrected
                    fpt_corr = fpt
                    fpt_buggy = fpt + scan_bug * (L - 1) / L     # reconstruct buggy value for the buggy axis
            else:
                fpt_buggy = fpt_corr = fpt                       # gpt2: no scan bug
            lut[(arch, scale, seed)] = (fpt_buggy, fpt_corr)
    return lut, gate_max_err


def build_curve(rows, x_of_step):
    c = [(math.log(x), v) for s, v, fl, rs in rows if (x := x_of_step(s, fl, rs)) and x > 0]
    c.sort()
    return c


def reversal(pairs, *, early_frac, n_boot, rng):
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
    rev = sum(st.mean(de[i] for i in idx) < 0 and st.mean(df[i] for i in idx) > 0
              for idx in ([rng.randrange(len(pairs)) for _ in pairs] for _ in range(n_boot)))
    return dict(n=len(pairs), d_early=mde, d_final=mdf, reversal=int(mde < 0 and mdf > 0), p_rev=rev / n_boot)


def analyze_axis(curves, scale, lut, axis, *, early_frac, n_boot, seed):
    gseeds = {s for (a, sc, s) in curves if a == AR[0] and sc == scale}
    mseeds = {s for (a, sc, s) in curves if a == AR[1] and sc == scale}
    pairs = []
    for s in sorted(gseeds & mseeds):
        if (AR[0], scale, s) not in lut or (AR[1], scale, s) not in lut:
            continue
        fptg = lut[(AR[0], scale, s)]; fptm = lut[(AR[1], scale, s)]
        if axis == "flop_buggy":
            xg = lambda st_, fl, rs: fptg[0] * st_; xm = lambda st_, fl, rs: fptm[0] * st_
        elif axis == "flop_fixed":
            xg = lambda st_, fl, rs: fptg[1] * st_; xm = lambda st_, fl, rs: fptm[1] * st_
        else:  # step
            xg = xm = lambda st_, fl, rs: float(st_)
        g = build_curve(curves[(AR[0], scale, s)], xg); m = build_curve(curves[(AR[1], scale, s)], xm)
        if len(g) >= 2 and len(m) >= 2 and min(g[-1][0], m[-1][0]) > max(g[0][0], m[0][0]):
            pairs.append((g, m))
    return reversal(pairs, early_frac=early_frac, n_boot=n_boot, rng=random.Random(seed))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path, default=Path(".cache/nanoprot/release"))
    ap.add_argument("--protein-dir", type=Path, default=Path(".cache/nanoprot/trajectory_results"))
    ap.add_argument("--genome-dir", type=Path, default=Path(".cache/nanoprot/trajectory_results_genome"))
    ap.add_argument("--scales", nargs="+", default=["S", "M", "L"])
    ap.add_argument("--early-frac", type=float, default=0.01)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lut, gate_err = corrected_fpt_lookup(args.release)
    print(f"  CORRECTNESS GATE: max |recon-logged|/logged over mamba cells = {gate_err:.4%} "
          f"({'PASS' if gate_err < 0.02 else 'FAIL'})")
    print("  corrected vs buggy FLOPs/token (mamba; gpt2 unchanged):")
    for sc in args.scales:
        for a in AR:
            vs = [lut[k] for k in lut if k[0] == a and k[1] == sc]
            if vs:
                b = st.mean(v[0] for v in vs); c = st.mean(v[1] for v in vs)
                print(f"    {a:5s} {sc}: buggy={b:.3e}  corrected={c:.3e}  (ratio corr/buggy={c/b:.3f})")
    HEAD = ("ss3", "ss8", "fold", "pfam_family", "frame")

    for label, d in (("PROTEIN", args.protein_dir), ("GENOME", args.genome_dir)):
        if not d.exists():
            continue
        data = load_raw(d)
        print(f"\n===== {label}: reversal under buggy-FLOP (paper) vs corrected-FLOP (fix) vs step =====")
        print(f"  {'concept/scale':17s}{'n':>2s} | {'BUGGY-FLOP dE/dF/P':>21s} | {'CORRECTED-FLOP dE/dF/P':>23s} |"
              f" {'STEP dE/dF/P':>19s}  survives?")
        for concept in sorted(data):
            if concept == "val_bpr":
                continue
            for sc in args.scales:
                r = {ax: analyze_axis(data[concept], sc, lut, ax, early_frac=args.early_frac,
                                      n_boot=args.n_boot, seed=args.seed)
                     for ax in ("flop_buggy", "flop_fixed", "step")}
                if r["flop_buggy"] is None:
                    continue

                def fmt(x):
                    return f"{x['d_early']:+.3f}/{x['d_final']:+.3f}/{x['p_rev']:.2f}" if x else "   --   "
                surv = "YES" if (r["flop_fixed"] and r["flop_fixed"]["p_rev"] >= 0.5) else "no"
                mark = " <<" if concept in HEAD else ""
                print(f"  {concept+'/'+sc:17s}{r['flop_buggy']['n']:>2d} | {fmt(r['flop_buggy']):>21s} | "
                      f"{fmt(r['flop_fixed']):>23s} | {fmt(r['step']):>19s}  {surv}{mark}")
    print("\n  dE=d_early(<0 mamba-early), dF=d_final(>0 gpt2-converged), P=P(reversal). survives? = corrected-FLOP P>=0.5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
