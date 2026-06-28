#!/usr/bin/env python3
"""P3.1/P3.3(a) — fit the additive loss surface L(N,D)=E+A·N^-α+B·D^-β from the data-budget sweep.

Separates the parameter-bottleneck exponent α (capacity) from the data-bottleneck exponent β
(data-efficiency), per architecture — the AR pair (gpt2, mamba) on bits-per-residue; esm2 on its
own masked-CE surface. Robust (Huber) least-squares with log-parametrized positives; bootstrap CIs
over cells. Writes sweep_results.csv + prints α,β,E per arch.

NOTE: uses the sweep metas' training-time `last_val_bpr` (eval_tokens=524288, noisy). This is a
PRELIMINARY fit; the submission number uses a post-hoc clean eval (larger fixed eval_tokens) on the
final checkpoints. See the plan P3.1 §5.
"""
import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

NAME = re.compile(r"nanoprot-sweep-(gpt2|esm2|mamba)-(XS|S|M|L)-r(\d+)-s(\d+)")


def load_cells(sweep_dir: Path):
    rows = []
    for d in glob.glob(str(sweep_dir / "nanoprot-sweep-*")):
        m = NAME.search(d)
        metas = sorted(glob.glob(d + "/meta_*.json"))
        if not m or not metas:
            continue
        j = json.load(open(metas[-1]))
        if j.get("step") != j.get("num_iterations"):
            continue
        rows.append(dict(arch=m[1], scale=m[2], ratio=int(m[3]), seed=int(m[4]),
                         N=float(j["n_params"]), D=float(j["total_residues"]),
                         C=float(j["total_flops"]), bpr=float(j["last_val_bpr"])))
    return rows


def fit(N, D, L, n_boot=1000, seed=0):
    """L(N,D)=E+A N^-α+B D^-β via robust least-squares; bootstrap CIs over cells."""
    def model(p, N, D):
        lA, lB, lE, a, b = p
        return np.exp(lE) + np.exp(lA) * N ** (-a) + np.exp(lB) * D ** (-b)
    def resid(p, N, D, L):
        return model(p, N, D) - L
    lo = [-30, -30, -30, 0.0, 0.0]; hi = [40, 40, np.log(max(L)), 1.0, 1.0]
    p0 = [np.log(5), np.log(5), np.log(min(L) * 0.9), 0.08, 0.08]
    base = least_squares(resid, p0, args=(N, D, L), loss="huber", f_scale=0.03,
                         bounds=(lo, hi), max_nfev=20000).x
    pred = model(base, N, D)
    ss_res = float(np.sum((L - pred) ** 2)); ss_tot = float(np.sum((L - L.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rng = np.random.default_rng(seed); n = len(L); alphas = []; betas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            x = least_squares(resid, base, args=(N[idx], D[idx], L[idx]), loss="huber",
                              f_scale=0.03, bounds=(lo, hi), max_nfev=8000).x
            alphas.append(x[3]); betas.append(x[4])
        except Exception:
            pass
    q = lambda v, p: float(np.quantile(v, p)) if v else float("nan")
    return dict(alpha=base[3], beta=base[4], E=float(np.exp(base[2])), r2=r2,
                A=float(np.exp(base[0])), B=float(np.exp(base[1])),
                alpha_lo=q(alphas, .025), alpha_hi=q(alphas, .975),
                beta_lo=q(betas, .025), beta_hi=q(betas, .975), n_cells=n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path,
                    default=Path(os.environ.get("NANOPROT_BASE_DIR", ".cache/nanoprot")) / "sweep")
    ap.add_argument("--out", type=Path, default=Path("docs/sweep_results.csv"))
    args = ap.parse_args()
    rows = load_cells(args.sweep_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "scale", "ratio", "seed", "N", "D", "C", "bpr"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[fit_lnd] {len(rows)} cells -> {args.out}")
    print(f"{'arch':6}{'cells':>6}{'alpha [95% CI]':>22}{'beta [95% CI]':>22}{'E':>7}{'R2':>7}")
    res = {}
    for arch in ("gpt2", "mamba", "esm2"):
        c = [r for r in rows if r["arch"] == arch]
        if len(c) < 6:
            continue
        N = np.array([r["N"] for r in c]); D = np.array([r["D"] for r in c])
        L = np.array([r["bpr"] for r in c])
        res[arch] = fit(N, D, L)
        r = res[arch]
        print(f"{arch:6}{r['n_cells']:>6}  α={r['alpha']:.3f}[{r['alpha_lo']:.3f},{r['alpha_hi']:.3f}]"
              f"  β={r['beta']:.3f}[{r['beta_lo']:.3f},{r['beta_hi']:.3f}]  E={r['E']:.2f} R²={r['r2']:.3f}")
    if "gpt2" in res and "mamba" in res:
        g, m = res["gpt2"], res["mamba"]
        print(f"\nAR comparison (PRELIMINARY, training-time bpr):")
        print(f"  Δα (gpt2−mamba) = {g['alpha']-m['alpha']:+.3f}  (capacity exponent)")
        print(f"  Δβ (gpt2−mamba) = {g['beta']-m['beta']:+.3f}  (data-efficiency exponent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
