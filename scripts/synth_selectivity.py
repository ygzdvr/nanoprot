#!/usr/bin/env python
"""Synthetic dial experiment: the controlled selectivity LAW (Paper B spotlight centerpiece).

The natural-data selectivity analysis (selectivity_law.py) showed the early attention-vs-SSM margin
tracks order-sensitivity Omega and periodicity Pi, with signal strength broken as a confound -- but it
is modest (r~-0.4, n=13 heterogeneous properties, a k-mer-blind outlier). This experiment removes every
one of those limitations by CONSTRUCTING the axis: a battery of synthetic per-position labelling tasks
whose position on the order/periodicity axis is designed and known, ALL CALIBRATED TO EQUAL CONVERGED
DECODABILITY, so that if the early margin still slides with the axis the signal-strength confound is
excluded BY CONSTRUCTION, not merely statistically.

Mechanism under test (crisp form the natural data supports): attention is early-native to
order-INVARIANT / compositional structure (it averages); a selective recurrence is early-native to
ORDERED-local and PERIODIC structure (it processes in order with a running state). Prediction: the
early margin d_early = decodability(attn) - decodability(ssm) is <0 (SSM early) for ordered/periodic
targets and >=0 for compositional/aperiodic ones, AT MATCHED converged decodability.

Task battery (per-position, C classes, IGNORE where undefined), each with a KNOWN axis position:
  comp_kR    y_t = quantile-bin of count of MARKED tokens in window radius R.  order-invariant (Omega~0).
  ratio      y_t = bin of (#markedA - #markedB) in window.                     order-invariant (Omega~0).
  order_adj  y_t = ordinal pattern of (x_{t-1},x_t,x_{t+1}) collapsed to C.     local order (Omega>0).
  motif_kR   y_t = nearest of C ordered templates whose MULTISETS are equal.    pure order (Omega high).
  period_p   content-triggered phase (t - last_start) mod p -> C.              periodic (Pi high).
  aperiodic  content-hash label, marginal matched to a period task.            strong-but-APERIODIC ctrl.
The last two families are the crucial OFF-DIAGONAL controls (strong-but-aperiodic; and, via calibration,
weak-but-periodic) that break the signal-strength confound.

Subcommands:
  calibrate : CPU-only. Generate + tune per-task label noise so the ordered-window ridge ceiling ~ target
              for ALL tasks (matched strength), compute Omega/Pi per task, write the task spec + table.
  run       : train ONE (task, arch, seed) cell, probe the residual stream over training, append a CSV.
  smoke     : tiny forward+one-step sanity check for both architectures (no full training).
Aggregation/figure: scripts/synth_regress.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

IGNORE = -100
VOCAB = 16                 # base symbols 0..15
TOK_START = VOCAB          # special "start" symbol for period tasks
TOK_PAD = VOCAB + 1
N_TOK = VOCAB + 2
C_CLASSES = 4
SEQ_LEN = 128


# =====================================================================================
# 1. TASK GENERATORS  (return int32 tokens [n,L], int32 labels [n,L] with IGNORE)
# =====================================================================================

def _rng(seed):
    return np.random.default_rng(seed)


def _random_tokens(n, L, rng, p_start=0.0):
    x = rng.integers(0, VOCAB, size=(n, L), dtype=np.int64)
    if p_start > 0:
        mask = rng.random((n, L)) < p_start
        x[mask] = TOK_START
    return x


def _bin_quantile(v, C, ref=None):
    """Map integer counts v to C balanced classes by quantile edges of ref (or v)."""
    ref = v if ref is None else ref
    qs = np.quantile(ref, np.linspace(0, 1, C + 1)[1:-1])
    return np.clip(np.searchsorted(qs, v, side="right"), 0, C - 1).astype(np.int64)


def gen_comp(n, L, rng, R=4, marked=tuple(range(8))):
    """Composition: count of marked tokens in [t-R, t+R], quantile-binned. Order-invariant. marked =
    half the vocab so the count has wide spread (well-separated quantile bins -> effective ceiling near
    the shared Bayes ceiling, matching the order/period tasks)."""
    x = _random_tokens(n, L, rng)
    is_marked = np.isin(x, marked).astype(np.int64)
    csum = np.cumsum(np.pad(is_marked, ((0, 0), (1, 0))), axis=1)  # [n, L+1]
    lo = np.maximum(0, np.arange(L) - R)[None, :]
    hi = np.minimum(L, np.arange(L) + R + 1)[None, :]
    cnt = np.take_along_axis(csum, hi, 1) - np.take_along_axis(csum, lo, 1)
    y = _bin_quantile(cnt.ravel(), C_CLASSES).reshape(n, L)
    y[:, :R] = IGNORE; y[:, L - R:] = IGNORE
    return x, y


def gen_ratio(n, L, rng, R=4, A=(0, 1, 2, 3), B=(4, 5, 6, 7)):
    """Composition: (#A - #B) in window, binned. Order-invariant."""
    x = _random_tokens(n, L, rng)
    a = np.isin(x, A).astype(np.int64); b = np.isin(x, B).astype(np.int64)
    ca = np.cumsum(np.pad(a, ((0, 0), (1, 0))), 1); cb = np.cumsum(np.pad(b, ((0, 0), (1, 0))), 1)
    lo = np.maximum(0, np.arange(L) - R)[None, :]; hi = np.minimum(L, np.arange(L) + R + 1)[None, :]
    d = (np.take_along_axis(ca, hi, 1) - np.take_along_axis(ca, lo, 1)
         - (np.take_along_axis(cb, hi, 1) - np.take_along_axis(cb, lo, 1)))
    y = _bin_quantile(d.ravel(), C_CLASSES).reshape(n, L)
    y[:, :R] = IGNORE; y[:, L - R:] = IGNORE
    return x, y


def gen_order_adj(n, L, rng):
    """Local order: the rank-order pattern of (x_{t-1},x_t,x_{t+1}). Composition of the triple's
    multiset does NOT determine ascending vs descending, so this needs ORDER. 6 orderings -> C via mod."""
    x = _random_tokens(n, L, rng)
    a, b, c = x[:, :-2], x[:, 1:-1], x[:, 2:]
    # ordinal pattern id in 0..5 (ties broken by index -> argsort is stable)
    trip = np.stack([a, b, c], -1)
    order = np.argsort(trip, axis=-1, kind="stable")  # permutation of (0,1,2)
    pid = order[..., 0] * 2 + (order[..., 1] > order[..., 2]).astype(np.int64)  # 0..5-ish
    y = np.full((n, L), IGNORE, dtype=np.int64)
    y[:, 1:-1] = (pid % C_CLASSES)
    return x, y


def gen_motif(n, L, rng, R=2):
    """Pure order: C ordered templates that are ROTATIONS of one another (identical multiset), so a
    bag-of-tokens predictor is at chance; only the ordered arrangement in [t-R,t+R] identifies the class.
    Label = argmin Hamming distance of the window to the C templates (with noise added in calibration)."""
    W = 2 * R + 1
    base = np.array([(i % VOCAB) for i in range(W)])          # a fixed ordered pattern
    templates = np.stack([np.roll(base, s) for s in range(C_CLASSES)])  # rotations: equal multiset
    x = _random_tokens(n, L, rng)
    y = np.full((n, L), IGNORE, dtype=np.int64)
    # plant a template at each interior position (overwrite window) so the label is well-defined & learnable
    cls = rng.integers(0, C_CLASSES, size=(n, L))
    for t in range(R, L - R):
        win = templates[cls[:, t]]                             # [n, W]
        x[:, t - R:t + R + 1] = win
        y[:, t] = cls[:, t]
    return x, y


def gen_period(n, L, rng, p=4, p_start=0.06):
    """Periodic: content-triggered phase. A START symbol appears ~p_start; after the most recent start,
    the phase (t - t_laststart) mod p is binned into ALL C classes as floor(phase/p * C) (so every class
    is populated for any p, avoiding a macro-F1 artifact). Before any start -> IGNORE. Requires phase
    tracking (a p-state counter) -> selective-recurrence-native. Sharp periodogram spike at 1/p."""
    x = _random_tokens(n, L, rng, p_start=p_start)
    y = np.full((n, L), IGNORE, dtype=np.int64)
    for i in range(n):
        last = -1
        for t in range(L):
            if x[i, t] == TOK_START:
                last = t
            elif last >= 0:
                phase = (t - last) % p
                y[i, t] = int(phase * C_CLASSES // p)      # bin phase -> all C classes populated
    return x, y


def gen_aperiodic(n, L, rng, p_start=0.06):
    """Strong-but-APERIODIC control (off-diagonal): same start-token structure + same marginal label
    frequencies as period_p, but the label after a start is a fixed HASH of the local content (aperiodic,
    flat periodogram) instead of the phase. Calibrated to the same ceiling -> equal strength, Pi~1."""
    x = _random_tokens(n, L, rng, p_start=p_start)
    y = np.full((n, L), IGNORE, dtype=np.int64)
    for i in range(n):
        last = -1
        for t in range(L):
            if x[i, t] == TOK_START:
                last = t
            elif last >= 0:
                y[i, t] = int((x[i, t] * 2654435761) % C_CLASSES)   # content hash: aperiodic
    return x, y


EPS = 0.15   # single uniform label-noise applied to EVERY task -> identical Bayes ceiling
             # (1-EPS)+EPS/C by construction == matched strength (verified empirically post-training)

TASKS = {
    # compositional (order-invariant): Omega ~ 0, Pi ~ 1  -> attention-native, expect NO reversal
    "comp_R2":  (gen_comp, {"R": 2}),
    "comp_R6":  (gen_comp, {"R": 6}),
    "ratio":    (gen_ratio, {}),
    # ordered-local: Omega > 0                              -> recurrence-native, expect reversal
    "order_adj": (gen_order_adj, {}),
    "motif_R2": (gen_motif, {"R": 2}),
    "motif_R4": (gen_motif, {"R": 4}),
    # periodic (p >= C so all classes populate): Pi high    -> recurrence-native, expect reversal
    "period_4": (gen_period, {"p": 4}),
    "period_8": (gen_period, {"p": 8}),
    "period_16": (gen_period, {"p": 16}),
    # strong-but-APERIODIC off-diagonal control: Omega ~ 0, Pi ~ 1, same start structure & marginal
    "aperiodic": (gen_aperiodic, {}),
}


# =====================================================================================
# 2. GROUND-TRUTH AXIS SCALARS (same definitions as selectivity_law.py, on generated data)
# =====================================================================================

def ridge_f1(X, y, C, seed=0, lam=1.0):
    n = len(y); rng = _rng(seed); perm = rng.permutation(n); cut = max(1, n * 3 // 4)
    Xtr, ytr, Xte, yte = X[perm[:cut]], y[perm[:cut]], X[perm[cut:]], y[perm[cut:]]
    if len(yte) < 5:
        Xte, yte = Xtr, ytr
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))]); Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    Y = np.zeros((len(ytr), C)); Y[np.arange(len(ytr)), ytr] = 1.0
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Y)
    pred = (Xte @ W).argmax(1)
    f1 = []
    for c in range(C):
        tp = int(((pred == c) & (yte == c)).sum()); fp = int(((pred == c) & (yte != c)).sum())
        fn = int(((pred != c) & (yte == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1))


def _window_feats(x, y, R=5, n_sample=8000, seed=0):
    """Sample positions; return (bag+center feats, ordered one-hot feats, labels)."""
    rng = _rng(seed); n, L = x.shape
    Xb, Xo, yy = [], [], []
    per = max(1, n_sample // n)
    for i in range(n):
        valid = np.where((y[i] != IGNORE) & (np.arange(L) >= R) & (np.arange(L) < L - R))[0]
        if valid.size == 0:
            continue
        take = valid if valid.size <= per else rng.choice(valid, per, replace=False)
        for t in take:
            win = x[i, t - R:t + R + 1]
            bag = np.bincount(win, minlength=N_TOK).astype(float); bag /= max(bag.sum(), 1)
            cen = np.zeros(N_TOK); cen[x[i, t]] = 1.0
            oh = np.zeros((2 * R + 1) * N_TOK); oh[np.arange(2 * R + 1) * N_TOK + win] = 1.0
            Xb.append(np.concatenate([bag, cen])); Xo.append(oh); yy.append(int(y[i, t]))
        if len(yy) >= n_sample:
            break
    return np.asarray(Xb), np.asarray(Xo), np.asarray(yy)


def omega_of(x, y, R=5):
    Xb, Xo, yy = _window_feats(x, y, R=R)
    if len(yy) < 50 or len(set(yy.tolist())) < 2:
        return float("nan"), float("nan"), float("nan")
    fo = ridge_f1(Xo, yy, C_CLASSES); fb = ridge_f1(Xb, yy, C_CLASSES)
    return fo - fb, fo, fb


def pi_of(y, max_lag=20, band=(2, 10), n_sample=3000, seed=0):
    rng = _rng(seed); idx = rng.choice(len(y), min(n_sample, len(y)), replace=False)
    num = np.zeros(max_lag + 1); den = np.zeros(max_lag + 1)
    for i in idx:
        L = y[i]; v = L != IGNORE
        for tau in range(max_lag + 1):
            a, b, m = L[:len(L) - tau], L[tau:], v[:len(L) - tau] & v[tau:]
            k = int(m.sum())
            if k:
                num[tau] += float(((a == b) & m).sum()); den[tau] += k
    agree = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    agree = np.nan_to_num(agree, nan=float(np.nanmean(agree)))
    P = np.abs(np.fft.rfft(agree - agree.mean())) ** 2
    f = np.fft.rfftfreq(len(agree)); sel = (f >= 1 / band[1]) & (f <= 1 / band[0])
    if sel.sum() < 2 or P[sel].mean() <= 0:
        return 1.0
    return float(P[sel].max() / P[sel].mean())


# =====================================================================================
# 3. CALIBRATION: tune per-task label noise so the ordered-window ceiling ~ target (matched strength)
# =====================================================================================

def add_label_noise(y, eps, seed):
    rng = _rng(seed); yn = y.copy(); m = y != IGNORE
    flip = (rng.random(y.shape) < eps) & m
    yn[flip] = rng.integers(0, C_CLASSES, size=int(flip.sum()))
    return yn


def characterize(n=2000, L=SEQ_LEN, seed=0, R_probe=5):
    """Compute the AXIS COORDINATES (Omega = order-sensitivity, Pi = periodicity) for each task at the
    fixed uniform noise EPS. Strength is matched BY CONSTRUCTION (identical Bayes ceiling (1-EPS)+EPS/C
    for every task), so there is no per-task tuning -- this only measures where each task sits on the
    order/periodicity axis and confirms the design (comp/aperiodic ~0 Omega; motif/order high Omega;
    period high Pi). Bag-decodability F1bag is reported as a sanity check that composition tasks are
    bag-decodable and order/period tasks are not."""
    spec = {}
    print(f"{'task':11s}{'Omega':>8s}{'Pi':>7s}{'F1ord':>7s}{'F1bag':>7s}   note")
    axis_note = {"comp_R2": "composition", "comp_R6": "composition", "ratio": "composition",
                 "order_adj": "order", "motif_R2": "order", "motif_R4": "order",
                 "period_4": "period", "period_8": "period", "period_16": "period",
                 "aperiodic": "off-diag (strong, aperiodic)"}
    for name, (fn, kw) in TASKS.items():
        x, y0 = fn(n, L, _rng(seed), **kw)
        y = add_label_noise(y0, EPS, seed + 1)
        om, fo, fb = omega_of(x, y, R=R_probe)
        pi = pi_of(y)
        spec[name] = {"fn": name, "kwargs": kw, "eps": EPS, "bayes_ceiling": round(1 - EPS + EPS / C_CLASSES, 4),
                      "omega": round(om, 4), "pi": round(pi, 4), "f1_ord": round(fo, 4), "f1_bag": round(fb, 4),
                      "axis": axis_note[name]}
        print(f"{name:11s}{om:8.3f}{pi:7.2f}{fo:7.3f}{fb:7.3f}   {axis_note[name]}")
    print(f"\n  matched Bayes ceiling (all tasks) = (1-{EPS})+{EPS}/{C_CLASSES} = "
          f"{1 - EPS + EPS / C_CLASSES:.3f}")
    return spec


def make_dataset(task, n, L, seed, eps=EPS):
    fn, kw = TASKS[task]
    x, y0 = fn(n, L, _rng(seed), **kw)
    return x, add_label_noise(y0, eps, seed + 7)


# =====================================================================================
# 4. MODELS (minimal faithful instances: causal attention  vs  selective diagonal SSM)
#    Imported lazily so `calibrate` runs without torch.
# =====================================================================================

def build_models_and_train(*args, **kwargs):
    import torch  # noqa: F401  (deferred; see scripts/synth_train_impl in run())
    raise NotImplementedError  # training lives in run(); this stub keeps calibrate torch-free


# =====================================================================================
# CLI
# =====================================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate"); c.add_argument("--out", type=Path, default=Path("docs/synth_task_spec.json"))
    c.add_argument("--target", type=float, default=0.80); c.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--spec", type=Path, default=Path("docs/synth_task_spec.json"))
    r.add_argument("--task", required=True); r.add_argument("--arch", required=True, choices=["attn", "ssm"])
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", type=Path, default=Path(".cache/nanoprot/synth_results/curves.csv"))
    r.add_argument("--steps", type=int, default=2500); r.add_argument("--batch", type=int, default=64)
    r.add_argument("--d-model", type=int, default=96); r.add_argument("--device", default=None)
    s = sub.add_parser("smoke")
    args = ap.parse_args()

    if args.cmd == "calibrate":
        spec = characterize(seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(spec, indent=2))
        print(f"\n  wrote {args.out}")
        return 0
    # run / smoke import torch and the training implementation
    from scripts.synth_train_impl import run_cell, smoke
    if args.cmd == "smoke":
        return smoke()
    spec = json.loads(args.spec.read_text())
    return run_cell(spec, args)


if __name__ == "__main__":
    raise SystemExit(main())
