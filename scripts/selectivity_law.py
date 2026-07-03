#!/usr/bin/env python
"""Selectivity-law analysis (Paper B spotlight centerpiece).

Tests the mechanistic prediction that the attention-vs-SSM EARLY margin is governed by how much a
property is determined by SSM-favorable SEQUENTIAL structure, INDEPENDENT of signal strength. Two
DATA-ONLY scalars per property (computed from labels + raw tokens, never from the trained models, so
the regression is not circular):

  Pi_k    (periodicity):   peakiness of the label periodogram in the period-[2,10] band.
                           reading-frame -> high (sharp 1/3 spike); run/compositional labels -> ~1.
  Omega_k (order-sens.):   macro-F1 of a linear label predictor from ORDERED local tokens MINUS the
                           same from order-INVARIANT composition. secondary-structure/fold/family ->
                           high; GC (composition IS the label) -> ~0.

  sigma_k = z(Pi_k) + z(Omega_k)   ("SSM-favorable sequential-structure content").

Mechanistic prediction: the signed early margin d_early = score(attn) - score(ssm) DECREASES with
sigma_k (more sequential structure -> SSM leads earlier -> more negative -> reversal). GC is the
off-diagonal control: strong signal, sigma ~ 0, does NOT reverse (positive d_early).

Stage 1 (this file): compute Pi_k, Omega_k per cache. Stage 2 (--join): merge with reversal stats
(rank_reversal CSVs) + signal strength, regress d_early ~ sigma_k | s_k, and plot.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

IGNORE = -100
_RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# cache IO
# ---------------------------------------------------------------------------

def load_cache(d: Path, max_lines: int = 12000):
    """Read a bounded prefix (these are stationary data statistics; the full 1.88M-window caches need
    not be materialized). Windows are written in genomic/split order, so a prefix is representative."""
    meta = json.loads((d / "meta.json").read_text())
    seqs, labs = [], []
    with (d / "data.jsonl").open() as fh:
        for k, line in enumerate(fh):
            if k >= max_lines:
                break
            r = json.loads(line)
            seqs.append(r["sequence"]); labs.append(r["labels"])
    return {
        "concept": meta["concept"], "source": meta.get("source", ""),
        "level": meta.get("level", "residue"), "n_classes": meta["n_classes"],
        "task": meta.get("task", "classification"), "seqs": seqs, "labs": labs,
    }


def normalize_labels(cache):
    """Map None -> IGNORE and, for regression properties (e.g. solvent accessibility), tercile-bin the
    continuous target into 3 classes (buried/intermediate/exposed) so the nominal periodicity- and
    order-sensitivity estimators apply uniformly. Idempotent for integer classification labels."""
    labs = cache["labs"]
    seq = lambda L: (L if isinstance(L, list) else [L])
    if cache["task"] == "regression":
        vals = np.array([x for L in labs for x in seq(L) if x is not None and x != IGNORE], float)
        q1, q2 = np.percentile(vals, [33.3, 66.7]) if vals.size else (0.0, 0.0)
        binv = lambda x: IGNORE if (x is None or x == IGNORE) else (0 if x < q1 else 1 if x < q2 else 2)
        cache["labs"] = [[binv(x) for x in seq(L)] for L in labs]
        cache["n_classes"] = 3
    else:
        cache["labs"] = [[IGNORE if x is None else int(x) for x in seq(L)] for L in labs]
    return cache


def build_vocab(seqs, cap=2000):
    chars = set()
    for s in seqs[:cap]:
        chars.update(s)
    return {c: i for i, c in enumerate(sorted(chars))}


# ---------------------------------------------------------------------------
# Pi_k : periodicity of the (per-position) label sequence
# ---------------------------------------------------------------------------

def periodicity(labs, *, max_lag=20, n_sample=1500, band=(2, 10)):
    """Peakiness of the label-agreement periodogram in the period-[band] frequency window.

    agreement(tau) = P(L_t == L_{t+tau}) over VALID (non-ignore) pairs, averaged over sampled windows;
    a periodic label (period p) has a sharp spectral spike at f=1/p, a run/compositional label does
    not. Pi = max/mean of the periodogram over f in [1/band_hi, 1/band_lo] (>=1; 1 = no periodicity)."""
    idx = _RNG.choice(len(labs), min(n_sample, len(labs)), replace=False)
    num = np.zeros(max_lag + 1); den = np.zeros(max_lag + 1)
    for i in idx:
        L = np.asarray(labs[i])
        if L.ndim == 0 or L.size < 4:
            continue
        v = L != IGNORE
        for tau in range(max_lag + 1):
            a, b = L[:L.size - tau], L[tau:]
            va, vb = v[:L.size - tau], v[tau:]
            m = va & vb
            k = int(m.sum())
            if k:
                num[tau] += float(np.sum((a == b) & m)); den[tau] += k
    agree = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    if np.isnan(agree).any():
        agree = np.nan_to_num(agree, nan=float(np.nanmean(agree)))
    x = agree - agree.mean()
    P = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x))
    lo, hi = 1.0 / band[1], 1.0 / band[0]
    sel = (freqs >= lo) & (freqs <= hi)
    if sel.sum() < 2 or P[sel].mean() <= 0:
        return 1.0, None
    peak_f = float(freqs[sel][np.argmax(P[sel])])
    return float(P[sel].max() / P[sel].mean()), (1.0 / peak_f if peak_f > 0 else None)


# ---------------------------------------------------------------------------
# Omega_k : order-sensitivity (ordered vs order-invariant label predictability)
# ---------------------------------------------------------------------------

def _fit_f1(X, y, n_classes, *, seed=0, lam=1.0):
    """Macro-F1 of a closed-form ridge one-vs-rest linear classifier (a clean, fast linear probe;
    W = (X'X + lam I)^-1 X' Y_onehot, predict argmax). No sklearn dependency; identical estimator is
    applied to the ordered and the composition features so their F1 gap is a fair order-sensitivity."""
    n = len(y); rng = np.random.default_rng(seed)
    perm = rng.permutation(n); cut = max(1, n * 3 // 4)
    Xtr, ytr = X[perm[:cut]], y[perm[:cut]]
    Xte, yte = X[perm[cut:]], y[perm[cut:]]
    if len(yte) < 5:
        Xte, yte = Xtr, ytr
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))]); Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    Y = np.zeros((len(ytr), n_classes)); Y[np.arange(len(ytr)), ytr] = 1.0
    d = Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ Y)
    pred = (Xte @ W).argmax(1)
    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (yte == c)).sum()); fp = int(((pred == c) & (yte != c)).sum())
        fn = int(((pred != c) & (yte == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def order_sensitivity(cache, vocab, *, w=7, n_sample=6000):
    """Omega = F1(ordered features) - F1(composition features). Per-position: composition = token
    histogram of [t-w,t+w]; ordered = per-relative-position one-hot (captures order + local position).
    Per-sequence: composition = aa histogram; ordered = 2-mer histogram."""
    V = len(vocab); nc = cache["n_classes"]; seqs, labs = cache["seqs"], cache["labs"]
    Xb, Xo, y = [], [], []
    if cache["level"] == "sequence":
        idx = _RNG.choice(len(seqs), min(n_sample, len(seqs)), replace=False)
        for i in idx:
            s = seqs[i]; lab = labs[i][0] if isinstance(labs[i], list) else labs[i]
            if lab == IGNORE:
                continue
            ids = np.array([vocab.get(c, 0) for c in s])
            comp = np.bincount(ids, minlength=V).astype(float); comp /= max(comp.sum(), 1)
            bg = np.zeros(V * V)
            if len(ids) > 1:
                bi = ids[:-1] * V + ids[1:]
                bg = np.bincount(bi, minlength=V * V).astype(float); bg /= max(bg.sum(), 1)
            Xb.append(comp); Xo.append(bg); y.append(int(lab))
    else:
        per = max(1, n_sample // max(1, len(seqs) // 50 + 1))
        order_dim = V * (2 * w + 1)
        for i in _RNG.choice(len(seqs), min(len(seqs), 4000), replace=False):
            s, L = seqs[i], np.asarray(labs[i])
            ids = np.array([vocab.get(c, 0) for c in s]); n = len(ids)
            valid = np.where((L != IGNORE) & (np.arange(n) >= w) & (np.arange(n) < n - w))[0]
            if valid.size == 0:
                continue
            take = valid if valid.size <= per else _RNG.choice(valid, per, replace=False)
            for t in take:
                win = ids[t - w: t + w + 1]
                comp = np.bincount(win, minlength=V).astype(float); comp /= max(comp.sum(), 1)
                cen = np.zeros(V); cen[ids[t]] = 1.0     # center identity in BOTH feature sets, so the
                oh = np.zeros(order_dim)                  # ordered-minus-bag gap isolates CONTEXT ORDER,
                oh[np.arange(2 * w + 1) * V + win] = 1.0  # not mere dependence on the central token
                Xb.append(np.concatenate([comp, cen])); Xo.append(oh); y.append(int(L[t]))
            if len(y) >= n_sample:
                break
    if len(y) < 50 or len(set(y)) < 2:
        return float("nan"), float("nan"), float("nan"), len(y)
    Xb, Xo, y = np.asarray(Xb), np.asarray(Xo), np.asarray(y)
    fb = _fit_f1(Xb, y, nc); fo = _fit_f1(Xo, y, nc)
    return float(fo - fb), float(fo), float(fb), len(y)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caches", nargs="+", required=True, help="probe cache dirs")
    ap.add_argument("--out", type=Path, default=Path("docs/selectivity_scalars.csv"))
    args = ap.parse_args()
    rows = []
    print(f"{'property':22s}{'level':9s}{'Pi(peak)':>12s}{'period':>8s}{'Omega':>9s}{'F1ord':>8s}{'F1bag':>8s}")
    for d in args.caches:
        c = normalize_labels(load_cache(Path(d))); vocab = build_vocab(c["seqs"])
        if c["level"] == "sequence":
            pi, per = 1.0, None
        else:
            pi, per = periodicity(c["labs"])
        om, fo, fb, n = order_sensitivity(c, vocab)
        key = f"{c['concept']}/{c['source']}"
        print(f"{key:22s}{c['level']:9s}{pi:12.3f}{(per if per else 0):8.2f}{om:9.4f}{fo:8.3f}{fb:8.3f}")
        rows.append({"property": c["concept"], "source": c["source"], "level": c["level"],
                     "Pi": round(pi, 5), "period": round(per, 3) if per else "",
                     "Omega": round(om, 5), "F1_ord": round(fo, 5), "F1_bag": round(fb, 5), "n": n})
    import csv
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
