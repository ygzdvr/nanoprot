# P3.2 — Rank-reversal: results (v1, S/M/L, iso-FLOP)

Script `scripts/rank_reversal.py` · data `trajectory_results` (3 seeds) · seed=0, B=2000.
Compute axis **C(t)=train_flops·step/train_residues** (per-checkpoint; the raw `train_flops`
column is a per-run total). Margin Δ = gpt2 − mamba, score higher=better (probe macro_f1/r2;
loss = −val_bpr), **seed-paired** then bootstrapped over seeds. Two quantities:
**reversal@1%** (mamba ahead at the 1%-compute calibration budget AND gpt2 ahead at convergence)
and the scale-comparable **crossover fraction C×/C_f** = compute fraction below which mamba leads.

## Headline — a *scaling law of the mis-ranking window*
The early-proxy rank reversal is (i) a **biological-capability** phenomenon, not a loss one, and
(ii) its compute window **shrinks ~20–25× with scale**.

| task | scale | reversal@1% | P(rev) | Δ@1% | **C×/C_f** | Δ_final |
|---|---|---|---|---|---|---|
| ss8 | S | **yes** | 1.00 | −0.019 | **0.184** | +0.011 |
| ss8 | M | **yes** | 1.00 | −0.023 | **0.058** | +0.020 |
| ss8 | L | no (window closed <1%) | 0.00 | +0.011 | **0.0065** | +0.051 |
| ss3 | S | **yes** | 1.00 | −0.016 | **0.131** | +0.026 |
| ss3 | M | **yes** | 1.00 | −0.030 | **0.034** | +0.035 |
| ss3 | L | no (window closed <1%) | 0.00 | +0.020 | **0.0079** | +0.060 |
| val_bpr | S,M | no | 0.00 | + | <0.006 | + (gpt2 throughout) |
| rsa | S,M,L | no | 0.00 | + | small | + |
| disorder | S,M,L | no | 0.00 | + | (L 0.45 but Δ_final≈0.001 → tied/noisy) | + |
| active | S | no | 0.00 | ≈0 | 0.058 | + |
| active | M,L | marginal | 0.96 / 0.73 | ≈0 / −0.013 | 0.35 / 0.45 | +0.024 / **+0.006 (≈tied)** |

## Reading (rigorous and defensible)
- **Secondary structure (ss3, ss8) reverses cleanly at S and M** — at the 1% calibration budget
  the early probe ranks mamba above gpt2, but gpt2 wins at convergence (P(reversal)=1.0).
- **The crossover compute fraction shrinks monotonically with scale**: ss8 0.184→0.058→0.0065,
  ss3 0.131→0.034→0.0079. In steps (≈C×·total_steps) the window closes at ≈170 steps (M) and
  ≈40 steps (L) — i.e. **larger models resolve the correct ranking earlier in training**, so by
  the 1% budget the L reversal has already closed. The reversal is real at L too but confined to
  <1% of compute. This explains the motivating observation (a ~120-step calibration ranking
  mamba>gpt2 sits inside the window at M but past it at L).
- **Loss does NOT reverse** (val_bpr crossover <0.6%, a step-1 transient; gpt2 monotone-ahead at
  every practical budget). The naive *loss* proxy gives no warning; the early *capability* probe
  actively mis-ranks. The mamba replicated-slice ~0.02 bpr offset inflates mamba's decodability
  throughout, so the convergence gpt2-win is a *conservative lower bound* — the reversal holds
  despite the slice favoring mamba (see `converged_decodability.md`).
- rsa, disorder do not reverse. active is **marginal/near-tied** (Δ_final≈0.006 at L) — report as
  weak, not a clean reversal.

## Why this is a strong Paper-B result
The thing that mis-ranks is the early *biological probe* a practitioner would actually use to pick
a backbone; the *loss* (the usual scaling proxy) gives no warning. And the danger is **scale-
dependent** — the mis-ranking window is widest at small/mid scale and narrows at large scale.
This quantitatively motivates the forecasting model (P3.3/P3.4): you cannot read the converged
capability ranking from an early snapshot, and how-early-is-safe itself scales.

## Caveats / deferred (not silently dropped)
- v1 uncertainty = **seed bootstrap (n=3)**; per-cluster block bootstrap (`--dump-cluster-scores`
  re-probe) is the submission-grade upgrade.
- L = the 5 concepts already fully probed (ss3/ss8/rsa/active/disorder); transmembrane/signal/
  val_bpr at L still probing (P1.4 running) — add on completion.
- active-site reversal rests on a ≈0 margin → weak; roster Kendall-τ across scales deferred to v2.
