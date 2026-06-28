# P3.2 — Rank-reversal: results (v1, S/M, iso-FLOP)

Script `scripts/rank_reversal.py` · data `trajectory_results` (S/M, 3 seeds) · seed=0, B=2000.
Compute axis **C(t)=train_flops·step/train_residues** (per-checkpoint, iso-FLOP — the raw
`train_flops` column is a per-run total, so this recovers cumulative compute). Early budget
C_e = 3 % of C_f. Score oriented higher=better (probe macro_f1/r2; loss = −val_bpr). Margin
Δ = gpt2 − mamba, **seed-paired** then bootstrapped over seeds.

## Headline
A statistically significant **early-proxy rank reversal exists on biological CAPABILITY —
not on loss**. mamba is the early iso-FLOP winner, gpt2 the convergence winner.

| task | scale | early | final | reversal | P(rev) | Δ_early [95% CI] | Δ_final [95% CI] |
|---|---|---|---|---|---|---|---|
| val_bpr | S | gpt2 | gpt2 | no | 0.00 | +0.038 [+0.029,+0.045] | +0.082 [+0.081,+0.086] |
| val_bpr | M | gpt2 | gpt2 | no | 0.00 | +0.076 [+0.063,+0.092] | +0.179 [+0.176,+0.186] |
| **ss8** | S | mamba | gpt2 | **yes** | **1.00** | −0.010 [−0.013,−0.005] | +0.011 [+0.011,+0.012] |
| **ss8** | M | mamba | gpt2 | **yes** | **1.00** | −0.008 [−0.015,−0.004] | +0.020 [+0.017,+0.024] |
| **ss3** | S | mamba | gpt2 | **yes** | **1.00** | −0.018 [−0.022,−0.011] | +0.026 [+0.024,+0.028] |
| ss3 | M | mamba | gpt2 | yes | 0.64 | −0.002 [−0.011,+0.007] | +0.035 [+0.030,+0.040] |
| active | S | (mamba) | gpt2 | yes\* | 1.00 | +0.000 [0,0] | +0.020 [+0.005,+0.032] |
| active | M | mamba | gpt2 | yes\* | 1.00 | −0.002 [−0.002,−0.001] | +0.024 [+0.004,+0.043] |
| rsa | S,M | gpt2 | gpt2 | no | 0.00 | gpt2 ahead throughout | |
| disorder | S,M | gpt2 | gpt2 | no | 0.00 | gpt2 ahead throughout | |

\* active-site early margin ≈ 0 → mark as a weak/marginal reversal.

## Key reading (sharper, and more defensible, than the pre-registered version)
- **Loss does NOT reverse iso-FLOP.** gpt2's val_bpr leads mamba's at *every* compute, early
  and late (Δ_early>0, CI excludes 0). The earlier "mamba leads early on loss" was a
  **fixed-step artifact**; at matched compute it disappears. → The loss-reversal claim is
  dropped; on the loss axis gpt2 is monotone-ahead. (Plan P3.2 §6/§7 should be updated: do NOT
  anchor a loss reversal at M.)
- **The reversal lives on CAPABILITY.** Secondary structure (ss8 at both scales, ss3 at S) and
  active-site show mamba as the early iso-FLOP winner, gpt2 as the convergence winner, with
  nested seed-bootstrap P(reversal)=1.0 (ss8, active). So an early *biological-capability* probe
  **actively mis-ranks** the architectures while the early *loss* does not. This is the crisp
  form of Paper B's phenomenon — and it is **robust to the mamba replicated-slice offset**,
  because probe metrics are insensitive to the ~0.02 bpr.
- ss3-M directional but not significant (P=0.64); rsa, disorder do not reverse (gpt2 throughout).
- Crossover C×/C_f: loss ≈0.001–0.006 (trivial/very early); capability ss8 ≈0.05–0.18, ss3
  ≈0.03–0.13 — i.e. capability reverses meaningfully later in training than the loss crossover.

## Why this is a stronger Paper-B story
The naive proxy that mis-ranks is the *early biological probe*, exactly the thing a practitioner
would use to pick a backbone for a downstream task — and the *loss*, the usual scaling proxy,
gives no warning (it already favors gpt2). Selecting on the early capability probe at C_e would
pick mamba for ss8/ss3/active and be wrong at convergence. This motivates the forecasting model
(P3.3/P3.4): you need the trajectory model, not the early snapshot.

## Caveats / deferred (not silently dropped)
- v1 uncertainty = **seed bootstrap (n=3)**; the per-cluster block bootstrap
  (`--dump-cluster-scores` re-probe) is the planned submission-grade upgrade.
- L scale gated on P1.4 (running); roster Kendall-τ across scales deferred to v2.
- active-site reversal rests on a ≈0 early margin → report as weak.
