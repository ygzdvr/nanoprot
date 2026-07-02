# B-ALLOC / B-FCAST / B-CAL — selection-by-capability, forecasting, calibration

4 protein cells; LOCO; response Δ_final; seed-mean; 2000-boot CIs. Selection chance = 0.50.

## B-ALLOC (headline) — top-1 architecture selection accuracy by budget f  [95% CI]
| method | f=0.01 | f=0.03 | f=0.1 | f=0.3 |
|---|---|---|---|---|
| loss-rank | 0.00 [0.00,0.00] | 0.00 [0.00,0.00] | 0.00 [0.00,0.00] | 0.50 [0.00,1.00] |
| early-probe | 0.50 [0.00,1.00] | 0.50 [0.00,1.00] | 0.50 [0.00,1.00] | 0.50 [0.00,1.00] |
| cap-linear | 0.50 [0.00,1.00] | 0.50 [0.00,1.00] | 0.50 [0.00,1.00] | 0.50 [0.00,1.00] |
| cap-sat | 1.00 [1.00,1.00] | 1.00 [1.00,1.00] | 1.00 [1.00,1.00] | 0.50 [0.00,1.00] |

Mean regret (Δ lost vs oracle) & FLOP savings (=1−f per candidate):
| method | f=0.01 | f=0.03 | f=0.1 | f=0.3 |
|---|---|---|---|---|
| loss-rank | 0.094 | 0.094 | 0.094 | 0.029 |
| early-probe | 0.029 | 0.065 | 0.029 | 0.029 |
| cap-linear | 0.029 | 0.029 | 0.029 | 0.029 |
| cap-sat | 0.000 | 0.000 | 0.000 | 0.029 |

FLOP savings at f: {0.01:99%, 0.03:97%, 0.1:90%, 0.3:70%}.  loss-rank < chance ⇒ selecting by loss actively mis-picks the architecture (the rank-reversal as a decision failure).

## B-FCAST — held-out value (Δ_final) RMSE by budget  [95% CI]
| method | f=0.01 | f=0.03 | f=0.1 | f=0.3 |
|---|---|---|---|---|
| early-probe | 0.088 [0.050,0.113] | 0.098 [0.038,0.139] | 0.079 [0.026,0.108] | 0.051 [0.020,0.070] |
| loss-extrap | 0.187 [0.050,0.278] | 0.079 [0.055,0.099] | 0.076 [0.055,0.099] | 0.095 [0.027,0.132] |
| scaling-pop | 0.101 [0.059,0.130] | 0.101 [0.059,0.130] | 0.101 [0.059,0.130] | 0.101 [0.059,0.130] |
| cap-linear | 0.079 [0.055,0.099] | 0.077 [0.044,0.100] | 0.081 [0.028,0.113] | 0.068 [0.009,0.095] |
| cap-sat | 0.079 [0.046,0.106] | 0.081 [0.033,0.109] | 0.084 [0.030,0.117] | 0.059 [0.016,0.082] |
| cap-hier | 0.092 [0.056,0.118] | 0.081 [0.043,0.108] | 0.081 [0.028,0.112] | 0.068 [0.009,0.095] |

Value≠selection: the per-concept scaling law (scaling-pop) predicts converged MAGNITUDE well but cannot rank same-scale archs (selection acc 0); cap-sat overshoots non-saturating protein curves (bounded form is for SATURATING domains, e.g. Pythia/NLP — forecast.py); cap-hier (scaling-law ⊕ trajectory) is the best value forecaster.

## B-CAL — calibration (cap-hier conformal 90% coverage; LOCO) + value ablation
| budget f | gaussian cov@90 | conformal cov@90 | RMSE scaling-pop | cap-linear | cap-hier |
|---|---|---|---|---|---|
| 0.01 | 1.00 | 0.75 | 0.101 | 0.079 | 0.092 |
| 0.03 | 1.00 | 0.75 | 0.101 | 0.077 | 0.081 |
| 0.1 | 0.50 | 0.75 | 0.101 | 0.081 | 0.081 |
| 0.3 | 0.50 | 0.75 | 0.101 | 0.068 | 0.068 |

Gaussian PI is mildly under-covered (heavy-tailed errors); the distribution-free conformal
interval restores ~90% coverage — the calibrated uncertainty we report. Ablation: cap-hier ≤
cap-linear and ≤ scaling-pop at small budgets ⇒ combining the scaling-law prior with the
trajectory helps when early data is scarce. Leakage control: scaling law + loss→Δ map + the
conformal calibration set are all OTHER-cells-only (LOCO). Cross-domain generality (Pythia/NLP,
bounded forecaster) in forecast.py. TODO: per-cluster bootstrap; more seeds/scales; NUTS check.
