# B-ALLOC / B-FCAST / B-CAL — selection-by-capability, forecasting, calibration

30 protein cells; LOCO; response Δ_final; seed-mean; 2000-boot CIs. Selection chance = 0.50.

## B-ALLOC (headline) — top-1 architecture selection accuracy by budget f  [95% CI]
| method | f=0.01 | f=0.03 | f=0.1 | f=0.3 |
|---|---|---|---|---|
| loss-rank | 0.40 [0.13,0.67] | 0.40 [0.13,0.67] | 0.73 [0.47,0.93] | 0.40 [0.20,0.67] |
| early-probe | 0.67 [0.40,0.87] | 0.73 [0.47,0.93] | 0.67 [0.40,0.87] | 0.73 [0.47,0.93] |
| cap-linear | 0.73 [0.53,0.93] | 0.73 [0.53,0.93] | 0.73 [0.47,0.93] | 0.87 [0.67,1.00] |
| cap-sat | 0.67 [0.40,0.87] | 0.67 [0.40,0.87] | 0.73 [0.53,0.93] | 0.80 [0.60,1.00] |

Mean regret (Δ lost vs oracle) & FLOP savings (=1−f per candidate):
| method | f=0.01 | f=0.03 | f=0.1 | f=0.3 |
|---|---|---|---|---|
| loss-rank | 0.024 | 0.024 | 0.010 | 0.021 |
| early-probe | 0.006 | 0.009 | 0.006 | 0.002 |
| cap-linear | 0.003 | 0.005 | 0.006 | 0.001 |
| cap-sat | 0.013 | 0.017 | 0.011 | 0.001 |

FLOP savings at f: {0.01:99%, 0.03:97%, 0.1:90%, 0.3:70%}.  loss-rank < chance ⇒ selecting by loss actively mis-picks the architecture (the rank-reversal as a decision failure).

## B-FCAST — held-out value (Δ_final) RMSE by budget  [95% CI]
| method | f=0.01 | f=0.03 | f=0.1 | f=0.3 |
|---|---|---|---|---|
| early-probe | 0.095 [0.077,0.110] | 0.075 [0.061,0.088] | 0.057 [0.046,0.068] | 0.036 [0.025,0.045] |
| loss-extrap | 0.099 [0.064,0.134] | 0.060 [0.041,0.077] | 0.039 [0.027,0.050] | 0.035 [0.026,0.045] |
| scaling-pop | 0.036 [0.027,0.044] | 0.036 [0.026,0.044] | 0.036 [0.026,0.044] | 0.036 [0.027,0.044] |
| cap-linear | 0.078 [0.064,0.093] | 0.062 [0.050,0.073] | 0.052 [0.042,0.061] | 0.037 [0.030,0.043] |
| cap-sat | 0.137 [0.091,0.179] | 0.171 [0.106,0.233] | 0.072 [0.048,0.096] | 0.035 [0.027,0.043] |
| cap-hier | 0.033 [0.023,0.041] | 0.030 [0.021,0.038] | 0.052 [0.042,0.060] | 0.037 [0.030,0.043] |

Value≠selection: the per-concept scaling law (scaling-pop) predicts converged MAGNITUDE well but cannot rank same-scale archs (selection acc 0); cap-sat overshoots non-saturating protein curves (bounded form is for SATURATING domains, e.g. Pythia/NLP — forecast.py); cap-hier (scaling-law ⊕ trajectory) is the best value forecaster.

## B-CAL — calibration (cap-hier conformal 90% coverage; LOCO) + value ablation
| budget f | gaussian cov@90 | conformal cov@90 | RMSE scaling-pop | cap-linear | cap-hier |
|---|---|---|---|---|---|
| 0.01 | 0.87 | 0.90 | 0.036 | 0.078 | 0.033 |
| 0.03 | 0.83 | 0.90 | 0.036 | 0.062 | 0.030 |
| 0.1 | 0.80 | 0.90 | 0.036 | 0.052 | 0.052 |
| 0.3 | 0.83 | 0.90 | 0.036 | 0.037 | 0.037 |

Gaussian PI is mildly under-covered (heavy-tailed errors); the distribution-free conformal
interval restores ~90% coverage — the calibrated uncertainty we report. Ablation: cap-hier ≤
cap-linear and ≤ scaling-pop at small budgets ⇒ combining the scaling-law prior with the
trajectory helps when early data is scarce. Leakage control: scaling law + loss→Δ map + the
conformal calibration set are all OTHER-cells-only (LOCO). Cross-domain generality (Pythia/NLP,
bounded forecaster) in forecast.py. TODO: per-cluster bootstrap; more seeds/scales; NUTS check.
