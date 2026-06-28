# P3.4 v2 — fair baseline + saturating forecaster (proteins + Pythia)

## Proteins — ranking accuracy (gpt2 vs mamba), 15 cells, mean [95% CI] by budget
| method | f=0.01 | f=0.03 | f=0.1 | f=0.3 |
|---|---|---|---|---|
| early-probe | 0.47 [0.20,0.73] | 0.60 [0.33,0.87] | 0.80 [0.60,1.00] | 0.93 [0.80,1.00] |
| loss-rank | 0.67 [0.40,0.87] | 0.67 [0.40,0.93] | 0.67 [0.40,0.87] | 0.67 [0.40,0.87] |
| cap-linear | 0.80 [0.60,1.00] | 0.93 [0.80,1.00] | 1.00 [1.00,1.00] | 1.00 [1.00,1.00] |
| cap-sat | 0.67 [0.40,0.87] | 0.67 [0.40,0.87] | 0.87 [0.67,1.00] | 0.93 [0.80,1.00] |

## Proteins — converged-value RMSE by budget
| method | f=0.01 | f=0.03 | f=0.1 | f=0.3 |
|---|---|---|---|---|
| early-probe | 0.093 | 0.073 | 0.056 | 0.030 |
| cap-linear | 0.084 | 0.061 | 0.049 | 0.033 |
| cap-sat | 0.082 | 0.063 | 0.050 | 0.031 |

## Pythia (NLP, 2 sizes) — converged-value RMSE by budget (generality)
| method | f=0.01 | f=0.03 | f=0.1 | f=0.3 |
|---|---|---|---|---|
| early-probe | 0.231 | 0.063 | 0.030 | 0.018 |
| cap-linear | 0.251 | 0.116 | 0.065 | 0.026 |
| cap-sat | 0.029 | 0.016 | 0.019 | 0.014 |

Fair early-probe = last checkpoint at/before C_e (no peeking). cap-sat = logistic-in-log-
compute (bounded). Proteins are gradual (linear ok); Pythia saturates (linear overshoots →
saturating form needed). v1 seed bootstrap on protein ranking; Pythia n_sizes small.
