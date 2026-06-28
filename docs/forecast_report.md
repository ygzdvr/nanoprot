# P3.4 — Forecasting converged capability from early checkpoints

15 (concept × scale) cells; concepts=['ss3', 'ss8', 'rsa', 'active', 'disorder']; scales=['S', 'M', 'L']; seed-mean curves;
ranking-accuracy CIs = 2000× bootstrap over cells.

## Ranking accuracy (gpt2 vs mamba vs converged truth), mean [95% CI], by budget f=C_e/C_final
| method | f=0.01 | f=0.03 | f=0.1 | f=0.3 |
|---|---|---|---|---|
| early-probe | 0.60 [0.33,0.87] | 0.60 [0.33,0.87] | 0.87 [0.67,1.00] | 0.93 [0.80,1.00] |
| loss-rank | 0.67 [0.40,0.87] | 0.67 [0.40,0.93] | 0.67 [0.40,0.87] | 0.67 [0.40,0.87] |
| cap-extrap | 0.87 [0.67,1.00] | 0.93 [0.80,1.00] | 1.00 [1.00,1.00] | 1.00 [1.00,1.00] |

## S_final forecast RMSE (predict converged capability value) by budget
| method | f=0.01 | f=0.03 | f=0.1 | f=0.3 |
|---|---|---|---|---|
| early-probe | 0.089 | 0.071 | 0.049 | 0.028 |
| cap-extrap | 0.086 | 0.061 | 0.047 | 0.034 |

Reading: cap-extrap (linear-in-log-compute capability extrapolation) recovers the
converged gpt2-vs-mamba ranking far better than the naive early-probe (which reverses) and
than loss-rank (loss does not track the capability ranking, capping at ~0.67). It reaches
perfect ranking from ~10% of compute. v1: seed-mean, n_cells=15; per-cluster/seed CIs TODO.
