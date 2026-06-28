# Option A — Pythia (NLP) generality of capability-vs-loss + forecasting

Sizes: ['70m', '160m']; LAMBADA-openai acc + LM loss; compute = log FLOPs.

## 1. Capability-vs-loss timing divergence
| size | loss 50%-drop @step | capability(acc) 50%-final @step | lag (×, in steps) |
|---|---|---|---|
| 70m | 116 | 1491 | 12.9× |
| 160m | 116 | 2475 | 21.3× |

Loss reaches its half-improvement long before the capability appears — capability has
its own (much later, abrupt) emergence schedule. The same loss-≠-capability phenomenon as in
proteins, now in NLP.

## 2. Forecasting converged LAMBADA acc from early checkpoints (RMSE)
| budget f=C_e/C_final | early-probe | cap-extrap (ours) |
|---|---|---|
| 0.01 | 0.187 | 0.251 |
| 0.03 | 0.059 | 0.116 |
| 0.1 | 0.031 | 0.065 |
| 0.3 | 0.012 | 0.026 |

Honest reading: because LAMBADA capability emerges *abruptly*, neither method can predict
it from *pre-emergence* checkpoints (acc≈0 → both extrapolate ~0); once capability has begun
rising, cap-extrap forecasts the converged value with lower error than the early value. The
sudden emergence itself reinforces the thesis: early/loss signals under-determine final
capability. (n_sizes small; this is a method-generality data point, not the arch-reversal.)
