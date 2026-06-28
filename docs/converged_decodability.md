# Converged decodability summary + the replicated-slice caveat (Paper A)

Final-checkpoint linear decodability (learned probe score, seed-mean) per concept × scale, from the
trajectory models. Two findings — one clean result, one claim-discipline caveat.

## Table (seed-mean learned_test at convergence)
| concept | scale | gpt2 | mamba | esm2 | gpt2−mamba |
|---|---|---|---|---|---|
| ss3 | M | 0.622 | 0.614 | **0.718** | +0.009 |
| ss3 | L | 0.683 | 0.669 | — | +0.013 |
| ss8 | M | 0.263 | 0.260 | **0.317** | +0.003 |
| ss8 | L | 0.318 | 0.298 | — | +0.020 |
| rsa | M | 0.281 | 0.266 | **0.348** | +0.015 |
| rsa | L | 0.358 | 0.313 | — | +0.045 |
| transmembrane | M | 0.781 | — | **0.896** | — |
| transmembrane | L | 0.832 | 0.828 | — | +0.004 |
| disorder | M | 0.665 | 0.677 | 0.668 | −0.012 |
| disorder | L | 0.639 | 0.646 | — | −0.008 |
| active | M | 0.554 | 0.545 | 0.586 | +0.009 |
| signal_peptide | M | 0.982 | — | 0.981 | — |

## Finding 1 (clean): the encoder decodes structure better than the AR models at matched scale
At scale M, the masked encoder (esm2) has the **highest** decodability on every structural concept
(transmembrane 0.896, ss3 0.718, rsa 0.348, ss8 0.317) — above both autoregressive models. This is
a valid cross-objective comparison (same probe metric) and a clean objective-axis result:
**bidirectional context yields more linearly decodable local structure than causal context.** (It
does not place esm2 on the AR loss axis; this is decodability, not loss.)

## Finding 2 (CAVEAT): the trajectory gpt2-vs-mamba comparison is replicated-slice-confounded
The trajectory runs are matched-compute **re-trains**: gpt2 reproduces the release to ≤0.005 bpr,
but **mamba re-trains ~0.02 bpr *better* than release**. A better-trained mamba has somewhat higher
decodability, so the trajectory-final gpt2−mamba margins here **understate** the true (release)
architecture gap — they are a *lower bound that favors mamba*. Consequences:
- Do **not** use this table as the architecture claim. The clean gpt2-vs-mamba comparisons are
  (i) the **static release probe table** (Pillar 3) and (ii) the **iso-FLOP scaling exponents**
  (gpt2 α=0.031 > mamba 0.023, non-overlapping CIs) — both on the released models.
- The disorder "mamba ahead" (−0.008/−0.012 at M/L) is within the slice confound → **not claimed**
  as a real mamba advantage.
- Even under this mamba-favoring confound, gpt2 is ahead on most concepts and the gap **grows with
  scale** where it is largest (rsa +0.045, ss8 +0.020 at L) — consistent with gpt2 out-scaling.

## Correction to the rank-reversal report
Earlier phrasing ("robust because probe metrics are insensitive to the ~0.02 bpr") is imprecise.
Correct statement: the slice **inflates mamba throughout**, so the rank-reversal's *convergence
gpt2-win is a conservative lower bound* — the reversal is real **despite** the slice favoring
mamba, which strengthens (not weakens) the claim. Updated in `rank_reversal_report.md`.
