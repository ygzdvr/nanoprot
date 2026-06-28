# Atlas — developmental-ordering robustness (Paper A reviewer-defense)

Reviewer question: *"Is the developmental ordering an artifact of how you define emergence?"*
We rank the seven concepts by **onset** (Δ first crosses a threshold ε) at ε ∈ {0.5×, 1×, 2×} the
default, and by the **threshold-free t50** (time to 50% of each concept's own final Δ), and report
Spearman rank agreement. Seed-mean per (arch, scale).

## Results
| cell | onset order (early→late) | ρ(onset@1× vs 0.5×) | ρ(vs 2×) | ρ(onset vs t50) |
|---|---|---|---|---|
| gpt2-M | transmembrane < disorder < ss3 < ss8 < signal < rsa < active | 0.96 | 0.96 | 0.57 |
| gpt2-L | signal < ss8 < transmembrane < ss3 < disorder < rsa < active | 0.96 | 0.75 | 0.54 |
| esm2-M | signal < ss3 < transmembrane < ss8 < disorder < rsa < active | 0.61 | 0.86 | 0.54 |
| mamba-M | ss3 < ss8 < disorder < rsa < active | 0.40 | 0.70 | 0.70 |

## Interpretation
- **Onset is robust to ε** (ρ ≈ 0.75–0.96 for the AR decoder and the encoder; mamba is noisier,
  ρ ≈ 0.4–0.7), so the order is not a knife-edge threshold artifact.
- **Onset and t50 agree only moderately (ρ ≈ 0.54–0.71)** — the *fine* ordering is
  emergence-metric-dependent. This is mechanistic, not noise: onset (an absolute ε-crossing)
  rewards concepts with large final Δ, whereas t50 (relative to each concept's own asymptote)
  measures maturation speed. The chief discrepancy is **solvent accessibility (rsa): early onset
  (crosses a small ε quickly) but late t50 (matures slowly toward a small final Δ)** — a
  slow-riser.

## What is robust (safe to claim) vs not
**Robust across ε and across the onset/t50 definitions:**
1. Local sequence composition (transmembrane, signal peptide) and secondary structure (ss3/ss8)
   become decodable **early**.
2. Catalytic/active-site identity emerges **last and is weakest** (smallest final Δ) — in every
   cell, by onset; and 2nd-latest by t50 with the smallest asymptote, so "last & weakest" holds
   under both definitions.
3. The early/late SEPARATION (structure-like vs function) is large and definition-invariant.

**NOT robust (do not claim):** a strict total order of all seven concepts — in particular rsa's
rank flips between onset and t50 (slow-riser). 

## Consequence for the paper
State the **coarse, defensible developmental order**: *local composition & secondary structure
first; catalytic function last and weakest; solvent accessibility a slow-riser whose exact rank is
emergence-metric-dependent.* Avoid asserting a precise total ordering. (`scripts`: reproduce via the
robustness snippet over `analyze_emergence.load_curves` + `emergence_times`.)
