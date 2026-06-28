# L(N,D) α/β separation — status (P3.1/P3.3a)

**Lattice banked:** 45 cells in `docs/sweep_results.csv` (arch, scale, ratio, N=n_params,
D=total_residues, C=total_flops, bpr=last_val_bpr). gpt2/mamba/esm2 × {XS:5, S:6, M:4 ratios} × seed0.

## Preliminary fit FAILED to identify α and β (do not report)
Robust (Huber) fit of `L(N,D)=E+A·N^−α+B·D^−β` on the **training-time** bpr:
- gpt2 α=0.007 **[0.000, 1.000]**, β=0.092 [0.027, 1.000], E→0, R²=0.80
- mamba α=0.018 [0,1], β=0.024 [0.016, 1.000], R²=0.56
- esm2 α=1.000 [0.011, 1.000], β=0.027, R²=0.82

The bootstrap CIs span the entire [0,1] bound range and E collapses to ~0 (degenerate: with α≈0 the
N-term becomes a constant absorbed into A). **α and β are not separated.** The preliminary
Δα=−0.011 / Δβ=+0.068 are artifacts, not results.

## Why (diagnosis)
1. **Low SNR:** the reducible-loss range across the lattice is only ~0.8 bpr, while the
   training-time eval (`eval_tokens=524288`) carries ~0.05–0.13 bpr noise/outliers (e.g. mamba
   S-r6, M-r24). Signal/noise ≈ 8 — too low to resolve a 5-parameter surface.
2. **Few cells:** 15 cells/arch for 5 parameters.
3. **Approximate shared-D:** D=ratio·N, so absolute D only *approximately* overlaps across N
   (XS-high-ratio ≈ S-mid-ratio ≈ M-low-ratio), weakly constraining α.
4. **E unconstrained:** the irreducible floor is not pinned, so it trades off with A,B,α,β.

## Path forward (in priority)
1. **Sweep-aware post-hoc clean eval** (larger fixed eval_tokens → ~8× less noise). The existing
   `posthoc_val_loss_eval.py` cannot be reused as-is: its `NAME_RE` does not match
   `nanoprot-sweep-{arch}-{scale}-r{ratio}-s{seed}` and it drops the ratio. **Build a sweep-aware
   eval** that parses the sweep name and records N/D/C + clean val_bpr.
2. **Better-conditioned fit:** constrain E to a data-driven floor (E < min L), fit in log-space with
   a log-sum-exp Huber objective (Hoffmann-style), better init.
3. If still unidentified, **add exact shared-D cells and/or seeds {1,2}** on the headline cells.
4. **FALLBACK (load-bearing claim is safe either way):** Paper B's scaling headline is the
   **combined compute-optimal exponent** from the released diagonal (gpt2 α=0.031 > mamba 0.023,
   non-overlapping CIs) — already established and *not* dependent on the α/β separation. The
   off-diagonal α/β decomposition is a **stretch** result; report it only if it identifies cleanly.

## Consequence for Paper B
The rank-reversal + forecasting + the combined exponent carry the paper. The α/β separation is an
enhancement, not a dependency — so a clean-but-negative identifiability outcome is acceptable and
should be stated honestly rather than forced.
