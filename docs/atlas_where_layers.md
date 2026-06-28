# Atlas — the "where": decoding depth and layer migration (Paper A)

Complement to the temporal "when" (the developmental ordering): *where* in the network does each
capability become decodable, and does that location move during training? We read
`best_layer_by_val` from the trajectory sidecars (per-checkpoint, per concept/arch/scale/seed) and
express it as **relative depth** = best_layer / (n_probed_layers − 1) ∈ [0,1] (0 = first layer,
1 = last). "Early" = the checkpoint at 25% of the saved trajectory; migration = final − early
(seed-mean). Sidecars complete for gpt2 (S/M/L), esm2-M, mamba-L; mamba-S/M partial (5 concepts).

## Relative decoding depth at convergence (seed-mean)
| cell | range over concepts | note |
|---|---|---|
| gpt2-L | 0.70–0.97 | deep; disorder/active shallowest (~0.70), signal deepest (0.97) |
| gpt2-M | 0.12–0.93 | deep except a transmembrane outlier (0.12) |
| esm2-M (encoder) | 0.47–0.82 | shallower overall (mid-to-deep) |
| mamba-L | 0.80–0.98 | deep |

## Signed layer migration over training (final − early, relative depth)
gpt2-L **+0.41**, esm2-M **+0.51**, mamba-L **+0.33**, gpt2-M ≈ 0.0.

## What is robust (safe to claim)
1. **Decoding is concentrated in deep layers** — for the autoregressive decoders the best probe
   layer sits at 0.7–0.98 of depth; the masked encoder reads somewhat shallower (0.47–0.82),
   plausibly because bidirectional context front-loads representation.
2. **The best-decoding layer tends to migrate deeper over training** (gpt2-L, esm2-M, mamba-L all
   strongly positive; gpt2-M flat) — capability consolidates into later layers as training
   proceeds, a spatial counterpart to the temporal emergence.

## What is NOT robust (do not claim)
- A concept-specific decoding depth or a depth *ordering* of concepts: it varies across cells
  (e.g. transmembrane 0.12 at gpt2-M vs 0.90 at gpt2-L). `best_layer_by_val` jumps between
  near-equivalent deep layers, so the fine spatial ordering is noisy.

## Consequence for the paper
The "where" is a **supporting** result, reported coarsely: deep decoding + a deepening trend over
training. Do **not** assert per-concept depths or a spatial concept-ordering. If the spatial signal
cannot be strengthened (e.g. with a layer-attribution method rather than argmax best-layer),
consider de-emphasizing "where" in the title and framing it as a secondary observation.
(Reproduce via the sidecar `best_layer_by_val` over `trajectory_results/traj_*.json`.)
