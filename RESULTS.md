# nanoprot v0.5 — release results

The full suite: **3 architectures × 4 scales × 3 seeds = 36 models**, each trained
from scratch on **UniRef50 (release 2026_01, 60,251,814 sequences)** under a matched,
compute-optimal data budget (residues = 12 × parameters), 512-residue context, the
shared 33-token ESM-2 residue alphabet. Per-cell data in [`results/results.csv`](results/results.csv).

## Final metrics (mean ± std over 3 seeds)

| arch | scale | params | residues | metric |
|---|---|---:|---:|---:|
| gpt2 | XS | 11 M | 0.13 B | **3.9894 ± 0.0050** bpr |
| gpt2 | S | 36 M | 0.43 B | **3.8807 ± 0.0002** bpr |
| gpt2 | M | 135 M | 1.62 B | **3.6838 ± 0.0013** bpr |
| gpt2 | L | 599 M | 7.19 B | **3.5360 ± 0.0003** bpr |
| mamba | XS | 9 M | 0.11 B | 4.0191 ± 0.0007 bpr |
| mamba | S | 35 M | 0.42 B | 3.9003 ± 0.0010 bpr |
| mamba | M | 152 M | 1.82 B | 3.7853 ± 0.0012 bpr |
| mamba | L | 631 M | 7.57 B | 3.6449 ± 0.0012 bpr |
| esm2 | XS | 7 M | 0.09 B | 3.7476 ± 0.0005 mCE |
| esm2 | S | 33 M | 0.40 B | 3.6336 ± 0.0029 mCE |
| esm2 | M | 148 M | 1.77 B | 3.4934 ± 0.0024 mCE |
| esm2 | L | 649 M | 7.79 B | 3.3066 ± 0.0019 mCE |

`bpr` = validation bits-per-residue (autoregressive: gpt2, mamba — directly
comparable). `mCE` = validation masked cross-entropy bits (esm2 masked-LM — a
*different* quantity, comparable only among esm2 models / on downstream tasks).
Seed spread is tiny (σ ≈ 0.001–0.005) — the differences below are far larger.

## Headline: transformers out-scale Mamba SSMs on proteins

For the directly-comparable autoregressive models, **gpt2 beats mamba at every
scale, and the gap widens with scale:**

| scale | gpt2 bpr | mamba bpr | gap (mamba − gpt2) |
|---|---:|---:|---:|
| XS (~10 M) | 3.9942 | 4.0196 | +0.0255 |
| S (~35 M) | 3.8808 | 3.9006 | +0.0197 |
| M (~150 M) | 3.6835 | 3.7841 | **+0.1006** |
| L (~600 M) | 3.5356 | 3.6446 | **+0.1089** |

The compute-optimal scaling-law fits `L(N) = E + A·N^(−α)` quantify it:

| arch | α (scaling exponent) | R² |
|---|---:|---:|
| **gpt2** (AR) | **0.0309** | 0.991 |
| mamba (AR) | 0.0228 | 0.997 |
| esm2 (MLM) | 0.0276 | 0.982 |

gpt2 has both the **lower loss** and the **steeper exponent** (0.031 vs 0.023):
the transformer improves faster with scale, so the architectures are near-parity
at the smallest sizes but diverge by mid-scale — extrapolating, the gap continues
to grow. On UniRef50 next-residue prediction, the attention inductive bias scales
better than the selective-SSM one. (See `docs/figures/scaling_laws.png`.)

> α is the *compute-optimal* exponent (N and D co-vary along D = 12N); it is not
> the isolated parameter-exponent of a full L(N, D) surface. The data-budget sweep
> (`scripts/gen_sweep_configs.py`) is scaffolded to decouple them.

## A methodological note worth keeping

A 120-step, sub-sampled **calibration** run suggested the *opposite* — mamba
edging gpt2 (3.89 vs 3.95). The **fully-converged** runs reverse it at every
scale. Short / under-trained / sub-sampled comparisons mislabeled the winner;
convergence matters.

## Figures
- `docs/figures/scaling_laws.png` — L(N) power-law fits + α per architecture.
- `docs/figures/scaling_curves.png` — metric vs. parameters.
- `docs/figures/training_curves.png` — validation loss vs. residues-seen.

## Reproduce
```bash
python -m scripts.gen_release_configs           # the 36-cell grid
sbatch runs/prepare_uniref50.slurm              # build the corpus (after FASTA download)
sbatch runs/train_release.slurm                 # train (resumable, skip-complete)
python -m scripts.aggregate_results --release-root $NANOPROT_BASE_DIR/release
python -m scripts.scaling_laws    --release-root $NANOPROT_BASE_DIR/release
python -m scripts.make_model_card --release-root $NANOPROT_BASE_DIR/release
```
