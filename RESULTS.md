# nanoprot v0.5 — release results

The full suite: **3 architectures × 4 scales × 3 seeds = 36 models**, each trained
from scratch on **UniRef50 (release 2026_01, 60,251,814 sequences)** under a matched,
compute-optimal data budget (residues = 12 × parameters), 512-residue context, the
shared 33-token ESM-2 residue alphabet. Per-cell data in [`results/results.csv`](results/results.csv).

## Models

All 12 `(arch, scale)` models (each carrying its 3 seeds) are public on the Hub:
**[🤗 nanoprot v0.5 collection](https://huggingface.co/collections/yagizdevre/nanoprot-v05-protein-lm-scaling-suite-6a2ad647b6cc80fa1b846cf4)**
— `yagizdevre/nanoprot-{gpt2,mamba,esm2}-{XS,S,M,L}`. Seed 0 is the repo default;
seeds 1–2 live in `seed1/`, `seed2/` subfolders. Optimizer state is not shipped.

```python
from huggingface_hub import snapshot_download
from nanoprot.training.checkpoint import load_pretrained
local = snapshot_download("yagizdevre/nanoprot-gpt2-L")
model, cfg, meta, tok = load_pretrained(local, device="cpu", return_tokenizer=True)
```

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

## Statistical robustness

The gpt2 > mamba result is not a point-estimate artifact. Using the per-seed
checkpoints (3 seeds × 4 scales), a **stratified seed bootstrap** (4000 resamples,
seeds resampled within each scale) puts 95% CIs on the log-log exponent
`α = −d(log L)/d(log N)` — and the AR architectures separate cleanly:

| arch | α | 95% CI |
|---|---:|---:|
| **gpt2** (bpr) | 0.0309 | **[0.0307, 0.0312]** |
| mamba (bpr) | 0.0228 | **[0.0228, 0.0229]** |
| esm2 (mCE, separate) | 0.0278 | [0.0277, 0.0280] |

The gpt2 and mamba intervals are **non-overlapping**; the exponent gap is
Δα = +0.0081 [95% CI +0.0078, +0.0084], **P(gpt2 steeper) = 1.000**. The per-scale
loss gap (mamba − gpt2) is significantly positive at *every* scale (all CIs exclude
0, P = 1.000) — small at XS/S (+0.020–0.030 bpr) and much larger at M/L (+0.10–0.11),
so the honest statement is **"significantly worse everywhere, and far worse at
scale,"** not strictly monotonic widening. The win also holds at matched **compute**,
not just matched parameters: gpt2's iso-FLOP frontier lies below mamba's across the
whole FLOP range. (`scripts/scaling_rigor.py`; see `docs/figures/scaling_rigor.png`.)

## Cross-architecture probing (Pillar 3, preliminary)

The scaling result above is *intrinsic loss*. Does the architecture that scales best at
next-residue prediction also produce the most **biologically decodable** representations?
We train frozen-model linear probes for **3-state secondary structure** (NetSurfP-2.0,
tested on CB513) on the residual stream of every released checkpoint at every layer, and
report the best layer (selected on validation) as **learned − baseline** — the same probe
on a random-init model of the identical architecture. Seed 0, all four scales:

| arch | XS | S | M | L | abs. macro-F1 (L) |
|---|---:|---:|---:|---:|---:|
| **gpt2** (AR) | +0.091 | +0.141 | +0.191 | **+0.232** | 0.671 |
| mamba (AR) | +0.090 | +0.102 | +0.136 | +0.186 | 0.658 |
| esm2 (MLM, encoder ref.) | +0.039 | +0.189 | +0.242 | +0.282 | 0.739 |

1. **All architectures scale** — SS decodability over the random baseline grows
   monotonically with model size for all three.
2. **The AR head-to-head cross-validates the headline:** gpt2 beats mamba on SS
   decodability at S/M/L (tied at XS), and gpt2's *absolute* score overtakes mamba at
   M/L (0.671 vs 0.658 at L) — the same crossover as bits-per-residue, now on a
   *downstream biological* task.
3. **esm2 wins absolute decodability** (bidirectional context sees both neighbours) —
   reported separately as the **encoder reference**, not an architecture win, per the
   encoder-vs-decoder caveat. Structure is most decodable in the **late layers**,
   deepening with scale (L best layer: gpt2 21/24, mamba 27/30, esm2 32/34).

> Caveat: **seed 0 only** (no error bars yet) and one of three planned label sources
> (NetSurfP/CB513); the Swiss-Prot + DSSP-from-AlphaFold triangulation and 3-seed error
> bars are next. (`scripts/run_probes.py`, `scripts/plot_probes.py`; `docs/figures/probes.png`.)

## A methodological note worth keeping

A 120-step, sub-sampled **calibration** run suggested the *opposite* — mamba
edging gpt2 (3.89 vs 3.95). The **fully-converged** runs reverse it at every
scale. Short / under-trained / sub-sampled comparisons mislabeled the winner;
convergence matters.

## Figures
- `docs/figures/scaling_laws.png` — L(N) power-law fits + α per architecture.
- `docs/figures/scaling_rigor.png` — bootstrap-CI α + iso-FLOP + per-scale gap.
- `docs/figures/scaling_curves.png` — metric vs. parameters.
- `docs/figures/training_curves.png` — validation loss vs. residues-seen.
- `docs/figures/probes.png` — Pillar 3: SS3 probe scaling transfer + layer-wise.

## Reproduce
```bash
python -m scripts.gen_release_configs           # the 36-cell grid
sbatch runs/prepare_uniref50.slurm              # build the corpus (after FASTA download)
sbatch runs/train_release.slurm                 # train (resumable, skip-complete)
python -m scripts.aggregate_results --release-root $NANOPROT_BASE_DIR/release
python -m scripts.scaling_laws    --release-root $NANOPROT_BASE_DIR/release
python -m scripts.make_model_card --release-root $NANOPROT_BASE_DIR/release
```
