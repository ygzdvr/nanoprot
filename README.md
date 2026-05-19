# nanoprot

**A minimal, hackable, config-driven training framework for protein language models.**

The best PLM that \$100 can buy — and the worst PLM you can iterate on the
fastest. Think *[Pythia](https://github.com/EleutherAI/pythia)* but for
protein language models: clean training code, multiple architectures, dense
checkpoint sweeps, and pretrained model releases small enough that a single
research group can re-train them.

> Status: **v0.4.0 — three architectures.** GPT-2-style autoregressive
> decoders, ESM-2-style masked-LM encoders, AND Mamba selective-SSM
> language models, all driven from a single YAML config
> (`arch: gpt2|esm2|mamba`). Discriminated-union config schema; the
> training loop dispatches the right model, tokenizer, data loader, and
> objective from one place. Eval loop, checkpointing, single-node DDP via
> torchrun. 94 unit + integration tests. See the [roadmap](#roadmap) for
> what lands next.

---

## Why?

Most public protein language models are released as a small number of large
final checkpoints. That's enough for downstream fine-tuning, but it makes a
class of questions intractable:

- **When** during training does a model first encode a given biological concept?
- **How** stable is that representation across random seeds?
- **What** is the right architectural family for a given downstream task?

Answering any of these requires *many* retraining runs at non-trivial scale.
That is routine in NLP (Pythia, OLMo, lm-evaluation-harness) and almost
non-existent in protein modeling. nanoprot is built to make it routine here
too.

## Design principle

**One YAML file fully specifies a training run.** The user picks a model
architecture, points at data, and the training script takes care of the rest.
A single complexity dial (`model.depth`) determines width, head count,
learning rate, batch size, weight decay, and the training horizon through
μP-style scaling. Other fields can be overridden explicitly.

```yaml
name: my-tiny-run
model:
  arch: gpt2
  depth: 6
data:
  shard_dir: $NANOPROT_DATA_DIR/uniref50_parquet
checkpointing:
  output_dir: checkpoints/tiny
```

That's a complete config. Everything else picks up sensible defaults from
`nanoprot/config.py`.

## What's in v0.4

```
nanoprot/
├── nanoprot/
│   ├── __init__.py
│   ├── config.py              Pydantic schema with discriminated-union ModelConfig
│   ├── runtime.py             device + dtype detection, DDP init, logging
│   ├── attention.py           FA3 wrapper + SDPA fallback (causal + bidirectional)
│   ├── optim.py               Muon + AdamW (single-process + DDP variants)
│   ├── models/
│   │   ├── __init__.py        model registry (build_model, register_model)
│   │   ├── gpt2.py            decoder-only GPT-2-style transformer (AR)
│   │   ├── esm2.py            ESM-2-style encoder-only transformer (MLM)
│   │   └── mamba.py           Mamba selective-SSM language model (AR)
│   ├── tokenizers/
│   │   ├── bpe.py             BPE tokenizer (UniRef50, V=50,256)
│   │   ├── residue.py         single-letter residue tokenizer alternative
│   │   └── esm2.py            33-token ESM-2 alphabet (matches the public release)
│   ├── data/
│   │   ├── dataset.py         UniRef50 parquet shard reader
│   │   ├── dataloader.py      BOS-aligned best-fit AR packing loader
│   │   ├── mlm.py             BERT-style MLM masking (15% / 80/10/10)
│   │   └── builder.py         objective + tokenizer dispatch (build_data_loader)
│   ├── training/
│   │   ├── loop.py            end-to-end loop with eval (config -> trained model)
│   │   └── checkpoint.py      save/load model + optimizer + meta
│   └── eval/
│       ├── loss.py            bits-per-byte / bits-per-residue eval
│       └── protein.py         protein-specific eval (validity, AA freq)
├── configs/
│   ├── README.md               config-system docs
│   ├── gpt2_d20_uniref50.yaml  reference 1.17 B-param AR run (gpt2 + bpe + AR)
│   ├── esm2_8M_uniref50.yaml   smallest ESM-2 scale (esm2 + esm2 tokenizer + MLM)
│   ├── esm2_650M_uniref50.yaml matches facebook/esm2_t33_650M_UR50D footprint
│   └── mamba_small_uniref50.yaml ~30 M-param Mamba selective SSM (mamba + bpe + AR)
├── scripts/
│   ├── train.py               training entry point (single GPU or torchrun)
│   └── show_config.py         load + inspect any config (no training)
├── tests/                     94 tests; 11 of them exercise the model end-to-end
└── pyproject.toml             uv-managed (pydantic + pyyaml + torch)
```

## Quickstart

```bash
git clone https://github.com/ygzdvr/nanoprot.git
cd nanoprot

# install (CPU-only dev; add the [gpu] extra on Hopper for FA3)
pip install -e ".[dev]"

# sanity-check a config (no training, no GPU needed)
python -m scripts.show_config configs/gpt2_d20_uniref50.yaml --estimate
python -m scripts.show_config configs/esm2_650M_uniref50.yaml --estimate
python -m scripts.show_config configs/mamba_small_uniref50.yaml --estimate

# run the test suite (94 tests total)
pytest -m "not slow"        # 83 fast tests in ~10 s
pytest                       # all 94 (includes ~3-min Mamba/loop integrations)

# launch training, single device — GPT-2 (autoregressive):
python -m scripts.train --config configs/gpt2_d20_uniref50.yaml

# launch training, single device — ESM-2 (masked LM):
python -m scripts.train --config configs/esm2_650M_uniref50.yaml

# launch training, single device — Mamba (selective SSM, autoregressive):
python -m scripts.train --config configs/mamba_small_uniref50.yaml

# launch training, torchrun on 8 GPUs:
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 \\
    -m scripts.train -- --config configs/esm2_650M_uniref50.yaml
```

Swapping between architectures is purely a config change. The training
loop dispatches the right model (gpt2 / esm2 / mamba), the right
tokenizer (`bpe` -> UniRef50 BPE V=50,256; `esm2` -> 33-token ESM-2
alphabet), and the right data loader (`ar` -> AR packing; `mlm` ->
packing + 15% / 80/10/10 BERT masking).

Expected output of `show_config --estimate`:

```
============================================================
  Run name           : nanoprot-d20-uniref50
  Architecture       : gpt2
  Model size         : depth=20, d_model=1280, n_heads=10
  Est. parameters    : 521.87M  (521,871,360)
  Training residues  : 6.26B  (6,262,456,320)
  Optimizer steps    : 11,944
  Tokens / step      : 524,288 (device batch 32)
============================================================
```

(The closed-form estimate is intentionally a lower bound — it counts
attention + MLP weights + token embeddings + lm_head, but omits the
ResFormer-style value embeddings on alternating layers and a handful of
scalar parameters. The real parameter count for `nanoprot-d20` is **1.17 B**;
that is printed once the model is instantiated in v0.2.)

## Roadmap

| Version | Status | What it adds |
|---|---|---|
| **v0.1** | shipped | Config schema, YAML loader, derivation rules, example config, tests |
| **v0.2** | shipped | GPT-2 model, BPE tokenizer, UniRef50 packing loader, Muon+AdamW optimizer, FA3 with SDPA fallback, training loop, checkpointing, single-node DDP via torchrun |
| **v0.2.1** | shipped | Patches: fixed `save_checkpoint` + dataloader signature regressions, wired `training.precision` and `training.flash_attention` through, added CPU end-to-end integration test, dump_config round-trip, seed inheritance, improved closed-form parameter estimate (includes value embeddings) |
| **v0.3** | shipped | Discriminated-union `ModelConfig` (gpt2 \| esm2), ESM-2 encoder model + 33-token ESM-2 tokenizer, BERT-style MLM data loader (15% / 80/10/10), `training.objective` (ar \| mlm) loop dispatch, eval loop integration (val_bpr + best_val_bpr), bidirectional-attention SDPA path, 80 tests total, ESM-2 8M + 650M config presets |
| **v0.4** | **shipped** | Mamba selective-SSM model (third arch in the discriminated union); depthwise causal conv + selective scan + gating + Pre-RMSNorm; pure-PyTorch reference scan (works on CPU/GPU/MPS); causal-by-construction so AR training pipeline reuses; new `MambaModelConfig` with `d_state`, `d_conv`, `expand`, auto-derived `dt_rank`; 14 Mamba tests including causality + scan correctness; `mamba_small_uniref50.yaml` config preset |
| **v0.5** | next+2 | Hugging Face model release: pretrained nanoprot-{S,M,L} × {GPT-2, ESM-2, Mamba} checkpoints |
| **v0.6** | next+3 | Cross-architecture benchmark suite (probing + downstream) |
| **v1.0** | when stable | Paper-ready release for MLSB / similar venue |

## Comparison to related projects

| | Pythia | OLMo | EleutherAI lm-evaluation-harness | nanoprot |
|---|---|---|---|---|
| Domain | NLP | NLP | NLP eval | **proteins** |
| Architectures | GPT-NeoX | OLMo | (eval-only) | GPT-2, ESM-2, Mamba (planned) |
| Released scales | 70M–12B | 1B–7B | — | tiny, S, M, L (planned) |
| Per-scale checkpoints | 154 | many | — | configurable sweep |
| Training-time studies | ✅ | partial | — | **first-class** |
| Config-driven | partial | yes | n/a | **yes (YAML)** |

## Adapted from

The training-loop ergonomics are inherited from Andrej Karpathy's
[nanochat](https://github.com/karpathy/nanochat) and
[nanoGPT](https://github.com/karpathy/nanoGPT), with proteins-specific
adaptations (BPE tokenizer on UniRef50, packing-aware loader, protein-eval
harness, single-knob depth scaling).

## Citing

A paper describing nanoprot will be linked here once available. Until then,
please cite the repository:

```bibtex
@software{nanoprot,
  author  = {Devre, H. Yagiz},
  title   = {nanoprot: a minimal training framework for protein language models},
  year    = {2026},
  url     = {https://github.com/ygzdvr/nanoprot},
  version = {0.1.0},
}
```

## License

MIT — see [`LICENSE`](LICENSE).
