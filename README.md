# nanoprot

**A minimal, hackable, config-driven training framework for protein language models.**

The best PLM that \$100 can buy — and the worst PLM you can iterate on the
fastest. Think *[Pythia](https://github.com/EleutherAI/pythia)* but for
protein language models: clean training code, multiple architectures, dense
checkpoint sweeps, and pretrained model releases small enough that a single
research group can re-train them.

> Status: **v0.1 — config-system scaffold.** No training loop yet. See the
> [roadmap](#roadmap) below for what lands when.

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

## What's in v0.1

```
nanoprot/
├── nanoprot/
│   ├── __init__.py
│   └── config.py              Pydantic schema + YAML loader + derivation rules
├── configs/
│   ├── README.md              config-system docs
│   └── gpt2_d20_uniref50.yaml reference nanoprot-d20 run config
├── scripts/
│   └── show_config.py         load + inspect any config (no training)
├── tests/
│   └── test_config.py
├── pyproject.toml             uv-managed Python project (pydantic + pyyaml only)
└── LICENSE
```

This first release is *only the design pattern*. The training loop, model
code, tokenizer, and data pipeline land in v0.2 (see roadmap).

## Quickstart (v0.1)

```bash
git clone https://github.com/ygzdvr/nanoprot.git
cd nanoprot

# minimal install (no torch yet — just config-system deps)
pip install -e ".[dev]"

# sanity-check the example config
python -m scripts.show_config configs/gpt2_d20_uniref50.yaml --estimate

# run the test suite
pytest
```

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

| Version | Lands | What it adds |
|---|---|---|
| **v0.1** | now | Config schema, YAML loader, derivation rules, example config, tests |
| **v0.2** | next | GPT-2 model code, tokenizer, UniRef50 data pipeline, training loop, checkpointing, single-node DDP |
| **v0.3** | next+1 | ESM-2-style masked encoder as a second architecture |
| **v0.4** | next+2 | Mamba / SSM as a third architecture |
| **v0.5** | next+3 | Hugging Face model release: pretrained nanoprot-{S,M,L} × {GPT-2, ESM-2, Mamba} checkpoints |
| **v0.6** | next+4 | Cross-architecture benchmark suite (probing + downstream) |
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
