# nanoprot configs

Every nanoprot training run is fully specified by a single YAML file. Pass it
to the training entry point and the rest is determined.

## Anatomy of a config

```yaml
name: human-readable-run-id
seed: 42

model:        # architecture (only model.depth is required; the rest is derived)
tokenizer:    # which tokenizer to use
data:         # where the data lives + how it's packed
optimizer:    # optimizer + learning rates
training:     # batch sizes, schedule, precision
eval:         # validation cadence + metric
logging:      # wandb + run name
checkpointing:  # where checkpoints land
```

The full schema and per-field defaults live in `nanoprot/config.py`.

## What's required vs. derived

The "single complexity dial" of nanoprot is `model.depth`. From that:

| Field | Derivation rule |
|---|---|
| `model.d_model` | `64 * depth`, rounded up to a multiple of `model.head_dim` (default 128) |
| `model.n_heads` | `d_model / head_dim` |
| `model.n_kv_heads` | equal to `n_heads` (set explicitly for Group-Query Attention) |
| `training.total_residues` | `param_data_ratio * estimated_params` if left `null` (Chinchilla-style) |

So the smallest valid config is something like:

```yaml
name: my-tiny-run
model:
  depth: 6
data:
  shard_dir: $NANOPROT_DATA_DIR/uniref50_parquet
checkpointing:
  output_dir: checkpoints/tiny
```

Everything else picks up sensible defaults.

## Environment variables

Any string field is run through `os.path.expandvars`, so you can write
`$NANOPROT_DATA_DIR` or `${HOME}/data` in `data.shard_dir`,
`checkpointing.output_dir`, and `tokenizer.path` and they will be resolved at
config-load time.

## Presets

| File | Scale | Intended use |
|---|---|---|
| `gpt2_d20_uniref50.yaml` | 1.17 B params | The reference `nanoprot-d20` training run |

More presets land alongside their model architectures (v0.2 adds ESM-2
configs, v0.3 adds Mamba/SSM configs).

## Inspecting a config without running training

```bash
python -m scripts.show_config configs/gpt2_d20_uniref50.yaml
```

prints the fully-resolved, derived config as JSON. Useful for sanity-checking
the derivation rules before kicking off a multi-hour run.
