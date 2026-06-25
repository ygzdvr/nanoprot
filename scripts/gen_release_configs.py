#!/usr/bin/env python
"""
Generate the nanoprot v0.5 model-release configs from the locked grid.

This is the **single source of truth** for the release: 3 architectures
(gpt2, esm2, mamba) x 4 scale rungs (XS/S/M/L, mirroring ESM-2's
~8M/35M/150M/650M ladder) x 3 seeds = 36 training configs, plus 3 throughput-
calibration probes. The full rationale lives in ``docs/v0.5_release_scope.md``.

Emits (under ``configs/release/``):
  nanoprot-{arch}-{scale}-s{seed}.yaml   36 training configs
  calib/calib-{arch}.yaml                 3 short throughput probes (S scale)
  MANIFEST.tsv                            one row per training config

Every emitted config is round-tripped through ``NanoprotConfig`` before it is
written, so a schema error surfaces here — not 13,000 steps into a GPU run.

Key release decisions encoded here (see scope doc S1):
  * Unified 33-token residue alphabet (tokenizer ``esm2``) for ALL archs, so
    per-residue probing aligns and gpt2<->mamba bpr is directly comparable.
  * Each arch follows its own natural width rule (d=64*depth for AR; the
    canonical ESM-2 grid for esm2). Archs are compared by scaling CURVES, not
    by contriving identical parameter counts.
  * esm2 uses SDPA (flash_attention: false) because its small rungs have
    head_dim < 64, which FA3 does not support; SDPA is exact and bidirectional-
    safe.

Usage:
  python -m scripts.gen_release_configs                  # default grid
  python -m scripts.gen_release_configs --intermediate   # + log-spaced ckpts
  python -m scripts.gen_release_configs --seeds 0 1 2     # override seeds
  python -m scripts.gen_release_configs --out configs/release --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Allow ``python scripts/gen_release_configs.py`` as well as ``-m``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoprot.config import NanoprotConfig  # noqa: E402


# ---------------------------------------------------------------------------
# The locked grid (docs/v0.5_release_scope.md S2). d_model is given explicitly
# for every cell so there are no derivation surprises in the release artifacts.
# ---------------------------------------------------------------------------

SCALES = ["XS", "S", "M", "L"]

# (depth, d_model, head_dim) per (arch, scale).
GRID: Dict[str, Dict[str, Dict[str, int]]] = {
    "gpt2": {
        "XS": dict(depth=6,  d_model=384,  head_dim=64),
        "S":  dict(depth=9,  d_model=576,  head_dim=64),
        "M":  dict(depth=14, d_model=896,  head_dim=64),
        "L":  dict(depth=23, d_model=1472, head_dim=64),
    },
    "mamba": {
        "XS": dict(depth=7,  d_model=448,  head_dim=64),
        "S":  dict(depth=11, d_model=704,  head_dim=64),
        "M":  dict(depth=18, d_model=1152, head_dim=64),
        "L":  dict(depth=29, d_model=1856, head_dim=64),
    },
    "esm2": {  # canonical ESM-2 grid: n_heads=20 fixed -> head_dim = d/20
        "XS": dict(depth=6,  d_model=320,  head_dim=16),
        "S":  dict(depth=12, d_model=480,  head_dim=24),
        "M":  dict(depth=30, d_model=640,  head_dim=32),
        "L":  dict(depth=33, d_model=1280, head_dim=64),
    },
}

# Per-arch model fields beyond the shared (depth/d_model/head_dim/...).
ARCH_MODEL_EXTRA: Dict[str, Dict[str, Any]] = {
    "gpt2": dict(
        rope=True, qk_norm=True, mlp_activation="relu_squared",
        logit_softcap=15.0, window_pattern="L",
    ),
    "mamba": dict(d_state=16, d_conv=4, expand=2, dt_rank=None, layer_norm="rmsnorm"),
    "esm2": dict(layer_norm="layernorm", mlp_activation="gelu"),
}

ARCH_OBJECTIVE = {"gpt2": "ar", "mamba": "ar", "esm2": "mlm"}
# weight_decay is held UNIFORM across all archs. The per-arch preset values
# (gpt2=0.28 from the nanochat English speedrun, esm2=0.01, mamba=0.0) are
# neither matched nor protein-tuned, so using them would confound the
# gpt2<->mamba comparison: a 28x difference in (applied) regularization is a
# second variable on top of architecture. A single uniform value keeps the
# comparison controlled (same regularization for everyone); 0.1 is the standard
# large-scale-pretraining weight decay (GPT-3 / Llama / OLMo). See scope doc S1.2.
WEIGHT_DECAY = 0.1
# esm2 forces SDPA (head_dim<64 on the small rungs; bidirectional). AR uses FA3.
ARCH_FLASH = {"gpt2": True, "mamba": True, "esm2": False}

# device_batch_size per scale (microbatch; total_batch_size is fixed at 524288
# so this only trades memory vs. gradient-accumulation, never the math).
DEVICE_BATCH = {"XS": 32, "S": 32, "M": 16, "L": 8}

# Constants held fixed across the entire grid (controlled comparison).
MAX_SEQ_LEN = 512
VOCAB_SIZE = 33
PARAM_DATA_RATIO = 12
TOTAL_BATCH_SIZE = 524_288
EVAL_TOKENS = 80 * 524_288  # ~41.9M residues per eval pass
N_INTERMEDIATE = 12         # log-spaced checkpoints when --intermediate

# Shared optimizer LRs (muP-transferred; the loop scales AdamW by (d/768)^-0.5
# internally, so the SAME base values are correct at every scale).
BASE_LRS = dict(matrix_lr=0.02, embedding_lr=0.3, unembedding_lr=0.008, scalar_lr=0.5)


def _log_spaced_steps(n_iter_est: int, k: int) -> List[int]:
    """k log-spaced step numbers in (0, n_iter_est), for training-dynamics."""
    if n_iter_est <= 1:
        return []
    out: List[int] = []
    for i in range(k):
        frac = (i + 1) / (k + 1)
        # geometric-ish spacing: denser early.
        step = int(round(n_iter_est ** frac))
        if 0 < step < n_iter_est and step not in out:
            out.append(step)
    return out


def _build_config_dict(arch: str, scale: str, seed: int, *, intermediate: bool,
                       n_intermediate: int = N_INTERMEDIATE, ckpt_root: str = "release",
                       full_logging: bool = False, eval_every: int = 250,
                       eval_tokens_mult: int = 80) -> Dict[str, Any]:
    g = GRID[arch][scale]
    objective = ARCH_OBJECTIVE[arch]
    name = f"nanoprot-{arch}-{scale}-s{seed}"

    model: Dict[str, Any] = dict(
        arch=arch,
        depth=g["depth"],
        d_model=g["d_model"],
        head_dim=g["head_dim"],
        max_seq_len=MAX_SEQ_LEN,
        vocab_size=VOCAB_SIZE,
    )
    model.update(ARCH_MODEL_EXTRA[arch])

    training: Dict[str, Any] = dict(
        objective=objective,
        total_residues=None,
        param_data_ratio=PARAM_DATA_RATIO,
        total_batch_size=TOTAL_BATCH_SIZE,
        device_batch_size=DEVICE_BATCH[scale],
        warmup_steps=40,
        warmdown_ratio=0.65,
        final_lr_frac=0.05,
        precision="bf16",
        flash_attention=ARCH_FLASH[arch],
    )
    if objective == "mlm":
        training["mlm_probability"] = 0.15

    save_steps: List[int] = []
    if intermediate:
        # Estimate num_iterations from the closed-form param count to place the
        # log-spaced checkpoints. The real count differs by <~1%, and the final
        # checkpoint is always saved regardless, so minor drift is harmless.
        probe = NanoprotConfig(
            name=name, seed=seed, model=dict(model),
            data={"shard_dir": "/tmp"}, checkpointing={"output_dir": "/tmp"},
            training=dict(training),
        )
        n_iter_est = int(PARAM_DATA_RATIO * probe.estimate_params() // TOTAL_BATCH_SIZE)
        save_steps = _log_spaced_steps(n_iter_est, n_intermediate)

    cfg: Dict[str, Any] = dict(
        name=name,
        seed=seed,
        model=model,
        tokenizer=dict(name="esm2"),  # 33-token residue alphabet for ALL archs
        data=dict(
            dataset="uniref50",
            shard_dir="${NANOPROT_BASE_DIR}/uniref50_parquet",
            packing="bos_aligned_best_fit",
            val_shard="last",
        ),
        optimizer=dict(name="muon_adamw", weight_decay=WEIGHT_DECAY, **BASE_LRS),
        training=training,
        eval=dict(eval_every=eval_every, eval_tokens=eval_tokens_mult * TOTAL_BATCH_SIZE,
                  metric="bpr"),
        logging=dict(
            run_name=f"{arch}-{scale}-s{seed}",
            wandb_project="nanoprot",
            wandb_mode="offline",
            history=full_logging,   # per-eval val_loss -> history.jsonl (matched-loss curves)
        ),
        checkpointing=dict(
            output_dir=f"${{NANOPROT_BASE_DIR}}/{ckpt_root}/{name}",
            save_every=-1,
            save_steps=save_steps,
        ),
    )
    return cfg


def _calib_config_dict(arch: str) -> Dict[str, Any]:
    """A short probe at S scale: ~120 steps. Doubles as an INTEGRATION GATE —
    it runs a couple of quick eval passes so the operator can confirm the full
    (objective, tokenizer) path trains and emits a *sane, decreasing* bpr
    (≈ log2(33) ≈ 5.04 bits at init) BEFORE launching the 36-cell array. This
    matters most for the AR archs, whose ``(objective=ar, tokenizer=esm2)``
    combination is new (the AR loader was originally built for BPE)."""
    cfg = _build_config_dict(arch, "S", seed=0, intermediate=False)
    cfg["name"] = f"calib-{arch}"
    cfg["logging"]["run_name"] = f"calib-{arch}"
    cfg["training"]["total_residues"] = 120 * TOTAL_BATCH_SIZE  # ~120 steps
    cfg["eval"]["eval_every"] = 50                              # 2 quick evals + final
    cfg["eval"]["eval_tokens"] = 2 * TOTAL_BATCH_SIZE           # small, fast
    cfg["checkpointing"]["output_dir"] = f"${{NANOPROT_BASE_DIR}}/release/_calib/{arch}"
    cfg["checkpointing"]["save_every"] = -1
    cfg["checkpointing"]["save_steps"] = []
    return cfg


def _validate(cfg_dict: Dict[str, Any]) -> NanoprotConfig:
    """Round-trip through env-resolving placeholders so derivation runs."""
    # Replace ${...} so model_validate doesn't choke on a missing var; the real
    # run resolves them via os.path.expandvars at load time.
    safe = yaml.safe_load(yaml.safe_dump(cfg_dict))
    safe["data"]["shard_dir"] = "/tmp/uniref50_parquet"
    safe["checkpointing"]["output_dir"] = "/tmp/ckpt"
    return NanoprotConfig.model_validate(safe)


def _dump_yaml(cfg_dict: Dict[str, Any], path: Path, banner: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(cfg_dict, sort_keys=False, default_flow_style=False, width=100)
    path.write_text(banner + body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("configs/release"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--intermediate", action="store_true",
                    help="Add log-spaced intermediate checkpoints (training dynamics).")
    ap.add_argument("--n-intermediate", type=int, default=N_INTERMEDIATE,
                    help=f"Number of log-spaced intermediate checkpoints (default {N_INTERMEDIATE}).")
    ap.add_argument("--archs", nargs="+", default=["gpt2", "esm2", "mamba"],
                    choices=["gpt2", "esm2", "mamba"], help="Subset of architectures to emit.")
    ap.add_argument("--scales", nargs="+", default=SCALES, choices=SCALES,
                    help="Subset of scale rungs to emit.")
    ap.add_argument("--ckpt-root", default="release",
                    help="Checkpoint dir prefix under NANOPROT_BASE_DIR. Use e.g. release_traj "
                         "for the trajectory re-train so the released release/ ckpts aren't clobbered.")
    ap.add_argument("--skip-calib", action="store_true", help="Do not emit calibration probes.")
    ap.add_argument("--full-logging", action="store_true",
                    help="Write per-step/per-eval history.jsonl (val_loss curve) — needed for "
                         "the trajectory matched-loss figure.")
    ap.add_argument("--eval-every", type=int, default=250,
                    help="Eval interval in steps (use a smaller value for trajectory runs so "
                         "val_loss is sampled densely enough to align with checkpoints).")
    ap.add_argument("--eval-tokens-mult", type=int, default=80,
                    help="eval_tokens = this x total_batch_size (use ~8 for cheap frequent "
                         "trajectory evals; default 80 = ~42M residues).")
    ap.add_argument("--force", action="store_true", help="Overwrite existing configs.")
    args = ap.parse_args()

    out = args.out
    if out.exists() and any(out.glob("*.yaml")) and not args.force:
        sys.exit(f"{out} already contains configs; pass --force to overwrite.")

    manifest_rows: List[str] = [
        "name\tarch\tscale\tseed\tdepth\td_model\tn_heads\test_params\tnum_iter_est\tconfig"
    ]
    n_written = 0
    summary: List[str] = []

    for arch in args.archs:
        for scale in args.scales:
            for seed in args.seeds:
                cfg_dict = _build_config_dict(arch, scale, seed, intermediate=args.intermediate,
                                              n_intermediate=args.n_intermediate,
                                              ckpt_root=args.ckpt_root,
                                              full_logging=args.full_logging,
                                              eval_every=args.eval_every,
                                              eval_tokens_mult=args.eval_tokens_mult)
                validated = _validate(cfg_dict)            # raises on schema error
                est = validated.estimate_params()
                n_iter = int(PARAM_DATA_RATIO * est // TOTAL_BATCH_SIZE)
                name = cfg_dict["name"]
                fpath = out / f"{name}.yaml"
                banner = (
                    f"# AUTO-GENERATED by scripts/gen_release_configs.py — DO NOT EDIT BY HAND.\n"
                    f"# nanoprot v0.5 release: {arch} {scale} (seed {seed}).\n"
                    f"# ~{est/1e6:.1f}M params, ~{n_iter:,} steps, "
                    f"objective={ARCH_OBJECTIVE[arch]}, tokenizer=esm2 (V=33).\n"
                    f"# Regenerate with: python -m scripts.gen_release_configs\n\n"
                )
                _dump_yaml(cfg_dict, fpath, banner)
                manifest_rows.append(
                    f"{name}\t{arch}\t{scale}\t{seed}\t{validated.model.depth}\t"
                    f"{validated.model.d_model}\t{validated.model.n_heads}\t"
                    f"{est}\t{n_iter}\t{fpath}"
                )
                n_written += 1
                if seed == args.seeds[0]:
                    summary.append(
                        f"  {arch:>5} {scale:>2}: depth={validated.model.depth:>2} "
                        f"d={validated.model.d_model:>4} ~{est/1e6:6.1f}M  ~{n_iter:>6,} steps"
                    )

    # Calibration probes (one per arch, S scale).
    for arch in ([] if args.skip_calib else args.archs):
        cfg_dict = _calib_config_dict(arch)
        _validate(cfg_dict)
        banner = (
            f"# AUTO-GENERATED throughput probe ({arch}, S scale, ~120 steps).\n"
            f"# Run via runs/calibrate_throughput.slurm; read tok/s from the log.\n\n"
        )
        _dump_yaml(cfg_dict, out / "calib" / f"calib-{arch}.yaml", banner)

    (out / "MANIFEST.tsv").write_text("\n".join(manifest_rows) + "\n")

    n_calib = 0 if args.skip_calib else len(args.archs)
    print(f"Wrote {n_written} training configs + {n_calib} calibration probes to {out}/")
    print(f"Manifest: {out / 'MANIFEST.tsv'}")
    print("\nGrid (seed {} shown; all {} seeds emitted):".format(args.seeds[0], len(args.seeds)))
    print("\n".join(summary))
    print(f"\nTotal training runs: {n_written}  "
          f"(intermediate checkpoints: {'ON' if args.intermediate else 'off'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
