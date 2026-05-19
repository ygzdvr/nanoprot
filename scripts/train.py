"""
nanoprot training entry point.

Usage
-----

Single GPU (or CPU smoke test):

    python -m scripts.train --config configs/gpt2_d20_uniref50.yaml

Multi-GPU on a single node (8x GPUs):

    OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 \\
        -m scripts.train -- --config configs/gpt2_d20_uniref50.yaml

The entry point loads the YAML, wires ``training.precision`` and
``training.flash_attention`` into the runtime *before* nanoprot is
imported (so the global ``COMPUTE_DTYPE`` / ``USE_FA3`` decisions
respect the config), then hands off to :func:`nanoprot.training.loop.train`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def _pre_import_setup_from_yaml(config_path: Path) -> None:
    """Set ``NANOPROT_DTYPE`` and an FA3-override env var from the YAML.

    This runs BEFORE any ``nanoprot`` modules are imported so the
    runtime-level constants (``COMPUTE_DTYPE``, ``USE_FA3``) pick up the
    user's config instead of falling back to GPU auto-detection.
    """
    with config_path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    training = raw.get("training", {}) or {}

    # Precision override (bf16 / fp32 / fp8) -> NANOPROT_DTYPE env var.
    # nanoprot.runtime maps "bf16"/"fp32"/"fp16" via _DTYPE_MAP; expand fp16 alias.
    precision = training.get("precision")
    if precision is not None:
        env_map = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16"}
        env_val = env_map.get(precision)
        if env_val is not None:
            os.environ.setdefault("NANOPROT_DTYPE", env_val)

    # FlashAttention-3 override: an opt-out switch only. nanoprot.attention
    # decides FA3 vs SDPA from CUDA capability; we add a SDPA-only override
    # if the user disables FA3 from the config.
    if training.get("flash_attention") is False:
        os.environ.setdefault("NANOPROT_DISABLE_FA3", "1")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a nanoprot YAML config (see nanoprot/configs/).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("cuda", "mps", "cpu"),
        help="Force a device type (autodetected by default).",
    )
    args = p.parse_args()

    if not args.config.is_file():
        sys.exit(f"config not found: {args.config}")

    # Honor the config's precision + flash_attention BEFORE nanoprot imports.
    _pre_import_setup_from_yaml(args.config)

    # Allow ``python scripts/train.py`` as well as ``python -m scripts.train``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from nanoprot.config import load_config
    from nanoprot.training.loop import train

    cfg = load_config(args.config)
    print(f"Loaded config: {cfg.name}  ({args.config})")
    state = train(cfg, device_type=args.device)
    print(f"Finished at step {state.step}, smoothed loss {state.smooth_loss:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
