"""
nanoprot training entry point.

Usage
-----

Single GPU (or CPU smoke test):

    python -m scripts.train --config configs/gpt2_d20_uniref50.yaml

Multi-GPU on a single node (8x GPUs):

    OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 \\
        -m scripts.train -- --config configs/gpt2_d20_uniref50.yaml

The entry point loads the YAML, validates + derives the config, and then
hands off to :func:`nanoprot.training.loop.train`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running both as ``python -m scripts.train`` (preferred) and as a
# bare script (``python scripts/train.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoprot.config import load_config  # noqa: E402
from nanoprot.training.loop import train  # noqa: E402


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
        choices=("cuda", "mps", "cpu", None),
        help="Force a device type (autodetected by default).",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    print(f"Loaded config: {cfg.name}  ({args.config})")
    state = train(cfg, device_type=args.device)
    print(f"Finished at step {state.step}, smoothed loss {state.smooth_loss:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
