"""
Load and pretty-print a nanoprot config, with all derived fields filled in.

Useful for sanity-checking before kicking off a long training run:

    python -m scripts.show_config configs/gpt2_d20_uniref50.yaml

Adds an optional ``--estimate`` flag that prints the closed-form parameter
count and Chinchilla-derived training-token budget for the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running both as a module (``python -m scripts.show_config``) and as a
# bare script (``python scripts/show_config.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoprot.config import load_config  # noqa: E402


def _format_number(n: float) -> str:
    """Human-readable number with suffix, e.g. 1.17B, 524.3K."""
    n = float(n)
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000.0 or unit == "T":
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}P"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="Path to a nanoprot YAML config (see nanoprot/configs/).",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Print parameter-count and total-residues estimates from the config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Pretty-print the fully-resolved config.
    print(json.dumps(cfg.model_dump(), indent=2, default=str))

    if args.estimate:
        n_params = cfg.estimate_params()
        n_residues = cfg.total_residues()
        n_iter = n_residues // cfg.training.total_batch_size
        print()
        print("=" * 60)
        print(f"  Run name           : {cfg.name}")
        print(f"  Architecture       : {cfg.model.arch}")
        print(
            f"  Model size         : depth={cfg.model.depth}, "
            f"d_model={cfg.model.d_model}, n_heads={cfg.model.n_heads}"
        )
        print(f"  Est. parameters    : {_format_number(n_params)}  ({n_params:,})")
        print(
            f"  Training residues  : {_format_number(n_residues)} "
            f"({n_residues:,})"
        )
        print(f"  Optimizer steps    : {n_iter:,}")
        print(
            f"  Tokens / step      : {cfg.training.total_batch_size:,} "
            f"(device batch {cfg.training.device_batch_size})"
        )
        print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
