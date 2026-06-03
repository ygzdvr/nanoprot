#!/usr/bin/env python
"""
Generate a DATA-BUDGET sweep — the runs needed for a full L(N, D) scaling law.

The release grid is compute-optimal (D = 12 N), so N and D co-vary and only the
combined exponent is identifiable. This sweep breaks that: it holds the model
size(s) fixed and trains each at several token budgets (param_data_ratio in
{3, 6, 12, 24, 48, 96} by default → D from 0.25x to 8x compute-optimal). Each run
is fully LR-decayed for its own budget (the schedule adapts to num_iterations),
so the points are clean — letting you fit

    L(N, D) = E + A * N^(-alpha) + B * D^(-beta)

and recover the isolated parameter- and data-exponents.

This only WRITES configs (into configs/sweep/) — it launches nothing. Submit with
runs/train_sweep.slurm when ready. Default scope (1 size, 6 budgets, 1 seed, 3
archs) = 18 runs; widen with the flags.

Usage:
  python -m scripts.gen_sweep_configs                       # default S-scale sweep
  python -m scripts.gen_sweep_configs --scales S M --ratios 3 6 12 24 48 96 --seeds 0 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gen_release_configs import (  # noqa: E402
    _build_config_dict, _validate, _dump_yaml, PARAM_DATA_RATIO, TOTAL_BATCH_SIZE,
)


def _sweep_config(arch: str, scale: str, ratio: float, seed: int) -> dict:
    cfg = _build_config_dict(arch, scale, seed, intermediate=False)
    rtag = f"{ratio:g}"
    name = f"nanoprot-sweep-{arch}-{scale}-r{rtag}-s{seed}"
    cfg["name"] = name
    cfg["training"]["param_data_ratio"] = ratio
    cfg["training"]["total_residues"] = None  # derived = ratio * params
    cfg["logging"]["run_name"] = f"sweep-{arch}-{scale}-r{rtag}-s{seed}"
    cfg["checkpointing"]["output_dir"] = f"${{NANOPROT_BASE_DIR}}/sweep/{name}"
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("configs/sweep"))
    ap.add_argument("--archs", nargs="+", default=["gpt2", "esm2", "mamba"])
    ap.add_argument("--scales", nargs="+", default=["S"])
    ap.add_argument("--ratios", type=float, nargs="+", default=[3, 6, 12, 24, 48, 96])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and any(args.out.glob("*.yaml")) and not args.force:
        sys.exit(f"{args.out} already has configs; pass --force.")

    rows: List[str] = ["name\tarch\tscale\tratio\tseed\test_params\tdata_residues\tsteps"]
    n = 0
    print(f"\n  Data-budget sweep -> {args.out}/\n")
    print(f"  {'arch':6} {'scale':5} {'ratio':>6} {'params':>8} {'residues':>9} {'steps':>8}")
    print("  " + "-" * 50)
    for arch in args.archs:
        for scale in args.scales:
            for ratio in args.ratios:
                for seed in args.seeds:
                    cfg = _sweep_config(arch, scale, ratio, seed)
                    v = _validate(cfg)
                    est = v.estimate_params()
                    D = int(ratio * est)
                    steps = D // TOTAL_BATCH_SIZE
                    fp = args.out / f"{cfg['name']}.yaml"
                    banner = (
                        f"# AUTO-GENERATED data-budget sweep ({arch} {scale}, "
                        f"ratio={ratio:g} -> D={D/1e9:.2f}B residues).\n"
                        f"# For L(N,D) scaling. Regenerate: python -m scripts.gen_sweep_configs\n\n"
                    )
                    _dump_yaml(cfg, fp, banner)
                    rows.append(f"{cfg['name']}\t{arch}\t{scale}\t{ratio:g}\t{seed}\t{est}\t{D}\t{steps}")
                    n += 1
                    if seed == args.seeds[0]:
                        print(f"  {arch:6} {scale:5} {ratio:>6g} {est/1e6:>7.1f}M "
                              f"{D/1e9:>8.2f}B {steps:>8,}")
    (args.out / "MANIFEST.tsv").write_text("\n".join(rows) + "\n")
    print(f"\n  Wrote {n} sweep configs + MANIFEST.tsv. Nothing launched.")
    print(f"  Submit when ready:  sbatch runs/train_sweep.slurm\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
