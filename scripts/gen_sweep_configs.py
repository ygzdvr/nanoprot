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


def _sweep_steps(cfg_dict: dict, ratio: float) -> int:
    """Optimizer steps for a run: D // total_batch_size, D = ratio * est_params."""
    est = _validate(cfg_dict).estimate_params()
    return int(ratio * est) // TOTAL_BATCH_SIZE


def _sweep_config(arch: str, scale: str, ratio: float, seed: int,
                  full_logging: bool = True, ckpts_per_run: int = 10,
                  save_every: int | None = None) -> dict:
    cfg = _build_config_dict(arch, scale, seed, intermediate=False)
    rtag = f"{ratio:g}"
    name = f"nanoprot-sweep-{arch}-{scale}-r{rtag}-s{seed}"
    cfg["name"] = name
    cfg["training"]["param_data_ratio"] = ratio
    cfg["training"]["total_residues"] = None  # derived = ratio * params
    cfg["logging"]["run_name"] = f"sweep-{arch}-{scale}-r{rtag}-s{seed}"
    cfg["checkpointing"]["output_dir"] = f"${{NANOPROT_BASE_DIR}}/sweep/{name}"
    if full_logging:
        # Full per-step history + fine, cheap validation — this is the data the
        # detailed L(N,D) scaling analysis wants. eval_tokens is small so eval
        # every 10 steps stays cheap (~8 val batches).
        cfg["logging"]["history"] = True
        cfg["logging"]["log_every"] = 1
        cfg["eval"]["eval_every"] = 10
        cfg["eval"]["eval_tokens"] = 524288
        cfg["eval"]["compute_accuracy"] = True
    # Intermediate model-only checkpoints at a fixed step interval: an explicit
    # --save-every, else evenly spaced so each run yields ~ckpts_per_run of them.
    steps = _sweep_steps(cfg, ratio)
    cfg["checkpointing"]["save_every"] = (
        save_every if save_every else max(steps // max(ckpts_per_run, 1), 1)
    )
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("configs/sweep"))
    ap.add_argument("--archs", nargs="+", default=["gpt2", "esm2", "mamba"])
    ap.add_argument("--scales", nargs="+", default=["S"])
    ap.add_argument("--ratios", type=float, nargs="+", default=[3, 6, 12, 24, 48, 96])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--no-full-logging", action="store_true",
                    help="Disable per-step history + fine validation (on by default).")
    ap.add_argument("--ckpts-per-run", type=int, default=10,
                    help=("Intermediate (model-only) checkpoints saved evenly across "
                          "each run; save_every = steps // this. Default 10."))
    ap.add_argument("--save-every", type=int, default=None,
                    help=("Explicit fixed step interval for intermediate checkpoints; "
                          "overrides --ckpts-per-run. Each run also always saves at the end."))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and any(args.out.glob("*.yaml")) and not args.force:
        sys.exit(f"{args.out} already has configs; pass --force.")

    rows: List[str] = ["name\tarch\tscale\tratio\tseed\test_params\tdata_residues\tsteps\tsave_every\tn_ckpts"]
    n = 0
    total_ckpts = 0
    print(f"\n  Data-budget sweep -> {args.out}/\n")
    print(f"  {'arch':6} {'scale':5} {'ratio':>6} {'params':>8} {'residues':>9} {'steps':>8} "
          f"{'save@':>6} {'ckpts':>6}")
    print("  " + "-" * 62)
    for arch in args.archs:
        for scale in args.scales:
            for ratio in args.ratios:
                for seed in args.seeds:
                    cfg = _sweep_config(arch, scale, ratio, seed,
                                        full_logging=not args.no_full_logging,
                                        ckpts_per_run=args.ckpts_per_run,
                                        save_every=args.save_every)
                    v = _validate(cfg)
                    est = v.estimate_params()
                    D = int(ratio * est)
                    steps = D // TOTAL_BATCH_SIZE
                    save_every = cfg["checkpointing"]["save_every"]
                    n_ckpts = (steps // save_every) if save_every > 0 else 0
                    total_ckpts += n_ckpts + 1  # + the final checkpoint
                    fp = args.out / f"{cfg['name']}.yaml"
                    banner = (
                        f"# AUTO-GENERATED data-budget sweep ({arch} {scale}, "
                        f"ratio={ratio:g} -> D={D/1e9:.2f}B residues, {steps} steps).\n"
                        f"# Intermediate model-only checkpoints every {save_every} steps "
                        f"(~{n_ckpts}) + final.\n"
                        f"# For L(N,D) scaling. Regenerate: python -m scripts.gen_sweep_configs\n\n"
                    )
                    _dump_yaml(cfg, fp, banner)
                    rows.append(f"{cfg['name']}\t{arch}\t{scale}\t{ratio:g}\t{seed}\t{est}\t{D}\t"
                                f"{steps}\t{save_every}\t{n_ckpts}")
                    n += 1
                    if seed == args.seeds[0]:
                        print(f"  {arch:6} {scale:5} {ratio:>6g} {est/1e6:>7.1f}M "
                              f"{D/1e9:>8.2f}B {steps:>8,} {save_every:>6} {n_ckpts:>6}")
    (args.out / "MANIFEST.tsv").write_text("\n".join(rows) + "\n")
    print(f"\n  Wrote {n} sweep configs + MANIFEST.tsv. Nothing launched.")
    print(f"  Intermediate checkpoints are MODEL-ONLY (no optimizer); ~{total_ckpts} "
          f"checkpoints total across all runs.")
    print(f"  Submit when ready:  sbatch runs/train_sweep.slurm\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
