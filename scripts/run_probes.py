#!/usr/bin/env python
"""
Run the cross-architecture probe on ONE checkpoint against ONE cached probe dataset,
appending a results row (+ a per-layer JSON sidecar). A Slurm array loops over
(checkpoint x source); see docs/probing_harness_plan.md.

For each run: load the trained checkpoint, build a random-init baseline of the same
architecture, gather residual-stream features per layer (streamed, CPU-accumulated),
fit a standardized linear probe per layer, select the best layer on validation, and
report it on test as ``learned - baseline``.

Usage:
  python -m scripts.run_probes \
      --checkpoint $NANOPROT_BASE_DIR/release/nanoprot-gpt2-L-s0 \
      --probe-data $NANOPROT_BASE_DIR/probes/ss3_netsurfp \
      --out $NANOPROT_BASE_DIR/probe_results/ss3.csv \
      [--max-residues 200000] [--device cuda] [--epochs 300]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from nanoprot.training.checkpoint import load_pretrained  # noqa: E402
from nanoprot.eval.probe.extract import relative_depths  # noqa: E402
from nanoprot.eval.probe.labels import load_probe_dataset  # noqa: E402
from nanoprot.eval.probe.run import build_random_init, run_probe  # noqa: E402

NAME_RE = re.compile(r"nanoprot-(?P<arch>gpt2|esm2|mamba)-(?P<scale>XS|S|M|L)-s(?P<seed>\d+)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="A release cell dir.")
    ap.add_argument("--probe-data", type=Path, required=True, help="A cached probe dataset dir.")
    ap.add_argument("--out", type=Path, required=True, help="Results CSV (appended).")
    ap.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-residues", type=int, default=None,
                    help="Cap residues per split (bounds memory at large scale).")
    ap.add_argument("--metric", default=None,
                    choices=["macro_f1", "accuracy", "r2", "spearman"],
                    help="Default: macro_f1 (classification) / r2 (regression), by task.")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    name = NAME_RE.search(args.checkpoint.name)
    arch = name["arch"] if name else "?"
    scale = name["scale"] if name else "?"
    seed = int(name["seed"]) if name else -1

    model, cfg, meta, tok = load_pretrained(
        args.checkpoint, step=args.step, device=args.device, return_tokenizer=True)
    baseline = build_random_init(cfg.model, device=args.device)
    dataset = load_probe_dataset(args.probe_data)

    out = run_probe(
        model, baseline, dataset, tokenizer=tok, metric=args.metric,
        max_residues=args.max_residues, device=args.device,
        fit_kw={"epochs": args.epochs, "seed": args.seed, "device": args.device},
    )

    row = {
        "arch": arch, "scale": scale, "seed": seed, "n_params": meta.get("n_params"),
        "concept": out["concept"], "source": out["source"], "task": out["task"],
        "metric": out["metric"],
        "best_layer": out["best_layer"], "n_layers": len(out["learned"]["per_layer"]),
        "learned_test": round(out["learned_test"], 6),
        "baseline_test": round(out["baseline_test"], 6),
        "learned_minus_baseline": round(out["learned_minus_baseline"], 6),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.out.exists()
    with args.out.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    # Per-layer sidecar (relative depth + learned/baseline per layer) for the figures.
    rd = relative_depths(len(out["learned"]["per_layer"]))
    sidecar = {
        "row": row,
        "class_names": dataset.class_names,
        "best": {
            "layer": out["best_layer"],
            "learned_test": out["learned"]["best_test"],
            "baseline_test": out["baseline"]["best_test"],
        },
        "learned_per_layer": [
            {"layer": r["layer"], "rel_depth": rd[r["layer"]], "val": r["val"], "test": r["test"]}
            for r in out["learned"]["per_layer"]
        ],
        "baseline_per_layer": [
            {"layer": r["layer"], "rel_depth": rd[r["layer"]], "test": r["test"]}
            for r in out["baseline"]["per_layer"]
        ],
    }
    side = args.out.parent / f"{arch}-{scale}-s{seed}_{out['source']}_{out['concept']}.json"
    side.write_text(json.dumps(sidecar, indent=2))

    print(f"  {arch}-{scale}-s{seed} [{out['source']}/{out['concept']}]: "
          f"best L{out['best_layer']} ({out['metric']}) test={out['learned_test']:.4f} "
          f"baseline={out['baseline_test']:.4f} learned-baseline={out['learned_minus_baseline']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
