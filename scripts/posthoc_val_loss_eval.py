#!/usr/bin/env python
"""
Post-hoc validation-loss eval for trajectory checkpoints (Pillar 2 matched-loss figure).

The trajectory runs in release_traj/ predate the history-logging fix, so they wrote no
val_loss curve. Rather than re-train (the checkpoints are valid and the seed reproduces),
we recover val_loss EXACTLY at each saved checkpoint here: load the checkpoint, run the
model on a FIXED set of UniRef50 validation batches, record the LM loss. Exact
per-checkpoint val_loss beats interpolating a sparse eval curve.

The val batches are materialised ONCE per model and reused for every checkpoint, so each
checkpoint of a model is scored on identical data — the val_loss curve then reflects the
model, not val-data variation. Forward passes only; no training.

Emits one CSV row per (model, checkpoint): arch, scale, seed, step, val_loss, val_bpr,
train_residues. Join it with the trajectory probe CSVs (by arch/scale/seed/step) to draw
Delta-vs-val_loss.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from nanoprot.training.checkpoint import load_pretrained  # noqa: E402
from nanoprot.data.builder import build_data_loader  # noqa: E402

NAME_RE = re.compile(r"nanoprot-(?P<arch>gpt2|esm2|mamba)-(?P<scale>XS|S|M|L)-s(?P<seed>\d+)")


def list_checkpoint_steps(traj_dir: Path) -> List[int]:
    steps = []
    for p in glob.glob(str(Path(traj_dir) / "model_*.pt")):
        try:
            steps.append(int(Path(p).stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(steps)


def materialize_val_batches(cfg, num_batches: int, device: str) -> list:
    """A fixed list of ``num_batches`` (idx, targets) val batches, drawn once."""
    loader = build_data_loader(cfg, split="val", device=device)
    return [next(loader) for _ in range(num_batches)]


@torch.no_grad()
def val_loss_on_batches(model, batches) -> float:
    """Mean LM loss of ``model`` over a fixed list of (idx, targets) batches."""
    was_training = model.training
    model.eval()
    losses = [float(model(idx, targets=targets).detach()) for idx, targets in batches]
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def run(traj_dir: Path, *, eval_tokens: int, device: Optional[str] = None,
        steps: Optional[List[int]] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    all_steps = list_checkpoint_steps(traj_dir)
    if not all_steps:
        raise SystemExit(f"no model_*.pt checkpoints in {traj_dir}")
    steps = [s for s in all_steps if (steps is None or s in steps)]

    # cfg is embedded in every checkpoint's meta; take it from the last and fix the val set.
    _, cfg, _, _ = load_pretrained(traj_dir, step=all_steps[-1], device=device, return_tokenizer=True)
    dbs, seq = cfg.training.device_batch_size, cfg.model.max_seq_len
    num_batches = max(1, eval_tokens // (dbs * seq))
    batches = materialize_val_batches(cfg, num_batches, device)

    rows = []
    for step in steps:
        model, _, meta, _ = load_pretrained(traj_dir, step=step, device=device, return_tokenizer=True)
        loss = val_loss_on_batches(model, batches)
        del model
        rows.append({"step": step, "val_loss": loss, "val_bpr": loss / math.log(2),
                     "train_residues": meta.get("total_residues")})
    return rows, num_batches


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, required=True,
                    help="release_traj/nanoprot-{arch}-{scale}-s{seed} with saved checkpoints.")
    ap.add_argument("--out", type=Path, required=True, help="val-loss CSV (appended).")
    ap.add_argument("--eval-tokens", type=int, default=4_194_304,
                    help="Residues to average val_loss over (default ~4.2M = 8 x batch).")
    ap.add_argument("--steps", type=int, nargs="+", default=None, help="Subset of steps.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    m = NAME_RE.search(args.traj_dir.name)
    arch = m["arch"] if m else "?"
    scale = m["scale"] if m else "?"
    seed = int(m["seed"]) if m else -1

    rows, num_batches = run(args.traj_dir, eval_tokens=args.eval_tokens, device=args.device,
                            steps=args.steps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.out.exists()
    with args.out.open("a", newline="") as fh:
        cols = ["arch", "scale", "seed", "step", "val_loss", "val_bpr", "train_residues"]
        w = csv.DictWriter(fh, fieldnames=cols)
        if is_new:
            w.writeheader()
        for r in rows:
            w.writerow({"arch": arch, "scale": scale, "seed": seed, "step": r["step"],
                        "val_loss": round(r["val_loss"], 6), "val_bpr": round(r["val_bpr"], 6),
                        "train_residues": r["train_residues"]})

    print(f"  {arch}-{scale}-s{seed}: {len(rows)} ckpts, {num_batches} val batches, "
          f"final val_loss={rows[-1]['val_loss']:.4f} (bpr {rows[-1]['val_bpr']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
