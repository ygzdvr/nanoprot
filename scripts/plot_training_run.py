#!/usr/bin/env python
"""
Visualise a single training run: loss / perplexity / accuracy / throughput /
compute-time, from the per-step history.jsonl a full-logging run writes
(logging.history=True), or reconstructed from a release-array stdout log.

Panels (drawn only if the data is present):
  (a) loss vs step      — train loss (+ EMA) and validation bpr/loss
  (b) perplexity & acc   — validation perplexity and token accuracy vs step
  (c) throughput         — residues/sec vs step (compute-time profile)
  (d) loss vs compute    — train/val loss vs cumulative FLOPs (6·N·D)

Usage:
  # full-logging run:
  python -m scripts.plot_training_run --history $DIR/history.jsonl --out fig
  # reconstruct from a release log (loss + tok/s + val_bpr only):
  python -m scripts.plot_training_run --from-log logs/JOB_0.nanoprot-release.out \
      --release-root $NANOPROT_BASE_DIR/release --out fig
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_TRAIN_RE = re.compile(
    r"step\s+(\d+)/\d+ \| loss ([\d.]+) \| smooth ([\d.]+) \| lr_mult [\d.]+ \| tok/s ([\d.eE+]+)")
_EVAL_RE = re.compile(r"\[eval\] step (\d+): val_loss=([\d.]+) val_bpr=([\d.]+)"
                      r"(?: ppl=([\d.]+))?(?: acc=([\d.]+))?")
_LOADED_RE = re.compile(r"Loaded config:\s+(nanoprot-\S+?)\s")
_BLUE, _GREEN, _GREY = "#1f4e79", "#3f7f3f", "0.6"


def _load_history(path: Path) -> Tuple[List[dict], List[dict]]:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return ([r for r in recs if r.get("type") == "train"],
            [r for r in recs if r.get("type") == "eval"])


def _from_log(path: Path, total_batch: int) -> Tuple[List[dict], List[dict], Optional[str]]:
    text = path.read_text(errors="replace")
    cell = (_LOADED_RE.search(text) or [None, None])[1] if _LOADED_RE.search(text) else None
    train = [{"step": int(s), "loss": float(l), "smooth_loss": float(sm),
              "tok_per_sec": float(t), "tokens": int(s) * total_batch}
             for s, l, sm, t in _TRAIN_RE.findall(text)]
    ev = []
    for s, vl, vb, ppl, acc in _EVAL_RE.findall(text):
        ev.append({"step": int(s), "val_loss": float(vl), "val_bpr": float(vb),
                   "tokens": int(s) * total_batch,
                   "val_ppl": float(ppl) if ppl else None,
                   "val_accuracy": float(acc) if acc else None})
    return train, ev, cell


def _meta_facts(release_root: Optional[Path], cell: Optional[str]) -> Tuple[int, Optional[float]]:
    """(total_batch_size, flops_per_token) from the cell's meta if available."""
    if release_root and cell:
        metas = sorted((release_root / cell).glob("meta_*.json"))
        if metas:
            m = json.loads(metas[-1].read_text())
            tbs = (m.get("config") or {}).get("training", {}).get("total_batch_size", 524288)
            return int(tbs), m.get("flops_per_token")
    return 524288, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--history", type=Path, help="A history.jsonl from a full-logging run.")
    g.add_argument("--from-log", type=Path, help="A release-array .out log to reconstruct from.")
    ap.add_argument("--release-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("figures/training_run"))
    args = ap.parse_args()

    cell = None
    if args.history:
        train, ev = _load_history(args.history)
        if args.release_root:
            cell = args.history.parent.name
        tbs, fpt = _meta_facts(args.release_root, cell)
    else:
        tbs, fpt = _meta_facts(args.release_root, None)
        train, ev, cell = _from_log(args.from_log, tbs)
        tbs, fpt = _meta_facts(args.release_root, cell)
    if not train and not ev:
        print("No records parsed.")
        return 0

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "figure.dpi": 150,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    (axa, axb), (axc, axd) = axes
    title = cell or (args.history.parent.name if args.history else "training run")
    fig.suptitle(title, fontsize=9)

    # (a) loss vs step
    if train:
        s = [r["step"] for r in train]
        axa.plot(s, [r["loss"] for r in train], color=_GREY, lw=0.6, alpha=0.6, label="train loss")
        axa.plot(s, [r["smooth_loss"] for r in train], color=_BLUE, lw=1.3, label="train (EMA)")
    if ev:
        axa.plot([r["step"] for r in ev], [r["val_loss"] for r in ev], color=_GREEN,
                 marker="o", ms=3, lw=1.0, label="val loss")
    axa.set_xlabel("step"); axa.set_ylabel("loss (nats)"); axa.set_title("loss")
    axa.text(-0.16, 1.04, "a", transform=axa.transAxes, fontsize=10, fontweight="bold")
    axa.legend(frameon=False)

    # (b) perplexity + accuracy (full-logging only)
    have_ppl = any(r.get("val_ppl") for r in ev)
    have_acc = any(r.get("val_accuracy") is not None for r in ev)
    if have_ppl:
        axb.plot([r["step"] for r in ev if r.get("val_ppl")],
                 [r["val_ppl"] for r in ev if r.get("val_ppl")], color=_GREEN, marker="o", ms=3)
        axb.set_ylabel("val perplexity", color=_GREEN)
    if have_acc:
        ax2 = axb.twinx(); ax2.spines.top.set_visible(False)
        ax2.plot([r["step"] for r in ev if r.get("val_accuracy") is not None],
                 [r["val_accuracy"] for r in ev if r.get("val_accuracy") is not None],
                 color=_BLUE, marker="s", ms=3)
        ax2.set_ylabel("val accuracy", color=_BLUE)
    if not (have_ppl or have_acc):
        axb.text(0.5, 0.5, "perplexity / accuracy\n(needs logging.history run)",
                 transform=axb.transAxes, ha="center", va="center", color="0.6", fontsize=7)
    axb.set_xlabel("step"); axb.set_title("perplexity & accuracy")
    axb.text(-0.16, 1.04, "b", transform=axb.transAxes, fontsize=10, fontweight="bold")

    # (c) throughput
    if train:
        axc.plot([r["step"] for r in train], [r["tok_per_sec"] for r in train],
                 color=_BLUE, lw=1.0)
    axc.set_xlabel("step"); axc.set_ylabel("residues / sec"); axc.set_title("throughput")
    axc.text(-0.16, 1.04, "c", transform=axc.transAxes, fontsize=10, fontweight="bold")

    # (d) loss vs compute (FLOPs = tokens * flops_per_token)
    if fpt and train:
        axd.plot([r["tokens"] * fpt for r in train], [r["smooth_loss"] for r in train],
                 color=_BLUE, lw=1.3, label="train (EMA)")
        if ev:
            axd.plot([r["tokens"] * fpt for r in ev], [r["val_loss"] for r in ev],
                     color=_GREEN, marker="o", ms=3, lw=1.0, label="val")
        axd.set_xscale("log"); axd.set_xlabel("training FLOPs"); axd.set_ylabel("loss (nats)")
        axd.legend(frameon=False)
    else:
        axd.text(0.5, 0.5, "loss vs FLOPs\n(needs flops_per_token from meta)",
                 transform=axd.transAxes, ha="center", va="center", color="0.6", fontsize=7)
    axd.set_title("compute efficiency")
    axd.text(-0.16, 1.04, "d", transform=axd.transAxes, fontsize=10, fontweight="bold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {args.out.with_suffix('.png')}  "
          f"({len(train)} train rows, {len(ev)} eval rows, cell={cell})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
