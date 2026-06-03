#!/usr/bin/env python
"""
Plot validation loss-vs-tokens trajectories (the data dimension at fixed N).

Parses the periodic `[eval]` lines from the training-array stdout logs — the
within-run val-metric trajectory each cell already records — and plots
val-metric vs. tokens-seen, one curve per (arch, scale). Bigger models reach
lower loss; each curve also shows the *data* axis (loss as a fixed-size model
sees more residues).

CAVEAT: intermediate points are NOT LR-decayed (the run is mid-schedule), so a
trajectory is the training *dynamics*, not a clean converged L(D) scaling law —
only each curve's final point is the LR-decayed value. Read it as convergence
behaviour, not as a substitute for a data-budget sweep.

Reads metadata (params, batch size) from the release dir, so it pairs with
`scripts.scaling_laws`. Safe to run mid-training.

Usage:
  python -m scripts.plot_training_curves --release-root $NANOPROT_BASE_DIR/release \
      --logs-dir logs --out figures/training_curves
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

NAME_RE = re.compile(r"nanoprot-(?P<arch>gpt2|esm2|mamba)-(?P<scale>XS|S|M|L)-s(?P<seed>\d+)")
_LOADED_RE = re.compile(r"Loaded config:\s+(nanoprot-\S+?)\s")
_EVAL_RE = re.compile(r"\[eval\] step (\d+): .*val_bpr=([0-9.]+)")
_SCALE_ORDER = {"XS": 0, "S": 1, "M": 2, "L": 3}
_COLOR = {"gpt2": "#1f4e79", "mamba": "#b5471f", "esm2": "#3f7f3f"}
# scale -> alpha shade (small = faint, large = solid) within an arch's color
_SHADE = {"XS": 0.35, "S": 0.55, "M": 0.75, "L": 1.0}


def parse_trajectories(logs_dir: Path, pattern: str) -> Dict[str, List[Tuple[int, float]]]:
    """{cell_name: [(step, val_bpr), ...]} from the array stdout logs."""
    out: Dict[str, List[Tuple[int, float]]] = {}
    for f in sorted(logs_dir.glob(pattern)):
        try:
            text = f.read_text(errors="replace")
        except Exception:
            continue
        m = _LOADED_RE.search(text)
        if not m:
            continue
        cell = m.group(1)
        pts = [(int(s), float(b)) for s, b in _EVAL_RE.findall(text)]
        if pts:
            out.setdefault(cell, [])
            # keep the longest trajectory if a cell appears in multiple logs (resubmit)
            if len(pts) > len(out[cell]):
                out[cell] = sorted(pts)
    return out


def _batch_and_params(release_root: Path, cell: str) -> Tuple[Optional[int], Optional[int]]:
    d = release_root / cell
    metas = sorted(d.glob("meta_*.json"))
    if not metas:
        return None, None
    try:
        meta = json.loads(metas[-1].read_text())
        tbs = (meta.get("config") or {}).get("training", {}).get("total_batch_size", 524288)
        return int(tbs), meta.get("n_params")
    except Exception:
        return 524288, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--logs-dir", type=Path, default=Path("logs"))
    ap.add_argument("--pattern", type=str, default="*.nanoprot-release.out")
    ap.add_argument("--out", type=Path, default=Path("figures/training_curves"))
    args = ap.parse_args()

    traj = parse_trajectories(args.logs_dir, args.pattern)
    if not traj:
        print(f"No eval trajectories found in {args.logs_dir}/{args.pattern}.")
        return 0

    # one representative curve per (arch, scale): the lowest seed available.
    chosen: Dict[Tuple[str, str], str] = {}
    for cell in traj:
        m = NAME_RE.match(cell)
        if not m:
            continue
        key = (m["arch"], m["scale"])
        if key not in chosen or int(m["seed"]) < int(NAME_RE.match(chosen[key])["seed"]):
            chosen[key] = cell

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.3, "figure.dpi": 150,
    })
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.0), constrained_layout=True)
    panels = {("gpt2", "mamba"): (axa, "val bits-per-residue", "autoregressive", "a"),
              ("esm2",): (axb, "val masked-CE (bits)", "masked LM", "b")}

    n_curves = 0
    for archs, (ax, ylabel, title, tag) in panels.items():
        drew = False
        for (arch, scale), cell in sorted(chosen.items(), key=lambda kv: (kv[0][0], _SCALE_ORDER[kv[0][1]])):
            if arch not in archs:
                continue
            tbs, _ = _batch_and_params(args.release_root, cell)
            if not tbs:
                continue
            pts = traj[cell]
            xs = [s * tbs for s, _ in pts]
            ys = [b for _, b in pts]
            if len(xs) < 2:
                continue
            drew = True
            n_curves += 1
            ax.plot(xs, ys, marker="o", ms=2.5, color=_COLOR[arch],
                    alpha=_SHADE[scale], label=f"{arch}-{scale}")
            ax.annotate(scale, (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(3, 0), fontsize=6, color=_COLOR[arch])
        ax.set_xscale("log")
        ax.set_xlabel("residues seen")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.text(-0.18, 1.03, tag, transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.grid(True, color="0.92", lw=0.6, zorder=0)
        if drew:
            ax.legend(frameon=False, loc="upper right", ncol=1)
        else:
            ax.text(0.5, 0.5, "(no trajectories yet)", transform=ax.transAxes,
                    ha="center", va="center", color="0.6")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {args.out.with_suffix('.png')} and .pdf  ({n_curves} trajectories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
