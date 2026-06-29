#!/usr/bin/env python3
"""Paper A Fig. 5 — the "where": deep, deepening decoding. Reads the trajectory probe sidecars
(`trajectory_results/traj_*.json`), takes the best-layer-by-val *relative depth* per checkpoint
(rel_depth of the layer == ``best_layer_by_val``), averages over concepts × seeds per (arch, scale)
cell, and plots relative decoding depth vs training step. Reproduces the numbers in
`docs/atlas_where_layers.md` (deep: 0.7–0.98 for the AR decoders, shallower for the encoder;
deepening migration over training). Reported coarsely — per-concept depths are NOT claimed.
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Cells the atlas reports (sidecars complete); label + colour.
CELLS = [("gpt2", "L"), ("gpt2", "M"), ("mamba", "L"), ("esm2", "M")]
COL = {("gpt2", "L"): "#4C72B0", ("gpt2", "M"): "#8FB0D9", ("mamba", "L"): "#C44E52",
       ("esm2", "M"): "#55A868"}


def best_rel(ck):
    bl = ck["best_layer_by_val"]
    for r in ck["per_layer"]:
        if r["layer"] == bl:
            return float(r["rel_depth"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path(".cache/nanoprot/trajectory_results"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures/where_layers"))
    a = ap.parse_args()

    # cell -> list of per-sidecar [(step, best_rel)] curves
    curves = defaultdict(list)
    for fn in glob.glob(str(a.results_dir / "traj_*.json")):
        d = json.load(open(fn))
        cell = (d["arch"], d["scale"])
        if cell not in COL:
            continue
        cur = [(c["step"], best_rel(c)) for c in d["checkpoints"]]
        cur = [(s, r) for s, r in cur if r is not None]
        if len(cur) >= 4:
            curves[cell].append(sorted(cur))

    print("cell | n_sidecars | rel_depth early(25%)->final | migration")
    plot_data = {}
    for cell in CELLS:
        cs = curves.get(cell, [])
        if not cs:
            continue
        n_ck = min(len(c) for c in cs)                 # align by checkpoint index (shared log schedule)
        steps = np.median([[c[i][0] for i in range(n_ck)] for c in cs], axis=0)
        depth_by_idx = np.array([[c[i][1] for i in range(n_ck)] for c in cs])  # (n_sidecars, n_ck)
        mean = depth_by_idx.mean(axis=0); sd = depth_by_idx.std(axis=0)
        early, final = mean[n_ck // 4], mean[-1]
        print(f"{cell[0]:>5}-{cell[1]} | {len(cs):>2} | {early:.2f}->{final:.2f} | {final-early:+.2f}")
        plot_data[cell] = (steps, mean, sd)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        for cell in CELLS:
            if cell not in plot_data:
                continue
            steps, mean, sd = plot_data[cell]
            lab = f"{cell[0]}-{cell[1]}" + (" (encoder)" if cell[0] == "esm2" else "")
            ax.plot(steps, mean, "-o", color=COL[cell], lw=1.7, ms=3.5, label=lab)
            ax.fill_between(steps, mean - sd, mean + sd, color=COL[cell], alpha=0.12)
        ax.set_xscale("log"); ax.set_ylim(0, 1.0)
        ax.set_xlabel("training step"); ax.set_ylabel("best-layer relative depth")
        ax.set_title("Decoding is deep and deepens over training", fontsize=9)
        ax.axhline(1.0, color="0.7", ls=":", lw=0.8)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(a.out) + ".png", dpi=200, bbox_inches="tight")
        fig.savefig(str(a.out) + ".pdf", bbox_inches="tight")
        print(f"[where] wrote {a.out}.png/.pdf")
    except Exception as e:
        print(f"[where] figure skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
