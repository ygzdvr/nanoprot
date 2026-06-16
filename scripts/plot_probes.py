#!/usr/bin/env python
"""
Figures for the cross-architecture probing result (Pillar 3).

Reads the results CSV + per-layer JSON sidecars that ``run_probes.py`` writes, and
draws (docs/probing_harness_plan.md §7):

  (a) scaling transfer: probe score (learned - baseline, mean+/-std over seeds) vs
      parameters, one line per architecture. Source is the *triangulation* axis
      (line style), so a replicated arch ordering across NetSurfP / Swiss-Prot /
      DSSP shows up as parallel families. "Does gpt2's scaling edge transfer to
      biology?"
  (b) layer-wise: probe test metric vs relative depth, one line per arch at the
      largest scale. "Where in the network does structure live?"

Both panels draw only what is present, so this is safe to run on partial results.

Usage:
  python -m scripts.plot_probes --results $NANOPROT_BASE_DIR/probe_results/ss3.csv \
      --sidecar-dir $NANOPROT_BASE_DIR/probe_results --out docs/figures/probes
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_COLOR = {"gpt2": "#1f4e79", "mamba": "#b5471f", "esm2": "#3f7f3f"}
_MARK = {"gpt2": "o", "mamba": "s", "esm2": "D"}
_SCALE_ORDER = {"XS": 0, "S": 1, "M": 2, "L": 3}


def load_results(path: Path) -> List[dict]:
    with Path(path).open() as fh:
        return list(csv.DictReader(fh))


def load_sidecars(dirpath: Path) -> List[dict]:
    out = []
    for p in sorted(Path(dirpath).glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "learned_per_layer" in d and "row" in d:
            out.append(d)
    return out


def aggregate_scaling(rows: List[dict], value: str = "learned_minus_baseline") -> Dict[Tuple[str, str], list]:
    """(source, arch) -> sorted list of (n_params, scale, mean, std) over seeds."""
    groups: Dict[Tuple[str, str, str], List[float]] = {}
    nparams: Dict[Tuple[str, str], float] = {}
    for r in rows:
        if not r.get(value):
            continue
        groups.setdefault((r["source"], r["arch"], r["scale"]), []).append(float(r[value]))
        if r.get("n_params"):
            nparams[(r["arch"], r["scale"])] = float(r["n_params"])
    out: Dict[Tuple[str, str], list] = {}
    for (source, arch, scale), vals in groups.items():
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
        out.setdefault((source, arch), []).append((nparams.get((arch, scale), 0.0), scale, mu, sd))
    for key in out:
        out[key].sort()
    return out


def _largest_scale_sidecars(sidecars: List[dict], source: str) -> Dict[str, dict]:
    """Per arch, the sidecar at the largest scale for one source."""
    best: Dict[str, dict] = {}
    for d in sidecars:
        row = d["row"]
        if row.get("source") != source:
            continue
        arch = row["arch"]
        rank = _SCALE_ORDER.get(row["scale"], -1)
        if arch not in best or rank > _SCALE_ORDER.get(best[arch]["row"]["scale"], -1):
            best[arch] = d
    return best


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "figure.dpi": 150,
    })


_SRC_ORDER = {"netsurfp": 0, "swissprot": 1, "dssp": 2}   # canonical-benchmark first


def figure(rows: List[dict], sidecars: List[dict], out: Path, metric: str = "macro_f1") -> None:
    """One scaling panel PER source (each: 3 archs, shared y-axis so the replication is
    obvious) + a layer-wise panel. Adaptive grid; degrades to 1x2 for a single source."""
    _style()
    agg = aggregate_scaling(rows)
    sources = sorted({s for (s, _a) in agg}, key=lambda s: (_SRC_ORDER.get(s, 9), s))
    n_src = len(sources)
    n_panels = n_src + 1                                   # scaling-per-source + layer-wise
    ncols = n_panels if n_panels <= 3 else 2
    nrows = -(-n_panels // ncols)                          # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows),
                             constrained_layout=True, squeeze=False)
    flat = [ax for row in axes for ax in row]
    tag = "abcdefghij"

    # shared y-range across sources so the panels are directly comparable
    all_ys = [p[2] for (_s, _a), pts in agg.items() for p in pts if p[0] > 0]
    ylim = (min(all_ys + [0.0]) - 0.02, max(all_ys) + 0.02) if all_ys else None

    # (a..) one scaling panel per source
    for i, source in enumerate(sources):
        ax = flat[i]
        for arch in ("gpt2", "mamba", "esm2"):
            pts = agg.get((source, arch))
            if not pts:
                continue
            xs = [p[0] for p in pts if p[0] > 0]
            ys = [p[2] for p in pts if p[0] > 0]
            es = [p[3] for p in pts if p[0] > 0]
            if xs:
                ax.errorbar(xs, ys, yerr=es, color=_COLOR.get(arch, "0.3"),
                            marker=_MARK.get(arch, "o"), ms=4, lw=1.3, capsize=2, label=arch)
        ax.axhline(0, color="0.6", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_xlabel("parameters")
        if ylim:
            ax.set_ylim(*ylim)
        if i == 0:
            ax.set_ylabel(f"learned − baseline ({metric})")
        ax.set_title(source); ax.grid(True, color="0.93", lw=0.6)
        ax.text(-0.18, 1.04, tag[i], transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.legend(frameon=False, fontsize=6.5)

    # (last) layer-wise test metric vs relative depth (primary source, largest scale)
    axb = flat[n_src]
    primary = sources[0] if sources else (rows[0]["source"] if rows else None)
    drew = False
    if primary:
        for arch, d in sorted(_largest_scale_sidecars(sidecars, primary).items()):
            pl = d["learned_per_layer"]
            axb.plot([p["rel_depth"] for p in pl], [p["test"][metric] for p in pl],
                     color=_COLOR.get(arch, "0.3"), marker=_MARK.get(arch, "o"),
                     ms=3, lw=1.2, label=f"{arch} ({d['row']['scale']})")
            drew = True
    axb.set_xlabel("relative depth"); axb.set_ylabel(f"test {metric}")
    axb.set_title(f"where structure lives{f'  ·  {primary}' if primary else ''}")
    axb.grid(True, color="0.93", lw=0.6)
    axb.text(-0.18, 1.04, tag[n_src], transform=axb.transAxes, fontsize=10, fontweight="bold")
    if drew:
        axb.legend(frameon=False)
    else:
        axb.text(0.5, 0.5, "layer-wise\n(needs sidecar JSONs)", transform=axb.transAxes,
                 ha="center", va="center", color="0.6", fontsize=7)

    for ax in flat[n_panels:]:                            # hide any spare grid cells
        ax.set_visible(False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.with_suffix('.png')}  ({len(rows)} rows, {len(sidecars)} sidecars, "
          f"{n_src} source(s))")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="run_probes results CSV.")
    ap.add_argument("--sidecar-dir", type=Path, default=None, help="Dir of per-layer JSON sidecars.")
    ap.add_argument("--out", type=Path, default=Path("figures/probes"))
    ap.add_argument("--metric", default="macro_f1", choices=["macro_f1", "accuracy"])
    args = ap.parse_args()

    rows = load_results(args.results) if args.results.exists() else []
    sidecars = load_sidecars(args.sidecar_dir) if args.sidecar_dir else []
    if not rows and not sidecars:
        print("No results or sidecars found.")
        return 0
    figure(rows, sidecars, args.out, metric=args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
