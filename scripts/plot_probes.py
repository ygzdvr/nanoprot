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
_LINESTYLE = ["-", "--", ":", "-."]            # one per source (triangulation axis)
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


def figure(rows: List[dict], sidecars: List[dict], out: Path, metric: str = "macro_f1") -> None:
    _style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)

    # (a) scaling transfer: learned - baseline vs params, color=arch, linestyle=source
    agg = aggregate_scaling(rows)
    sources = sorted({s for (s, _a) in agg})
    style_of = {s: _LINESTYLE[i % len(_LINESTYLE)] for i, s in enumerate(sources)}
    for (source, arch), pts in sorted(agg.items()):
        xs = [p[0] for p in pts if p[0] > 0]
        ys = [p[2] for p in pts if p[0] > 0]
        es = [p[3] for p in pts if p[0] > 0]
        if not xs:
            continue
        axa.errorbar(xs, ys, yerr=es, color=_COLOR.get(arch, "0.3"), ls=style_of[source],
                     marker=_MARK.get(arch, "o"), ms=4, lw=1.2, capsize=2,
                     label=f"{arch}" + (f" · {source}" if len(sources) > 1 else ""))
    axa.axhline(0, color="0.6", lw=0.8, ls=":")
    axa.set_xscale("log"); axa.set_xlabel("parameters")
    axa.set_ylabel(f"probe score (learned − baseline, {metric})")
    axa.set_title("scaling transfer"); axa.grid(True, color="0.93", lw=0.6)
    axa.text(-0.2, 1.03, "a", transform=axa.transAxes, fontsize=10, fontweight="bold")
    if agg:
        axa.legend(frameon=False)
    else:
        axa.text(0.5, 0.5, "(no results yet)", transform=axa.transAxes,
                 ha="center", va="center", color="0.6")

    # (b) layer-wise test metric vs relative depth (largest scale, primary source)
    primary = sources[0] if sources else (rows[0]["source"] if rows else None)
    drew = False
    if primary:
        for arch, d in sorted(_largest_scale_sidecars(sidecars, primary).items()):
            pl = d["learned_per_layer"]
            xs = [p["rel_depth"] for p in pl]
            ys = [p["test"][metric] for p in pl]
            axb.plot(xs, ys, color=_COLOR.get(arch, "0.3"), marker=_MARK.get(arch, "o"),
                     ms=3, lw=1.2, label=f"{arch} ({d['row']['scale']})")
            drew = True
    axb.set_xlabel("relative depth"); axb.set_ylabel(f"test {metric}")
    axb.set_title(f"where structure lives{f'  ·  {primary}' if primary else ''}")
    axb.grid(True, color="0.93", lw=0.6)
    axb.text(-0.2, 1.03, "b", transform=axb.transAxes, fontsize=10, fontweight="bold")
    if drew:
        axb.legend(frameon=False)
    else:
        axb.text(0.5, 0.5, "layer-wise\n(needs sidecar JSONs)", transform=axb.transAxes,
                 ha="center", va="center", color="0.6", fontsize=7)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.with_suffix('.png')}  ({len(rows)} rows, {len(sidecars)} sidecars)")


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
