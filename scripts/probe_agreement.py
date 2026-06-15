#!/usr/bin/env python
"""
Label-level agreement between two SS3 label sources (the §6 "sanity check").

On proteins present in *both* cached datasets (matched by accession AND identical
sequence), compute per-residue SS3 agreement over positions both sources label. For
the Swiss-Prot vs DSSP-from-AlphaFold pair this is largely a measure of
annotation-vs-predicted-structure concordance — a pipeline sanity check, not "our
labels are clean" (docs/probing_harness_plan.md §6).

Usage:
  python -m scripts.probe_agreement --a $NANOPROT_BASE_DIR/probes/ss3_swissprot \
      --b $NANOPROT_BASE_DIR/probes/ss3_dssp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoprot.eval.probe.labels import ProbeDataset, load_probe_dataset  # noqa: E402

_SS3 = ["helix", "strand", "coil"]


def label_agreement(ds_a: ProbeDataset, ds_b: ProbeDataset) -> dict:
    """Per-residue SS3 agreement on shared, sequence-identical proteins.

    Returns n_proteins compared, n_residues compared (both labelled), overall
    agreement, per-class recall (diagonal/row), and the 3x3 confusion (rows = source A).
    """
    by_id = {p.id: p for p in ds_b.proteins}
    confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    n_proteins = n_res = n_agree = 0
    for pa in ds_a.proteins:
        pb = by_id.get(pa.id)
        if pb is None or pa.sequence != pb.sequence:
            continue
        n_proteins += 1
        for la, lb in zip(pa.labels, pb.labels):
            if la == ds_a.ignore_index or lb == ds_b.ignore_index:
                continue
            confusion[la][lb] += 1
            n_res += 1
            n_agree += (la == lb)
    per_class = {}
    for c in range(3):
        row = sum(confusion[c])
        per_class[_SS3[c]] = (confusion[c][c] / row) if row else float("nan")
    return {
        "n_proteins": n_proteins, "n_residues": n_res,
        "agreement": (n_agree / n_res) if n_res else float("nan"),
        "per_class_agreement": per_class, "confusion": confusion,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="First cached probe dataset dir.")
    ap.add_argument("--b", type=Path, required=True, help="Second cached probe dataset dir.")
    args = ap.parse_args()

    ds_a = load_probe_dataset(args.a)
    ds_b = load_probe_dataset(args.b)
    r = label_agreement(ds_a, ds_b)
    print(f"\n  Label agreement: {ds_a.source} vs {ds_b.source}")
    print(f"  shared sequence-identical proteins: {r['n_proteins']}  "
          f"| residues compared: {r['n_residues']:,}")
    print(f"  overall per-residue agreement: {r['agreement']:.3f}\n")
    head = "A row / B col"
    print(f"  {head:>20} {'helix':>8} {'strand':>8} {'coil':>8}   recall")
    for c in range(3):
        row = r["confusion"][c]
        rec = r["per_class_agreement"][_SS3[c]]
        print(f"  {_SS3[c]:>20} {row[0]:>8} {row[1]:>8} {row[2]:>8}   {rec:.3f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
