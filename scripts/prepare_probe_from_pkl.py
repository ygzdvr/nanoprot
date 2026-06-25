#!/usr/bin/env python
"""
Build a nanoprot probe cache from a thesis Swiss-Prot label pickle.

The thesis parsed Swiss-Prot per-residue annotations into pickles (a list of
``{accession, sequence, <label>: ndarray}``) for active_site / disorder / transmembrane /
signal_peptide (binary per-residue) and ss (3-class). This converts one such pickle into
the nanoprot probe-cache format (meta.json + data.jsonl + provenance.json) with
homology-aware (mmseqs2 cluster) splits, so the trajectory and cross-architecture probing
pipeline can consume it directly. That gives the developmental (training-time) emergence
of these richer biological concepts — functional sites, disorder, membrane topology — not
just local secondary structure.

Pfam is per-PROTEIN (sequence-level), not per-residue, so it needs a pooling probe and is
NOT handled here.

Usage:
  python -m scripts.prepare_probe_from_pkl \
      --pkl ../.cache/nanochat/probes/swissprot_active_site.pkl \
      --out .cache/nanoprot/probes/active_site_swissprot --max-proteins 8000
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from nanoprot.eval.probe.cluster import assign_splits_clustered  # noqa: E402
from nanoprot.eval.probe.labels import assign_splits  # noqa: E402
from scripts.prepare_probe_data import write_cache  # noqa: E402

_META_KEYS = {"accession", "sequence", "pfam_ids", "pfam_names", "name", "id"}


def label_key(rec: dict) -> str:
    """The per-residue label array key (the one that isn't accession/sequence/pfam_*)."""
    cands = [k for k in rec if k not in _META_KEYS and hasattr(rec[k], "__len__")]
    if not cands:
        raise SystemExit(f"no per-residue label array in record keys {list(rec)} "
                         "(per-protein labels like pfam are not supported here)")
    return cands[0]


def sample_records(records, key, max_proteins):
    """Up to ``max_proteins`` proteins, preferring those with >=1 positive residue so the
    probe has positives; pad with negatives. Deterministic (input order, no shuffle)."""
    pos, neg = [], []
    for r in records:
        if len(pos) >= max_proteins:
            break
        if bool(np.asarray(r[key]).any()):
            pos.append(r)
        elif len(neg) < max_proteins:
            neg.append(r)
    out = pos[:max_proteins]
    if len(out) < max_proteins:
        out += neg[:max_proteins - len(out)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, required=True, help="A thesis swissprot_*.pkl label file.")
    ap.add_argument("--source", default="swissprot")
    ap.add_argument("--concept", default=None, help="Concept name (default: the pickle's label key).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-proteins", type=int, default=8000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-seq-id", type=float, default=0.3)
    ap.add_argument("--no-cluster-split", dest="cluster_split", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.set_defaults(cluster_split=True)
    args = ap.parse_args()

    records = pickle.load(open(args.pkl, "rb"))
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{args.pkl} is not a non-empty list")
    key = label_key(records[0])
    concept = args.concept or key
    sampled = sample_records(records, key, args.max_proteins)

    ids = [r["accession"] for r in sampled]
    seqs = [r["sequence"] for r in sampled]
    n_classes = int(max(int(x) for r in sampled for x in r[key])) + 1
    class_names = ["neg", concept] if n_classes == 2 else [f"c{i}" for i in range(n_classes)]

    fracs = (1 - 2 * args.val_frac, args.val_frac, args.val_frac)
    splits = None
    if args.cluster_split:
        try:
            splits = assign_splits_clustered(ids, seqs, fracs=fracs, seed=args.seed,
                                             min_seq_id=args.min_seq_id)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}\n  -> per-protein hash split (NOT homology-safe).")
    if splits is None:
        splits = assign_splits(ids, fracs=fracs, seed=args.seed)

    proteins = [{"id": r["accession"], "sequence": r["sequence"],
                 "labels": [int(x) for x in r[key]], "split": splits[r["accession"]]}
                for r in sampled]

    counts: dict = {}
    n_lab = n_pos = 0
    for p in proteins:
        counts[p["split"]] = counts.get(p["split"], 0) + 1
        n_lab += len(p["labels"])
        n_pos += sum(1 for x in p["labels"] if x > 0)
    prov = {"source": f"Swiss-Prot {concept} (thesis pickle {args.pkl.name})",
            "concept": concept, "max_proteins": args.max_proteins, "seed": args.seed,
            "cluster_split": args.cluster_split,
            "min_seq_id": args.min_seq_id if args.cluster_split else None,
            "split_counts": counts, "n_labelled_residues": n_lab, "n_positive_residues": n_pos}
    write_cache(args.out, proteins, prov, source=args.source, concept=concept,
                task="classification", n_classes=n_classes, class_names=class_names)
    print(f"  wrote {len(proteins)} proteins ({concept}/{args.source}) to {args.out}  "
          f"splits={counts}  residues={n_lab:,}  positives={n_pos:,} "
          f"({100 * n_pos / max(n_lab, 1):.1f}%)  n_classes={n_classes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
