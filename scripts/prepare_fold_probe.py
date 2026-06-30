#!/usr/bin/env python
"""Build a SEQUENCE-LEVEL fold (CATH topology) probe cache — A-BREADTH topology axis.

Joins two CATH downloads (staged under probes/raw/):
  - cath-domain-seqs-S35.fa     domain sequences (35% non-redundant), header `cath|ver|<domain_id>/range`
  - cath-domain-list.txt        <domain_id> C A T H S O L I D ... -> the C.A.T = TOPOLOGY (fold) label
Each S35 domain -> (sequence, topology "C.A.T"). Keep the top-K most common topologies, cap per class,
and split with the same stratified-within-class homology-safe scheme as the other breadth caches. Cache
is level="sequence" (the probe mean-pools residue reps). A 4th breadth axis: structural topology.

Usage:
  python -m scripts.prepare_fold_probe --fasta .cache/nanoprot/probes/raw/cath-domain-seqs-S35.fa \
      --domain-list .cache/nanoprot/probes/raw/cath-domain-list.txt \
      --out .cache/nanoprot/probes/fold_cath --top-k 30 --max-per-class 800 --max-len 512
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoprot.eval.probe.cluster import _hash_unit, cluster_sequences  # noqa: E402
from nanoprot.eval.probe.labels import assign_splits  # noqa: E402
from scripts.prepare_probe_data import write_cache  # noqa: E402

_AA = set("ACDEFGHIKLMNPQRSTVWYX")


def parse_domain_topology(path: Path):
    """domain_id -> 'C.A.T' (topology/fold)."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.split()
            if len(f) >= 4:
                out[f[0]] = f"{f[1]}.{f[2]}.{f[3]}"
    return out


def parse_cath_fasta(path: Path):
    """domain_id -> sequence (header 'cath|ver|<domain_id>/range')."""
    seqs, did, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if did and buf:
                    seqs[did] = "".join(buf)
                parts = line[1:].strip().split("|")
                did = parts[2].split("/")[0] if len(parts) >= 3 else None
                buf = []
            else:
                buf.append(line.strip())
        if did and buf:
            seqs[did] = "".join(buf)
    return seqs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("--domain-list", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--max-per-class", type=int, default=800)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-seq-id", type=float, default=0.3)
    ap.add_argument("--no-cluster-split", dest="cluster_split", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.set_defaults(cluster_split=True)
    args = ap.parse_args()

    topo = parse_domain_topology(args.domain_list)
    seqs = parse_cath_fasta(args.fasta)
    print(f"  {len(topo):,} domains in list, {len(seqs):,} sequences in FASTA")

    recs = []
    for did, seq in seqs.items():
        s = seq.upper()
        t = topo.get(did)
        if t and args.min_len <= len(s) <= args.max_len and set(s) <= _AA:
            recs.append({"accession": did, "sequence": s, "topology": t})
    print(f"  {len(recs):,} domains with sequence + topology, in length bounds")

    topk = [t for t, _ in Counter(r["topology"] for r in recs).most_common(args.top_k)]
    topk_set = set(topk)
    t_idx = {t: i for i, t in enumerate(topk)}
    per_class: dict = defaultdict(list)
    for r in recs:
        if r["topology"] in topk_set:
            per_class[r["topology"]].append(r)
    sampled = []
    for t in topk:
        sampled.extend(per_class[t][:args.max_per_class])
    lab1 = {r["accession"]: t_idx[r["topology"]] for r in sampled}
    ids = [r["accession"] for r in sampled]
    seqs_l = [r["sequence"] for r in sampled]

    # Stratified-within-class homology-safe split (same scheme as family/EC/subcellular).
    cls_of = {r["accession"]: lab1[r["accession"]] for r in sampled}
    m2r = None
    if args.cluster_split:
        try:
            m2r = cluster_sequences(ids, seqs_l, min_seq_id=args.min_seq_id)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}\n  -> per-protein hash split (NOT homology-safe).")
    splits: dict = {}
    keep: set = set()
    if m2r is not None:
        cls_reps: dict = defaultdict(set)
        for pid in ids:
            cls_reps[cls_of[pid]].add(m2r[pid])
        rep_split: dict = {}
        for cl, reps in cls_reps.items():
            reps = sorted(reps, key=lambda rr: _hash_unit(f"{args.seed}:{cl}:{rr}"))
            n = len(reps)
            n_test = max(1, round(args.val_frac * n))
            n_val = max(1, round(args.val_frac * n))
            n_train = n - n_test - n_val
            if n < 3 or n_train < 1:
                continue
            for i, rr in enumerate(reps):
                rep_split[(cl, rr)] = ("train" if i < n_train else "val" if i < n_train + n_val else "test")
            keep.add(cl)
        for pid in ids:
            s = rep_split.get((cls_of[pid], m2r[pid]))
            if s is not None:
                splits[pid] = s
    else:
        fracs = (1 - 2 * args.val_frac, args.val_frac, args.val_frac)
        splits = assign_splits(ids, fracs=fracs, seed=args.seed)
        by: dict = defaultdict(Counter)
        for pid in ids:
            by[cls_of[pid]][splits[pid]] += 1
        keep = {c for c in by if by[c]["train"] and by[c]["test"]}

    kept_sorted = sorted(keep)
    reidx = {old: new for new, old in enumerate(kept_sorted)}
    class_names = [topk[ci] for ci in kept_sorted]
    n_classes = len(kept_sorted)
    if n_classes < 2:
        raise SystemExit(f"only {n_classes} topologies survived — loosen --min-seq-id / --max-per-class")

    proteins = [{"id": r["accession"], "sequence": r["sequence"],
                 "labels": [reidx[lab1[r["accession"]]]], "split": splits[r["accession"]]}
                for r in sampled if lab1[r["accession"]] in keep and r["accession"] in splits]
    counts = Counter(p["split"] for p in proteins)
    prov = {"source": "CATH S35 domain topology (C.A.T)", "concept": "fold",
            "top_k": args.top_k, "max_per_class": args.max_per_class, "max_len": args.max_len,
            "cluster_split": args.cluster_split, "min_seq_id": args.min_seq_id if args.cluster_split else None,
            "split_counts": dict(counts), "seed": args.seed, "class_names": class_names}
    write_cache(args.out, proteins, prov, source="cath", concept="fold",
                task="classification", n_classes=n_classes, class_names=class_names, level="sequence")
    print(f"  wrote {len(proteins)} domains, {n_classes} topologies, splits={dict(counts)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
