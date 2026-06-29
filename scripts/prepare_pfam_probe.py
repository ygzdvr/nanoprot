#!/usr/bin/env python
"""Build a SEQUENCE-LEVEL Pfam family probe cache (A-BREADTH, the Nature breadth lever).

``swissprot_pfam.pkl`` gives ``{accession, sequence, pfam_ids, pfam_names}`` per protein. We take the
top-K most common families, keep proteins with EXACTLY ONE top-K family (clean single-label multi-class),
cap per family for balance, and split homology-safely (mmseqs cluster, whole clusters to one split).

Split subtlety (the no-mistakes part): a *global* homology split would risk isolating a family into one
split (test classes unseen in train); a *random* split would leak near-identical homologs (inflated).
Large Pfam families are internally diverse, so a global cluster split distributes each family's
sub-clusters across splits — remote-homolog generalization. We then VERIFY every kept family appears in
train AND test, dropping (with a warning) any family that does not, and re-index the survivors.

The cache is ``level="sequence"``: one label per protein; the probe mean-pools residue reps (run.py).

Usage:
  python -m scripts.prepare_pfam_probe --pkl ../.cache/nanochat/probes/swissprot_pfam.pkl \
      --out .cache/nanoprot/probes/pfam_swissprot --top-k 20 --max-per-family 800 --max-len 512
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoprot.eval.probe.cluster import _hash_unit, cluster_sequences  # noqa: E402
from nanoprot.eval.probe.labels import assign_splits  # noqa: E402
from scripts.prepare_probe_data import write_cache  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, required=True, help="swissprot_pfam.pkl (accession/sequence/pfam_ids)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-per-family", type=int, default=800)
    ap.add_argument("--min-kept", type=int, default=100,
                    help="drop families with fewer than this many clean single-label proteins")
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=512, help="drop proteins longer than the model context")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-seq-id", type=float, default=0.3)
    ap.add_argument("--no-cluster-split", dest="cluster_split", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.set_defaults(cluster_split=True)
    args = ap.parse_args()

    recs = pickle.load(open(args.pkl, "rb"))
    if not isinstance(recs, list) or not recs:
        raise SystemExit(f"{args.pkl} is not a non-empty list")

    fam_count: Counter = Counter()
    for r in recs:
        for f in (r.get("pfam_ids") or []):
            fam_count[f] += 1
    topk = [f for f, _ in fam_count.most_common(args.top_k)]
    topk_set = set(topk)
    fam_idx = {f: i for i, f in enumerate(topk)}
    print(f"  top-{args.top_k} families: {topk}")

    # keep proteins with EXACTLY ONE top-K family (clean single-label), within length bounds
    per_fam: dict = defaultdict(list)
    for r in recs:
        seq = r.get("sequence") or ""
        if not (args.min_len <= len(seq) <= args.max_len):
            continue
        hits = {f for f in (r.get("pfam_ids") or []) if f in topk_set}
        if len(hits) != 1:
            continue
        per_fam[next(iter(hits))].append(r)

    # keep only families with enough clean single-label examples (avoid tiny, unlearnable classes)
    topk = [f for f in topk if len(per_fam[f]) >= args.min_kept]
    fam_idx = {f: i for i, f in enumerate(topk)}
    if len(topk) < 2:
        raise SystemExit(f"only {len(topk)} families with >= {args.min_kept} clean examples")
    print(f"  {len(topk)} families with >= {args.min_kept} single-label proteins (of top-{args.top_k})")

    sampled = []
    for f in topk:
        sampled.extend(per_fam[f][:args.max_per_family])
    if not sampled:
        raise SystemExit("no proteins after filtering")

    ids = [r["accession"] for r in sampled]
    seqs = [r["sequence"] for r in sampled]
    lab1 = {r["accession"]: fam_idx[next(iter({f for f in r["pfam_ids"] if f in topk_set}))]
            for r in sampled}

    # Stratified-WITHIN-family homology-safe split: cluster globally (homologs share a cluster), then
    # assign EACH family's clusters ~(1-2v)/v/v across train/val/test, so every kept family spans all
    # three splits. (A plain global cluster split dumps a family's clusters into one split -> per-class
    # imbalance, e.g. ~1 test example.) Whole clusters stay in one split, so no homolog straddles the
    # boundary. Families with <3 clusters cannot be 3-way split and are dropped.
    fam_of = {r["accession"]: lab1[r["accession"]] for r in sampled}
    m2r = None
    if args.cluster_split:
        try:
            m2r = cluster_sequences(ids, seqs, min_seq_id=args.min_seq_id)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}\n  -> per-protein hash split (NOT homology-safe).")
    splits: dict = {}
    keep: set = set()
    if m2r is not None:
        fam_reps: dict = defaultdict(set)
        for pid in ids:
            fam_reps[fam_of[pid]].add(m2r[pid])
        rep_split: dict = {}
        for fam, reps in fam_reps.items():
            reps = sorted(reps, key=lambda rr: _hash_unit(f"{args.seed}:{fam}:{rr}"))
            n = len(reps)
            n_test = max(1, round(args.val_frac * n))
            n_val = max(1, round(args.val_frac * n))
            n_train = n - n_test - n_val
            if n < 3 or n_train < 1:
                continue                          # too few clusters to 3-way split this family
            for i, rr in enumerate(reps):
                rep_split[(fam, rr)] = ("train" if i < n_train
                                        else "val" if i < n_train + n_val else "test")
            keep.add(fam)
        for pid in ids:
            s = rep_split.get((fam_of[pid], m2r[pid]))
            if s is not None:
                splits[pid] = s
    else:
        fracs = (1 - 2 * args.val_frac, args.val_frac, args.val_frac)
        splits = assign_splits(ids, fracs=fracs, seed=args.seed)
        by: dict = defaultdict(Counter)
        for pid in ids:
            by[fam_of[pid]][splits[pid]] += 1
        keep = {f for f in by if by[f]["train"] and by[f]["test"]}

    dropped = [topk[fi] for fi in range(len(topk)) if fi not in keep]
    if dropped:
        print(f"  DROPPED {len(dropped)} families (<3 clusters / not in all splits): {dropped}")
    kept_sorted = sorted(keep)
    reidx = {old: new for new, old in enumerate(kept_sorted)}
    class_names = [topk[fi] for fi in kept_sorted]
    n_classes = len(kept_sorted)
    if n_classes < 2:
        raise SystemExit(f"only {n_classes} families survived — loosen --min-seq-id / --min-kept")

    proteins = [{"id": r["accession"], "sequence": r["sequence"],
                 "labels": [reidx[lab1[r["accession"]]]], "split": splits[r["accession"]]}
                for r in sampled if lab1[r["accession"]] in keep and r["accession"] in splits]

    counts = Counter(p["split"] for p in proteins)
    prov = {"source": f"Swiss-Prot Pfam ({args.pkl.name})", "concept": "pfam_family",
            "top_k": args.top_k, "kept_families": n_classes, "max_per_family": args.max_per_family,
            "max_len": args.max_len, "cluster_split": args.cluster_split,
            "min_seq_id": args.min_seq_id if args.cluster_split else None,
            "split_counts": dict(counts), "seed": args.seed}
    write_cache(args.out, proteins, prov, source="swissprot", concept="pfam_family",
                task="classification", n_classes=n_classes, class_names=class_names, level="sequence")
    print(f"  wrote {len(proteins)} proteins, {n_classes} families, splits={dict(counts)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
