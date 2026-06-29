#!/usr/bin/env python
"""Build a SEQUENCE-LEVEL subcellular-localization probe cache — A-BREADTH localization axis.

Streams the staged Swiss-Prot XML and extracts, per entry, (accession, sequence, primary subcellular
location) from the `<comment type="subcellular location"><subcellularLocation><location>` text. Keeps
the top-K most common major locations (Cytoplasm, Cell membrane, Secreted, Nucleus, Mitochondrion, ...),
one label per protein (the primary location), caps per class, and splits with the SAME stratified-within-
class homology-safe scheme as the family/EC caches. Cache is level="sequence" (the probe mean-pools reps).

A third breadth axis alongside family (homology) and EC (function): localization. Self-contained (no download).

Usage:
  python -m scripts.prepare_subcellular_probe --xml .cache/nanoprot/probes/raw/uniprot_sprot.xml.gz \
      --out .cache/nanoprot/probes/subcellular_swissprot --top-k 10 --max-per-class 1500 --max-len 512
"""
from __future__ import annotations

import argparse
import gzip
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoprot.eval.probe.cluster import _hash_unit, cluster_sequences  # noqa: E402
from nanoprot.eval.probe.labels import assign_splits  # noqa: E402
from scripts.prepare_probe_data import write_cache  # noqa: E402

_AA = set("ACDEFGHIKLMNPQRSTVWYX")


def parse_swissprot_location(xml_gz: Path, max_entries: int = 0):
    """Stream entries; yield {accession, sequence, location (the primary subcellular location string)}."""
    out = []
    with gzip.open(xml_gz, "rb") as fh:
        acc = None
        seq = None
        locs: list = []
        for _, elem in ET.iterparse(fh, events=("end",)):
            tag = elem.tag.rsplit("}", 1)[-1]
            if tag == "accession" and acc is None:
                acc = (elem.text or "").strip()
            elif tag == "sequence" and elem.text:
                s = "".join(elem.text.split()).upper()
                if len(s) > 10 and set(s) <= _AA:
                    seq = s
            elif tag == "subcellularLocation":
                for child in elem:
                    if child.tag.rsplit("}", 1)[-1] == "location" and child.text:
                        locs.append(child.text.strip())
            elif tag == "entry":
                if acc and seq and locs:
                    out.append({"accession": acc, "sequence": seq, "location": locs[0]})
                acc, seq, locs = None, None, []
                elem.clear()
                if max_entries and len(out) >= max_entries:
                    break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-per-class", type=int, default=1500)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-seq-id", type=float, default=0.3)
    ap.add_argument("--no-cluster-split", dest="cluster_split", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-entries", type=int, default=0)
    ap.set_defaults(cluster_split=True)
    args = ap.parse_args()

    print(f"  streaming {args.xml} ...")
    recs = parse_swissprot_location(args.xml, max_entries=args.max_entries)
    print(f"  {len(recs):,} entries with sequence + subcellular location")

    loc_count: Counter = Counter(r["location"] for r in recs)
    topk = [loc for loc, _ in loc_count.most_common(args.top_k)]
    topk_set = set(topk)
    loc_idx = {loc: i for i, loc in enumerate(topk)}
    print(f"  top-{args.top_k} locations: {topk}")

    per_class: dict = defaultdict(list)
    for r in recs:
        if r["location"] not in topk_set:
            continue
        if not (args.min_len <= len(r["sequence"]) <= args.max_len):
            continue
        per_class[r["location"]].append(r)
    sampled = []
    for loc in topk:
        sampled.extend(per_class[loc][:args.max_per_class])
    lab1 = {r["accession"]: loc_idx[r["location"]] for r in sampled}
    ids = [r["accession"] for r in sampled]
    seqs = [r["sequence"] for r in sampled]

    # Stratified-within-class homology-safe split (same scheme as family/EC).
    cls_of = {r["accession"]: lab1[r["accession"]] for r in sampled}
    m2r = None
    if args.cluster_split:
        try:
            m2r = cluster_sequences(ids, seqs, min_seq_id=args.min_seq_id)
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
        raise SystemExit(f"only {n_classes} locations survived — loosen --min-seq-id / --max-per-class")

    proteins = [{"id": r["accession"], "sequence": r["sequence"],
                 "labels": [reidx[lab1[r["accession"]]]], "split": splits[r["accession"]]}
                for r in sampled if lab1[r["accession"]] in keep and r["accession"] in splits]
    counts = Counter(p["split"] for p in proteins)
    prov = {"source": "Swiss-Prot subcellular location (primary)", "concept": "subcellular",
            "top_k": args.top_k, "max_per_class": args.max_per_class, "max_len": args.max_len,
            "cluster_split": args.cluster_split, "min_seq_id": args.min_seq_id if args.cluster_split else None,
            "split_counts": dict(counts), "seed": args.seed, "class_names": class_names}
    write_cache(args.out, proteins, prov, source="swissprot", concept="subcellular",
                task="classification", n_classes=n_classes, class_names=class_names, level="sequence")
    print(f"  wrote {len(proteins)} proteins, {n_classes} locations {class_names}, splits={dict(counts)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
