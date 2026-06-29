#!/usr/bin/env python
"""Build a SEQUENCE-LEVEL EC (enzyme function) probe cache — A-BREADTH function axis.

Streams the staged Swiss-Prot XML (probes/raw/uniprot_sprot.xml.gz) and extracts, per entry, the
accession, amino-acid sequence, and EC number(s). The label is the **top-level EC class** (the first
digit: 1=oxidoreductase, 2=transferase, 3=hydrolase, 4=lyase, 5=isomerase, 6=ligase, 7=translocase).
We keep proteins with EXACTLY ONE top-level EC class (clean single-label), cap per class, and split
homology-safely with the SAME stratified-within-class scheme as the family probe (each class's mmseqs
clusters spread ~(1-2v)/v/v across train/val/test, whole clusters to one split → no homolog leakage,
every class in all splits). Cache is level="sequence"; the probe mean-pools residue reps.

Usage:
  python -m scripts.prepare_ec_probe --xml .cache/nanoprot/probes/raw/uniprot_sprot.xml.gz \
      --out .cache/nanoprot/probes/ec_swissprot --max-per-class 1500 --max-len 512
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

EC_NAMES = {1: "oxidoreductase", 2: "transferase", 3: "hydrolase", 4: "lyase",
            5: "isomerase", 6: "ligase", 7: "translocase"}
_AA = set("ACDEFGHIKLMNPQRSTVWYX")


def parse_swissprot_ec(xml_gz: Path, max_entries: int = 0):
    """Stream entries; yield {accession, sequence, ec_classes(set of top-level ints)}."""
    out = []
    with gzip.open(xml_gz, "rb") as fh:
        acc = None
        seq = None
        ecs: set = set()
        for _, elem in ET.iterparse(fh, events=("end",)):
            tag = elem.tag.rsplit("}", 1)[-1]
            if tag == "accession" and acc is None:
                acc = (elem.text or "").strip()
            elif tag == "ecNumber" and elem.text:
                d = elem.text.strip()[:1]
                if d.isdigit():
                    ecs.add(int(d))
            elif tag == "sequence" and elem.text:
                s = "".join(elem.text.split()).upper()
                if len(s) > 10 and set(s) <= _AA:
                    seq = s
            elif tag == "entry":
                if acc and seq and ecs:
                    out.append({"accession": acc, "sequence": seq, "ec_classes": set(ecs)})
                acc, seq, ecs = None, None, set()
                elem.clear()
                if max_entries and len(out) >= max_entries:
                    break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-per-class", type=int, default=1500)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-seq-id", type=float, default=0.3)
    ap.add_argument("--no-cluster-split", dest="cluster_split", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-entries", type=int, default=0, help="parse cap (0=all); for quick tests")
    ap.set_defaults(cluster_split=True)
    args = ap.parse_args()

    print(f"  streaming {args.xml} ...")
    recs = parse_swissprot_ec(args.xml, max_entries=args.max_entries)
    print(f"  {len(recs):,} entries with sequence + EC")

    # EXACTLY ONE top-level EC class (clean single-label), within length bounds
    per_class: dict = defaultdict(list)
    for r in recs:
        if len(r["ec_classes"]) != 1:
            continue
        if not (args.min_len <= len(r["sequence"]) <= args.max_len):
            continue
        per_class[next(iter(r["ec_classes"]))].append(r)
    classes = sorted(per_class)
    print("  per-class single-label counts: " + ", ".join(f"{EC_NAMES[c]}={len(per_class[c])}" for c in classes))

    sampled = []
    for c in classes:
        sampled.extend(per_class[c][:args.max_per_class])
    cls_idx = {c: i for i, c in enumerate(classes)}
    lab1 = {r["accession"]: cls_idx[next(iter(r["ec_classes"]))] for r in sampled}
    ids = [r["accession"] for r in sampled]
    seqs = [r["sequence"] for r in sampled]

    # Stratified-within-class homology-safe split (same scheme as the family cache).
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
        cls_reps: dict = defaultdict(set)
        for pid in ids:
            cls_reps[fam_of[pid]].add(m2r[pid])
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
            s = rep_split.get((fam_of[pid], m2r[pid]))
            if s is not None:
                splits[pid] = s
    else:
        fracs = (1 - 2 * args.val_frac, args.val_frac, args.val_frac)
        splits = assign_splits(ids, fracs=fracs, seed=args.seed)
        by: dict = defaultdict(Counter)
        for pid in ids:
            by[fam_of[pid]][splits[pid]] += 1
        keep = {c for c in by if by[c]["train"] and by[c]["test"]}

    kept_sorted = sorted(keep)
    reidx = {old: new for new, old in enumerate(kept_sorted)}
    class_names = [EC_NAMES[classes[ci]] for ci in kept_sorted]
    n_classes = len(kept_sorted)
    if n_classes < 2:
        raise SystemExit(f"only {n_classes} EC classes survived — loosen --min-seq-id / --max-per-class")

    proteins = [{"id": r["accession"], "sequence": r["sequence"],
                 "labels": [reidx[lab1[r["accession"]]]], "split": splits[r["accession"]]}
                for r in sampled if lab1[r["accession"]] in keep and r["accession"] in splits]
    counts = Counter(p["split"] for p in proteins)
    prov = {"source": "Swiss-Prot EC (top-level enzyme class)", "concept": "ec_class",
            "max_per_class": args.max_per_class, "max_len": args.max_len,
            "cluster_split": args.cluster_split, "min_seq_id": args.min_seq_id if args.cluster_split else None,
            "split_counts": dict(counts), "seed": args.seed, "class_names": class_names}
    write_cache(args.out, proteins, prov, source="swissprot", concept="ec_class",
                task="classification", n_classes=n_classes, class_names=class_names, level="sequence")
    print(f"  wrote {len(proteins)} proteins, {n_classes} EC classes {class_names}, splits={dict(counts)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
