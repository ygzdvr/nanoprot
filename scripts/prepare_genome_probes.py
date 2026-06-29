#!/usr/bin/env python
"""
Build genomics probe datasets for B-GEN (PRD: brain/plans/B-GEN_genomics_PRD.md) — the genomics
analog of prepare_probe_data.py. Emits the exact format nanoprot.eval.probe.labels.load_probe_dataset
reads, so run_trajectory_probes.py consumes it UNCHANGED:
    meta.json   {"concept","source","task","n_classes","class_names","ignore_index","provenance"}
    data.jsonl  one JSON per window: {"id","sequence","labels","split"}   (len(labels)==len(sequence))
    provenance.json

Concept (phase 1): ``exon`` — per-position binary intron(0)/exon(1) within gene bodies, from a GENCODE
GTF. This is the *locally-determined* concept that should expose the early-SSM-lead (the reversal
mechanism). Sequence-level promoter/enhancer + 3-class splice donor/acceptor are follow-ups.

**Locus-safety (the homology-safe analog):** train/val/test are assigned by **whole chromosome**
(default train chr1–18, val chr19–20, test chr21–22), so no locus leaks between splits — the genomic
counterpart of the protein homology-cluster split. An overlap check asserts the chromosome sets are
disjoint.

Inputs (download on a LOGIN node; compute nodes have no internet):
  genome FASTA  : UCSC hg38  (https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz)
  GENCODE GTF   : e.g. https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.annotation.gtf.gz

Usage:
  python -m scripts.prepare_genome_probes --fasta hg38.fa.gz --gtf gencode.v46.annotation.gtf.gz \\
      --out $NANOPROT_BASE_DIR/probes/genome_exon --concept exon --window-len 1024
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_genome import iter_fasta_records, _clean, _MAIN_CHROM_RE  # noqa: E402 (validated)

IGNORE_INDEX = -100
CONCEPT_SPEC = {  # concept -> (task, n_classes, class_names)
    "exon": ("classification", 2, ["intron", "exon"]),
}


def _norm_chrom(name: str) -> str:
    """Normalise 'chr1'/'1' -> 'chr1' for matching FASTA headers to GTF seqnames."""
    n = name.strip()
    return n if n.lower().startswith("chr") else f"chr{n}"


def parse_gtf(gtf_path: Path, keep_chroms: set) -> Tuple[Dict[str, List[Tuple[int, int]]],
                                                         Dict[str, List[Tuple[int, int]]]]:
    """Return (genes, exons): chrom -> list of 0-based half-open [start,end) intervals.
    GENCODE GTF is 1-based inclusive; we convert to 0-based half-open."""
    genes: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    exons: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    op = gzip.open if str(gtf_path).endswith(".gz") else open
    with op(gtf_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            chrom = _norm_chrom(f[0])
            if chrom not in keep_chroms:
                continue
            feat = f[2]
            if feat not in ("gene", "exon"):
                continue
            try:
                s0 = int(f[3]) - 1   # 1-based inclusive -> 0-based
                e0 = int(f[4])       # inclusive end -> half-open end
            except ValueError:
                continue
            if e0 <= s0:
                continue
            (genes if feat == "gene" else exons)[chrom].append((s0, e0))
    return genes, exons


def _exon_labels(window_lo: int, win: int, exon_iv: List[Tuple[int, int]]) -> List[int]:
    """Per-position intron(0)/exon(1) for window [window_lo, window_lo+win), given sorted exon
    intervals overlapping it. (Caller passes only intervals overlapping the window.)"""
    labels = [0] * win
    for es, ee in exon_iv:
        lo = max(window_lo, es) - window_lo
        hi = min(window_lo + win, ee) - window_lo
        for j in range(max(0, lo), min(win, hi)):
            labels[j] = 1
    return labels


def write_cache(out: Path, records: List[dict], provenance: dict, *, concept: str,
                task: str, n_classes: int, class_names: list) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({
        "concept": concept, "source": "gencode", "task": task, "n_classes": n_classes,
        "class_names": class_names, "ignore_index": IGNORE_INDEX, "provenance": provenance,
    }, indent=2))
    with (out / "data.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", type=Path, required=True, help="Genome FASTA(.gz).")
    ap.add_argument("--gtf", type=Path, required=True, help="GENCODE annotation GTF(.gz).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concept", default="exon", choices=list(CONCEPT_SPEC))
    ap.add_argument("--window-len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=0, help="0 -> non-overlapping (=window-len).")
    ap.add_argument("--max-n-frac", type=float, default=0.10)
    ap.add_argument("--max-windows-per-split", type=int, default=0, help="If >0, cap each split.")
    ap.add_argument("--train-chroms", default="1-18", help="e.g. '1-18' or '1,2,3'.")
    ap.add_argument("--val-chroms", default="19,20")
    ap.add_argument("--test-chroms", default="21,22")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    def _chroms(spec: str) -> set:
        out = set()
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-"); out |= {str(i) for i in range(int(a), int(b) + 1)}
            elif part:
                out.add(part)
        return {_norm_chrom(c) for c in out}

    split_of: Dict[str, str] = {}
    for nm, sp in (("train", args.train_chroms), ("val", args.val_chroms), ("test", args.test_chroms)):
        for c in _chroms(sp):
            split_of[c] = nm
    keep = set(split_of)
    # locus-safety assertion: the three chromosome sets must be disjoint
    tr, va, te = (_chroms(args.train_chroms), _chroms(args.val_chroms), _chroms(args.test_chroms))
    if (tr & va) or (tr & te) or (va & te):
        sys.exit(f"chromosome splits overlap (leakage!): tr∩va={tr&va} tr∩te={tr&te} va∩te={va&te}")

    win = args.window_len
    stride = args.stride or win
    task, n_classes, class_names = CONCEPT_SPEC[args.concept]

    print(f"Parsing GTF {args.gtf} (chroms {sorted(keep)}) ...")
    genes, exons = parse_gtf(args.gtf, keep)
    for c in exons:
        exons[c].sort()
    n_genes = sum(len(v) for v in genes.values())
    print(f"  {n_genes:,} gene intervals, {sum(len(v) for v in exons.values()):,} exon intervals")

    records: List[dict] = []
    per_split = defaultdict(int)
    dropped_n = 0
    for chrom, seq in iter_fasta_records(open_fasta := _open(args.fasta)):
        chrom = _norm_chrom(chrom)
        if chrom not in keep:
            continue
        sp = split_of[chrom]
        if args.max_windows_per_split and per_split[sp] >= args.max_windows_per_split:
            continue
        ex = exons.get(chrom, [])
        # Per-chromosome exon bitmask, built ONCE: O(genome). Replaces an O(windows x n_exons)
        # per-window rescan that is pathological on large chromosomes (chr1 ~200k exons -> hours).
        ex_mask = np.zeros(len(seq), dtype=np.uint8)
        for es, ee in ex:
            if ee > 0 and es < len(seq):
                ex_mask[max(0, es):min(len(seq), ee)] = 1
        for gs, ge in sorted(genes.get(chrom, [])):
            for w0 in range(gs, ge - win + 1, stride):
                dna = _clean(seq[w0:w0 + win])
                if dna.count("N") / win > args.max_n_frac:
                    dropped_n += 1
                    continue
                labels = ex_mask[w0:w0 + win].tolist()
                records.append({"id": f"{chrom}:{w0}-{w0+win}", "sequence": dna,
                                "labels": labels, "split": sp})
                per_split[sp] += 1
                if args.max_windows_per_split and per_split[sp] >= args.max_windows_per_split:
                    break
            if args.max_windows_per_split and per_split[sp] >= args.max_windows_per_split:
                break
    open_fasta.close()

    provenance = {
        "dataset": "GENCODE+hg38", "concept": args.concept,
        "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "fasta": str(args.fasta), "gtf": str(args.gtf),
        "window_len": win, "stride": stride, "max_n_frac": args.max_n_frac,
        "split_chroms": {"train": args.train_chroms, "val": args.val_chroms, "test": args.test_chroms},
        "split_sizes": dict(per_split), "num_windows": len(records),
        "num_dropped_highN": dropped_n, "seed": args.seed,
    }
    write_cache(args.out, records, provenance, concept=args.concept,
                task=task, n_classes=n_classes, class_names=class_names)
    print("=" * 60)
    print(f"  Wrote {len(records):,} windows to {args.out}  (split sizes: {dict(per_split)})")
    print(f"  Dropped {dropped_n:,} high-N windows. Concept={args.concept} (n_classes={n_classes}).")
    print(f"  Locus-safe: train/val/test are chromosome-disjoint.")
    print("=" * 60)
    if not all(per_split.get(s, 0) > 0 for s in ("train", "val", "test")):
        sys.exit("ERROR: a split is empty — check --{train,val,test}-chroms vs the FASTA/GTF.")
    return 0


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if str(path).endswith(".gz") \
        else open(path, "rt", encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
