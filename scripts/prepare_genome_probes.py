#!/usr/bin/env python
"""
Build genomics probe datasets for B-GEN (PRD: brain/plans/B-GEN_genomics_PRD.md) — the genomics
analog of prepare_probe_data.py. Emits the exact format nanoprot.eval.probe.labels.load_probe_dataset
reads, so run_trajectory_probes.py consumes it UNCHANGED:
    meta.json   {"concept","source","task","n_classes","class_names","ignore_index","provenance"}
    data.jsonl  one JSON per window: {"id","sequence","labels","split"}   (len(labels)==len(sequence))
    provenance.json

Concepts:
  ``exon``   per-position binary intron(0)/exon(1) within gene bodies (phase-1 concept). WEAK capability
             for char-level S/M models (converged macro-F1 ~0.48, learned-vs-random Delta ~0.02) -> too
             little signal to expose the architecture rank-reversal.
  ``splice`` per-position 3-class neither(0)/donor(1)/acceptor(2). Exon-union boundaries: an exon->intron
             transition is a donor (intron 5', GT); an intron->exon transition is an acceptor (intron 3',
             AG). Each site is dilated +/- --splice-tol bp. The SpliceAI capability; sharp local motif in
             long-range context (attention-favorable late).
  ``gc``     per-position 3-class low/mid/high based on the local +/- --gc-win bp G+C fraction. A RUNNING
             COMPOSITION statistic -> the SSM's recurrence tracks it early -> a prime early-SSM-lead (the
             reversal mechanism the protein sweep showed: reversals fire where mamba leads early).
  ``frame``  per-position 3-class codon position {0,1,2} within CDS (IGNORE outside CDS), strand- and
             phase-aware from the GENCODE CDS frame field. PERIOD-3 STRUCTURE is exactly what a recurrent
             SSM captures early -> theoretically the strongest early-SSM-lead / reversal candidate.
             Caveat: overlapping isoforms can assign conflicting frames to a position (last-wins); this
             adds modest label noise at alternatively-spliced CDS.

**Locus-safety (the homology-safe analog):** train/val/test are assigned by **whole chromosome**
(default train chr1-18, val chr19-20, test chr21-22), so no locus leaks between splits — the genomic
counterpart of the protein homology-cluster split. An overlap check asserts the chromosome sets are
disjoint.

Inputs (download on a LOGIN node; compute nodes have no internet):
  genome FASTA  : UCSC hg38  (https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz)
  GENCODE GTF   : e.g. https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.annotation.gtf.gz

Usage:
  python -m scripts.prepare_genome_probes --fasta hg38.fa.gz --gtf gencode.v46.annotation.gtf.gz \\
      --out $NANOPROT_BASE_DIR/probes/genome_splice --concept splice --window-len 1024
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_genome import iter_fasta_records, _clean, _MAIN_CHROM_RE  # noqa: E402 (validated)

IGNORE_INDEX = -100
CONCEPT_SPEC = {  # concept -> (task, n_classes, class_names)
    "exon":   ("classification", 2, ["intron", "exon"]),
    "splice": ("classification", 3, ["neither", "donor", "acceptor"]),
    "gc":     ("classification", 3, ["low_gc", "mid_gc", "high_gc"]),
    "frame":  ("classification", 3, ["codon0", "codon1", "codon2"]),
}
# concepts whose labelling needs the CDS features (strand + phase); others only need exon/gene intervals
_NEEDS_CDS = {"frame"}


def _norm_chrom(name: str) -> str:
    """Normalise 'chr1'/'1' -> 'chr1' for matching FASTA headers to GTF seqnames."""
    n = name.strip()
    return n if n.lower().startswith("chr") else f"chr{n}"


def parse_gtf(gtf_path: Path, keep_chroms: set, *, need_cds: bool = False) -> Tuple[
        Dict[str, List[Tuple[int, int]]], Dict[str, List[Tuple[int, int]]],
        Dict[str, List[Tuple[int, int, str, int]]]]:
    """Return (genes, exons, cds): chrom -> intervals (0-based half-open).
    genes/exons: [start,end); cds: [start,end,strand,phase]. GENCODE GTF is 1-based inclusive."""
    genes: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    exons: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    cds: Dict[str, List[Tuple[int, int, str, int]]] = defaultdict(list)
    wanted = {"gene", "exon"} | ({"CDS"} if need_cds else set())
    op = gzip.open if str(gtf_path).endswith(".gz") else open
    with op(gtf_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            feat = f[2]
            if feat not in wanted:
                continue
            chrom = _norm_chrom(f[0])
            if chrom not in keep_chroms:
                continue
            try:
                s0 = int(f[3]) - 1   # 1-based inclusive -> 0-based
                e0 = int(f[4])       # inclusive end -> half-open end
            except ValueError:
                continue
            if e0 <= s0:
                continue
            if feat == "gene":
                genes[chrom].append((s0, e0))
            elif feat == "exon":
                exons[chrom].append((s0, e0))
            else:  # CDS
                strand = f[6] if f[6] in ("+", "-") else "+"
                try:
                    phase = int(f[7])
                except ValueError:
                    phase = 0
                cds[chrom].append((s0, e0, strand, phase))
    return genes, exons, cds


# ---------------------------------------------------------------------------
# Per-chromosome label arrays (built ONCE per chromosome: O(genome), then sliced per window).
# Unit-tested via test-friendly pure signatures.
# ---------------------------------------------------------------------------

def _exon_mask(n: int, exons_c: List[Tuple[int, int]]) -> np.ndarray:
    m = np.zeros(n, dtype=np.uint8)
    for es, ee in exons_c:
        if ee > 0 and es < n:
            m[max(0, es):min(n, ee)] = 1
    return m


def labels_exon(n: int, exons_c) -> np.ndarray:
    return _exon_mask(n, exons_c).astype(np.int16)


def labels_splice(n: int, exons_c, *, tol: int) -> np.ndarray:
    """3-class neither/donor/acceptor from exon-union transitions, dilated +/- tol bp.
    donor = exon(1)->intron(0) transition (intron 5'); acceptor = intron(0)->exon(1) (intron 3')."""
    mask = _exon_mask(n, exons_c)
    lab = np.zeros(n, dtype=np.int16)
    if n < 2:
        return lab
    donors = np.where((mask[:-1] == 1) & (mask[1:] == 0))[0] + 1     # intron start
    accept = np.where((mask[:-1] == 0) & (mask[1:] == 1))[0] + 1     # exon  start
    for i in donors:
        lab[max(0, i - tol):min(n, i + tol + 1)] = 1
    for i in accept:                                                # acceptor overwrites in overlap
        lab[max(0, i - tol):min(n, i + tol + 1)] = 2
    return lab


def labels_gc(seq: str, *, win: int, lo: float, hi: float) -> np.ndarray:
    """3-class low/mid/high local G+C fraction over a +/- win window (rolling mean via cumsum)."""
    b = np.frombuffer(seq.upper().encode("ascii", "replace"), dtype=np.uint8)
    n = b.shape[0]
    isgc = ((b == ord("G")) | (b == ord("C"))).astype(np.int32)
    csum = np.concatenate([[0], np.cumsum(isgc)])
    idx = np.arange(n)
    a = np.maximum(0, idx - win)
    z = np.minimum(n, idx + win + 1)
    frac = (csum[z] - csum[a]) / np.maximum(1, z - a)
    lab = np.full(n, 1, dtype=np.int16)  # mid
    lab[frac < lo] = 0                   # low
    lab[frac > hi] = 2                   # high
    return lab


def labels_frame(n: int, cds_c: List[Tuple[int, int, str, int]]) -> np.ndarray:
    """3-class codon position {0,1,2} within CDS (IGNORE elsewhere), strand- and phase-aware.
    + strand: codon_pos(g) = (g - s - phase) mod 3.  - strand: ((e-1) - g - phase) mod 3.
    (GTF phase = #bases to remove from the feature 5' to reach the first base of the next codon.)"""
    lab = np.full(n, IGNORE_INDEX, dtype=np.int16)
    for s, e, strand, phase in cds_c:
        s = max(0, s); e = min(n, e)
        if e <= s:
            continue
        rel = np.arange(e - s)
        if strand == "-":
            lab[s:e] = ((e - 1 - s) - rel - phase) % 3
        else:
            lab[s:e] = (rel - phase) % 3
    return lab


def build_chrom_labels(concept: str, seq: str, exons_c, cds_c, args) -> np.ndarray:
    n = len(seq)
    if concept == "exon":
        return labels_exon(n, exons_c)
    if concept == "splice":
        return labels_splice(n, exons_c, tol=args.splice_tol)
    if concept == "gc":
        return labels_gc(seq, win=args.gc_win, lo=args.gc_lo, hi=args.gc_hi)
    if concept == "frame":
        return labels_frame(n, cds_c)
    raise ValueError(concept)


def _window_ok(concept: str, lab_win: np.ndarray) -> bool:
    """Drop windows with no signal so class-0/IGNORE does not swamp the probe."""
    if concept == "splice":
        return bool((lab_win > 0).any())          # >=1 donor/acceptor
    if concept == "frame":
        return bool((lab_win != IGNORE_INDEX).any())  # >=1 CDS position
    return True                                    # exon/gc are dense


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


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if str(path).endswith(".gz") \
        else open(path, "rt", encoding="utf-8", errors="replace")


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
    ap.add_argument("--splice-tol", type=int, default=3, help="splice: +/- bp dilation of each site.")
    ap.add_argument("--gc-win", type=int, default=50, help="gc: +/- bp window for local G+C fraction.")
    ap.add_argument("--gc-lo", type=float, default=0.42, help="gc: low/mid boundary on G+C fraction.")
    ap.add_argument("--gc-hi", type=float, default=0.52, help="gc: mid/high boundary on G+C fraction.")
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
    tr, va, te = (_chroms(args.train_chroms), _chroms(args.val_chroms), _chroms(args.test_chroms))
    if (tr & va) or (tr & te) or (va & te):
        sys.exit(f"chromosome splits overlap (leakage!): tr∩va={tr&va} tr∩te={tr&te} va∩te={va&te}")

    win = args.window_len
    stride = args.stride or win
    concept = args.concept
    task, n_classes, class_names = CONCEPT_SPEC[concept]
    need_cds = concept in _NEEDS_CDS

    print(f"Parsing GTF {args.gtf} (chroms {sorted(keep)}, need_cds={need_cds}) ...")
    genes, exons, cds = parse_gtf(args.gtf, keep, need_cds=need_cds)
    for c in exons:
        exons[c].sort()
    print(f"  {sum(len(v) for v in genes.values()):,} genes, "
          f"{sum(len(v) for v in exons.values()):,} exons, "
          f"{sum(len(v) for v in cds.values()):,} CDS")

    records: List[dict] = []
    per_split = defaultdict(int)
    dropped_n = dropped_empty = 0
    class_hist: Dict[int, int] = defaultdict(int)
    for chrom, seq in iter_fasta_records(open_fasta := _open(args.fasta)):
        chrom = _norm_chrom(chrom)
        if chrom not in keep:
            continue
        sp = split_of[chrom]
        if args.max_windows_per_split and per_split[sp] >= args.max_windows_per_split:
            continue
        # Per-chromosome concept label array, built ONCE: O(genome). Sliced per window below.
        lab_arr = build_chrom_labels(concept, seq, exons.get(chrom, []), cds.get(chrom, []), args)
        for gs, ge in sorted(genes.get(chrom, [])):
            for w0 in range(gs, ge - win + 1, stride):
                dna = _clean(seq[w0:w0 + win])
                if dna.count("N") / win > args.max_n_frac:
                    dropped_n += 1
                    continue
                lab_win = lab_arr[w0:w0 + win]
                if not _window_ok(concept, lab_win):
                    dropped_empty += 1
                    continue
                for v in lab_win:
                    class_hist[int(v)] += 1
                records.append({"id": f"{chrom}:{w0}-{w0+win}", "sequence": dna,
                                "labels": lab_win.tolist(), "split": sp})
                per_split[sp] += 1
                if args.max_windows_per_split and per_split[sp] >= args.max_windows_per_split:
                    break
            if args.max_windows_per_split and per_split[sp] >= args.max_windows_per_split:
                break
    open_fasta.close()

    labelled = {k: v for k, v in sorted(class_hist.items()) if k != IGNORE_INDEX}
    tot = sum(labelled.values()) or 1
    provenance = {
        "dataset": "GENCODE+hg38", "concept": concept,
        "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "fasta": str(args.fasta), "gtf": str(args.gtf),
        "window_len": win, "stride": stride, "max_n_frac": args.max_n_frac,
        "splice_tol": args.splice_tol, "gc_win": args.gc_win, "gc_lo": args.gc_lo, "gc_hi": args.gc_hi,
        "split_chroms": {"train": args.train_chroms, "val": args.val_chroms, "test": args.test_chroms},
        "split_sizes": dict(per_split), "num_windows": len(records),
        "class_balance": {class_names[k]: round(v / tot, 4) for k, v in labelled.items()
                          if 0 <= k < len(class_names)},
        "num_dropped_highN": dropped_n, "num_dropped_no_signal": dropped_empty, "seed": args.seed,
    }
    write_cache(args.out, records, provenance, concept=concept,
                task=task, n_classes=n_classes, class_names=class_names)
    print("=" * 64)
    print(f"  Wrote {len(records):,} windows to {args.out}  (splits: {dict(per_split)})")
    print(f"  Concept={concept} n_classes={n_classes}  class_balance={provenance['class_balance']}")
    print(f"  Dropped {dropped_n:,} high-N, {dropped_empty:,} no-signal windows. Locus-safe (chrom-disjoint).")
    print("=" * 64)
    if not all(per_split.get(s, 0) > 0 for s in ("train", "val", "test")):
        sys.exit("ERROR: a split is empty — check --{train,val,test}-chroms vs the FASTA/GTF.")
    if len(labelled) < 2:
        sys.exit(f"ERROR: <2 populated classes ({labelled}) — loosen thresholds / check the concept.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
