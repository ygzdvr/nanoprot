#!/usr/bin/env python
"""
Prepare the GRCh38 (hg38) pretraining corpus as shuffled parquet shards — the genomics analog of
``prepare_uniref50.py`` for the B-GEN cross-domain experiment (PRD:
``brain/plans/B-GEN_genomics_PRD.md``).

nanoprot's loader reads ``shard_*.parquet`` (a ``text`` column; the last shard is the validation
split). Each row here is a fixed-length DNA window (single-nucleotide / char-level), windowed from the
main chromosomes. The sharding / shuffle / download machinery is *imported* from prepare_uniref50 so
this script only adds the genome-specific windowing (one tested code path, no duplication).

Pipeline (mirrors prepare_uniref50):
  1. (login node, internet) download a genome FASTA, e.g. UCSC hg38:
       wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz \\
            -O $NANOPROT_BASE_DIR/hg38.fa.gz
     (or pass --download-url to let this script fetch it).
  2. shard it (no internet; CPU only):
       python -m scripts.prepare_genome \\
           --fasta $NANOPROT_BASE_DIR/hg38.fa.gz --out $NANOPROT_BASE_DIR/hg38_parquet

Each main chromosome (chr1..22, X, Y; chrM / _alt / _random / chrUn skipped) is cut into
non-overlapping --window-len windows (--stride to overlap); windows with more than --max-n-frac
ambiguous (N) content are dropped; characters are upper-cased and any non-ACGT mapped to N
(char-level alphabet {A,C,G,T,N}). --reverse-complement additionally emits each window's reverse
complement (a standard DNA augmentation; off by default for the controlled baseline).

Note: locus-safety is enforced at the *probe* level (chromosome-disjoint train/test splits, see
prepare_genome_probes.py), exactly as the protein setup enforces it at the homology-cluster level —
so pretraining uses the whole genome, mirroring UniRef50.

Smoke subset:
   python -m scripts.prepare_genome --fasta F --out OUT --max-windows 200000 --num-shards 8
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterator, Tuple
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_uniref50 import (  # noqa: E402  (reuse the tested machinery)
    _ShardWriters, _open_maybe_gzip, _download, _SCHEMA,
)

HG38_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
# Main chromosomes only: chr1..22, X, Y (UCSC "chrN" or Ensembl "N"); skip chrM, _alt, _random, chrUn.
_MAIN_CHROM_RE = re.compile(r"^(?:chr)?(?:[1-9]|1[0-9]|2[0-2]|X|Y)$", re.IGNORECASE)
_NON_ACGT = re.compile(r"[^ACGT]")
_RC = str.maketrans("ACGTN", "TGCAN")


def _clean(window: str) -> str:
    """Upper-case; map any non-ACGT character (incl. soft-mask, IUPAC codes) to N."""
    return _NON_ACGT.sub("N", window.upper())


def _revcomp(window: str) -> str:
    return window.translate(_RC)[::-1]


def iter_fasta_records(fileobj) -> Iterator[Tuple[str, str]]:
    """Yield (chrom_name, sequence) per FASTA record; chrom_name = first token of the header."""
    header = None
    parts: list[str] = []
    for line in fileobj:
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(parts)
            header = line[1:].strip().split()[0] if len(line) > 1 else ""
            parts = []
        else:
            parts.append(line.strip())
    if header is not None:
        yield header, "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", type=Path, help="Path to the genome FASTA(.gz).")
    ap.add_argument("--download-url", type=str, default=None,
                    help="If set (and --fasta missing), download here first. Needs internet.")
    ap.add_argument("--out", type=Path, required=True, help="Output dir for shard_*.parquet.")
    ap.add_argument("--num-shards", type=int, default=200)
    ap.add_argument("--rows-per-rowgroup", type=int, default=16384)
    ap.add_argument("--window-len", type=int, default=1024, help="Window length in bp (=context).")
    ap.add_argument("--stride", type=int, default=0,
                    help="Step between window starts (0 -> non-overlapping = window-len).")
    ap.add_argument("--max-n-frac", type=float, default=0.10, help="Drop windows with >this N fraction.")
    ap.add_argument("--reverse-complement", action="store_true",
                    help="Also emit each kept window's reverse complement (DNA augmentation).")
    ap.add_argument("--include-chroms", type=str, default=None,
                    help="Comma-separated explicit chrom whitelist (default: main chr1..22,X,Y).")
    ap.add_argument("--max-windows", type=int, default=0, help="If >0, stop after N windows (subset).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compression", type=str, default="snappy")
    ap.add_argument("--source-url", type=str, default=HG38_URL,
                    help="Source URL recorded in provenance.json.")
    args = ap.parse_args()

    if args.num_shards < 2:
        sys.exit("--num-shards must be >= 2 (the last shard is the validation split).")
    win = args.window_len
    stride = args.stride or win
    if win < 1 or stride < 1:
        sys.exit("--window-len and --stride must be >= 1.")

    fasta = args.fasta
    if fasta is None or not fasta.exists():
        if args.download_url:
            assert fasta is not None, "--fasta must give the destination path when downloading"
            _download(args.download_url, fasta)
        else:
            sys.exit(f"FASTA not found: {fasta}. Provide --fasta, or --download-url to fetch it.")

    args.out.mkdir(parents=True, exist_ok=True)
    if sorted(args.out.glob("shard_*.parquet")):
        sys.exit(f"{args.out} already has shard(s); remove them to re-shard.")

    whitelist = (set(c.strip() for c in args.include_chroms.split(",")) if args.include_chroms else None)
    def _keep_chrom(name: str) -> bool:
        return (name in whitelist) if whitelist is not None else bool(_MAIN_CHROM_RE.match(name))

    rng = random.Random(args.seed)
    writers = _ShardWriters(args.out, args.num_shards, args.rows_per_rowgroup, rng, args.compression)

    kept = dropped_n = 0
    chroms_used: list[str] = []
    t0 = time.time()
    print(f"Windowing {fasta} -> {args.out}  (win={win}, stride={stride}, num_shards={args.num_shards})")
    stop = False
    with _open_maybe_gzip(fasta) as fh:
        for chrom, seq in iter_fasta_records(fh):
            if not _keep_chrom(chrom):
                continue
            chroms_used.append(chrom)
            L = len(seq)
            for start in range(0, L - win + 1, stride):
                w = _clean(seq[start:start + win])
                if w.count("N") / win > args.max_n_frac:
                    dropped_n += 1
                    continue
                writers.add(rng.randrange(args.num_shards), w)
                kept += 1
                if args.reverse_complement:
                    writers.add(rng.randrange(args.num_shards), _revcomp(w))
                    kept += 1
                if args.max_windows and kept >= args.max_windows:
                    stop = True
                    break
            print(f"  {chrom}: {kept:,} windows kept so far, {dropped_n:,} N-dropped "
                  f"({(time.time()-t0)/60:.1f} min)")
            if stop:
                break

    n_shards = writers.close()

    provenance = {
        "dataset": "GRCh38/hg38",
        "source_url": args.source_url,
        "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "fasta_path": str(fasta),
        "fasta_bytes": (fasta.stat().st_size if fasta.exists() else None),
        "window_len": win,
        "stride": stride,
        "max_n_frac": args.max_n_frac,
        "reverse_complement": bool(args.reverse_complement),
        "chroms_used": chroms_used,
        "num_windows": kept,
        "num_dropped_highN": dropped_n,
        "num_shards": n_shards,
        "max_windows": args.max_windows or None,
        "seed": args.seed,
        "tokenizer_alphabet": "genome-char-ACGTN",
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))

    print("=" * 60)
    print(f"  Done: {kept:,} windows kept, {dropped_n:,} dropped (high-N)")
    print(f"  Chromosomes used: {', '.join(chroms_used) or 'NONE'}")
    print(f"  Wrote {n_shards} non-empty shards to {args.out}")
    print(f"  Validation split = the last shard (highest-numbered).")
    print(f"  Provenance: {args.out / 'provenance.json'}")
    print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")
    print("=" * 60)
    if n_shards < 2:
        sys.exit("ERROR: fewer than 2 non-empty shards — need train + val. "
                 "Lower --num-shards or supply more windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
