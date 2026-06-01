#!/usr/bin/env python
"""
Prepare the UniRef50 pretraining corpus as shuffled parquet shards.

nanoprot's data loader reads ``shard_*.parquet`` files (a ``text`` column of
sequences; the last shard is the validation split). This script turns the
UniProt UniRef50 FASTA into exactly that, with a global shuffle, in bounded
memory — so the framework ships its own data prep instead of relying on the
(English) ClimbMix downloader inherited from nanochat.

Pipeline:
  1. (once, on a machine with internet — e.g. a login node) download the FASTA:
       wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz \\
            -O $NANOPROT_BASE_DIR/uniref50.fasta.gz
     (or pass --download-url to let this script fetch it).
  2. shard it (no internet needed; CPU only — see runs/prepare_uniref50.slurm):
       python -m scripts.prepare_uniref50 \\
           --fasta   $NANOPROT_BASE_DIR/uniref50.fasta.gz \\
           --out     $NANOPROT_BASE_DIR/uniref50_parquet

Normalisation matches the 33-token ESM-2 alphabet (the release tokenizer):
sequences are upper-cased; the 20 standard residues + X/B/U/Z/O are kept
verbatim, anything else becomes the tokenizer's <unk> at train time. Sequences
shorter than --min-len are dropped.

Shuffle: each sequence is assigned to a uniformly random shard (cross-shard
shuffle) and each row-group buffer is shuffled before writing (within-shard
shuffle); together this approximates a global shuffle without holding the whole
corpus in memory. Multiple row-groups per shard are emitted so DDP ranks (which
stride over row-groups) all get data.

A subset for quick calibration/smoke tests:
   python -m scripts.prepare_uniref50 --fasta F --out OUT --max-sequences 200000 --num-shards 8
"""

from __future__ import annotations

import argparse
import gzip
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


_SCHEMA = pa.schema([("text", pa.string())])


def _open_maybe_gzip(path: Path):
    """Open a possibly-gzipped text file for streaming reads."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def iter_fasta_sequences(fileobj) -> Iterator[str]:
    """Yield sequence strings from a FASTA stream (handles multi-line records)."""
    seq_parts: List[str] = []
    for line in fileobj:
        if line.startswith(">"):
            if seq_parts:
                yield "".join(seq_parts)
                seq_parts = []
        else:
            seq_parts.append(line.strip())
    if seq_parts:  # final record at EOF
        yield "".join(seq_parts)


def _download(url: str, dest: Path) -> None:
    """Stream a (large) file to disk. Needs internet — run on a login node."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"Downloading {url}\n  -> {dest}")
    t0 = time.time()
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        nbytes = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            nbytes += len(chunk)
            if nbytes % (1 << 30) < (1 << 20):  # ~every GB
                print(f"  ... {nbytes/1e9:.1f} GB ({(time.time()-t0)/60:.1f} min)")
    tmp.rename(dest)
    print(f"Downloaded {nbytes/1e9:.2f} GB in {(time.time()-t0)/60:.1f} min")


class _ShardWriters:
    """Holds one ParquetWriter per shard; flushes shuffled row-group buffers."""

    def __init__(self, out: Path, num_shards: int, rows_per_rowgroup: int,
                 rng: random.Random, compression: str = "snappy") -> None:
        self.out = out
        self.num_shards = num_shards
        self.rows_per_rowgroup = rows_per_rowgroup
        self.rng = rng
        self.compression = compression
        self.paths = [out / f"shard_{i:05d}.parquet" for i in range(num_shards)]
        self._writers: List[Optional[pq.ParquetWriter]] = [None] * num_shards
        self._buffers: List[List[str]] = [[] for _ in range(num_shards)]
        self.rows_per_shard = [0] * num_shards

    def add(self, shard: int, seq: str) -> None:
        buf = self._buffers[shard]
        buf.append(seq)
        if len(buf) >= self.rows_per_rowgroup:
            self._flush(shard)

    def _flush(self, shard: int) -> None:
        buf = self._buffers[shard]
        if not buf:
            return
        self.rng.shuffle(buf)  # within-row-group shuffle
        if self._writers[shard] is None:
            self._writers[shard] = pq.ParquetWriter(
                self.paths[shard], _SCHEMA, compression=self.compression
            )
        self._writers[shard].write_table(pa.table({"text": buf}, schema=_SCHEMA))
        self.rows_per_shard[shard] += len(buf)
        self._buffers[shard] = []

    def close(self) -> int:
        for s in range(self.num_shards):
            self._flush(s)
            if self._writers[s] is not None:
                self._writers[s].close()
        # Drop any shard files that never received rows (keeps the loader's
        # "last shard = val" invariant pointing at a real, non-empty split).
        kept = 0
        for s in range(self.num_shards):
            if self.rows_per_shard[s] == 0 and self.paths[s].exists():
                self.paths[s].unlink()
            elif self.rows_per_shard[s] > 0:
                kept += 1
        return kept


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", type=Path, help="Path to uniref50.fasta(.gz).")
    ap.add_argument("--download-url", type=str, default=None,
                    help="If set (and --fasta missing), download the FASTA here first. "
                         "Needs internet — run on a login node.")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for shard_*.parquet.")
    ap.add_argument("--num-shards", type=int, default=200)
    ap.add_argument("--rows-per-rowgroup", type=int, default=16384)
    ap.add_argument("--min-len", type=int, default=20, help="Drop sequences shorter than this.")
    ap.add_argument("--max-len", type=int, default=0, help="If >0, drop sequences longer than this.")
    ap.add_argument("--max-sequences", type=int, default=0, help="If >0, stop after N (subset).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compression", type=str, default="snappy")
    args = ap.parse_args()

    if args.num_shards < 2:
        sys.exit("--num-shards must be >= 2 (the last shard is the validation split).")

    fasta = args.fasta
    if (fasta is None or not fasta.exists()):
        if args.download_url:
            assert fasta is not None, "--fasta must give the destination path when downloading"
            _download(args.download_url, fasta)
        else:
            sys.exit(f"FASTA not found: {fasta}. Provide --fasta, or --download-url to fetch it.")

    args.out.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.out.glob("shard_*.parquet"))
    if existing:
        sys.exit(f"{args.out} already has {len(existing)} shard(s); remove them to re-shard.")

    rng = random.Random(args.seed)
    writers = _ShardWriters(args.out, args.num_shards, args.rows_per_rowgroup, rng, args.compression)

    kept = dropped = 0
    t0 = time.time()
    print(f"Sharding {fasta} -> {args.out}  (num_shards={args.num_shards}, seed={args.seed})")
    with _open_maybe_gzip(fasta) as fh:
        for seq in iter_fasta_sequences(fh):
            seq = seq.strip().upper()
            if len(seq) < args.min_len or (args.max_len and len(seq) > args.max_len):
                dropped += 1
                continue
            writers.add(rng.randrange(args.num_shards), seq)
            kept += 1
            if args.max_sequences and kept >= args.max_sequences:
                break
            if kept % 1_000_000 == 0:
                print(f"  {kept:,} kept, {dropped:,} dropped, {(time.time()-t0)/60:.1f} min")

    n_shards = writers.close()
    total_res = None  # residue count would need a second pass; report seqs only
    print("=" * 60)
    print(f"  Done: {kept:,} sequences kept, {dropped:,} dropped")
    print(f"  Wrote {n_shards} non-empty shards to {args.out}")
    print(f"  Validation split = the last shard (highest-numbered).")
    print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")
    print("=" * 60)
    if n_shards < 2:
        sys.exit("ERROR: fewer than 2 non-empty shards — need a train + val split. "
                 "Lower --num-shards or supply more sequences.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
