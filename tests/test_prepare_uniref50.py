"""The UniRef50 data-prep must produce shards the real loader can consume.

Guards: multi-line FASTA parsing, the train/val shard split, multiple row-groups
per shard (so DDP ranks striding over row-groups all get data), and that
``build_data_loader`` reads the produced shards for both objectives.
"""

from __future__ import annotations

import gzip
import importlib.util
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from nanoprot.config import NanoprotConfig  # noqa: E402
from nanoprot.data.builder import build_data_loader  # noqa: E402

# Load scripts/prepare_uniref50.py by path (scripts/ is not a package).
_PREP_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_uniref50.py"
_spec = importlib.util.spec_from_file_location("prepare_uniref50", _PREP_PATH)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)  # type: ignore[union-attr]


def test_iter_fasta_handles_multiline(tmp_path: Path) -> None:
    fasta = tmp_path / "x.fasta"
    fasta.write_text(
        ">UniRef50_A desc\nMKTAY\nIAKQR\n>UniRef50_B\nGGGSGGGS\n>UniRef50_C\nACDEF\n"
    )
    with open(fasta) as fh:
        seqs = list(prep.iter_fasta_sequences(fh))
    assert seqs == ["MKTAYIAKQR", "GGGSGGGS", "ACDEF"]  # multi-line record joined


def _synth_fasta(path: Path, n: int = 800) -> None:
    rng = random.Random(0)
    AA = "LAGVSERTIDPKQNFYMHWCXBUZO"
    with gzip.open(path, "wt") as f:
        for i in range(n):
            seq = "".join(rng.choice(AA) for _ in range(rng.randint(30, 200)))
            f.write(f">UniRef50_T{i:05d}\n")
            for j in range(0, len(seq), 60):
                f.write(seq[j:j + 60] + "\n")


def test_shard_and_loader_roundtrip(tmp_path: Path, monkeypatch) -> None:
    fasta = tmp_path / "synth.fasta.gz"
    _synth_fasta(fasta, n=800)
    out = tmp_path / "uniref50_parquet"
    out.mkdir()

    # Shard with the module's own writer (tiny row-groups to force >1 per shard).
    rng = random.Random(1)
    writers = prep._ShardWriters(out, num_shards=4, rows_per_rowgroup=64, rng=rng)
    with prep._open_maybe_gzip(fasta) as fh:
        for seq in prep.iter_fasta_sequences(fh):
            writers.add(rng.randrange(4), seq.strip().upper())
    n_shards = writers.close()

    shards = sorted(out.glob("shard_*.parquet"))
    assert n_shards >= 2 and len(shards) == n_shards
    # Multiple row-groups per shard is required for DDP striding.
    assert all(pq.ParquetFile(s).num_row_groups > 1 for s in shards)

    # The real loader consumes them (train + val), for both objectives.
    monkeypatch.setenv("NANOPROT_DATA_DIR", str(tmp_path / "BOGUS"))
    for objective, arch in [("ar", "gpt2"), ("mlm", "esm2")]:
        cfg = NanoprotConfig(
            name="t",
            model={"arch": arch, "depth": 2, "max_seq_len": 32, "vocab_size": 33,
                   "head_dim": 32, **({"window_pattern": "L"} if objective == "ar" else {})},
            tokenizer={"name": "esm2"},
            data={"shard_dir": str(out)},
            training={"objective": objective, "device_batch_size": 2},
            checkpointing={"output_dir": "/tmp/c"},
        )
        idx, _ = next(build_data_loader(cfg, split="train", device="cpu"))
        vidx, _ = next(build_data_loader(cfg, split="val", device="cpu"))
        assert idx.shape == (2, 32) and int(idx.max()) < 33
        assert vidx.shape == (2, 32)
