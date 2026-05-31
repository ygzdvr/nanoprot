"""``data.shard_dir`` must actually select the dataset.

Regression guard: the loader historically read a module-level ``DATA_DIR``
(env ``NANOPROT_DATA_DIR``, default = English ClimbMix shards) and ignored the
config's ``data.shard_dir``. A config-driven framework whose data-path field is
dead silently trains on the wrong corpus. These tests pin that ``shard_dir`` is
honored for both objectives, overriding even a (bogus) ``NANOPROT_DATA_DIR``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from nanoprot.config import NanoprotConfig  # noqa: E402
from nanoprot.data.builder import build_data_loader  # noqa: E402


_SEQS = ["MKTAYIAKQR" * 5, "GGGSGGGSEEEE" * 4, "ACDEFGHIKLMNPQRSTVWY" * 3, "MVLSPADKTNVKAAW" * 3]


def _write_shards(d: Path, n: int = 2) -> None:
    for i in range(n):
        pq.write_table(pa.table({"text": _SEQS}), d / f"shard_{i:05d}.parquet")


def _cfg(shard_dir: Path, objective: str) -> NanoprotConfig:
    return NanoprotConfig(
        name="t",
        model={"arch": "gpt2" if objective == "ar" else "esm2",
               "depth": 2, "max_seq_len": 16, "vocab_size": 33, "head_dim": 32,
               **({"window_pattern": "L"} if objective == "ar" else {})},
        tokenizer={"name": "esm2"},
        data={"shard_dir": str(shard_dir)},
        training={"objective": objective, "device_batch_size": 2},
        checkpointing={"output_dir": "/tmp/c"},
    )


@pytest.mark.parametrize("objective", ["ar", "mlm"])
def test_shard_dir_is_honored(tmp_path: Path, monkeypatch, objective: str) -> None:
    data = tmp_path / "uniref50_parquet"
    data.mkdir()
    _write_shards(data)
    # Point the legacy env override at a bogus path: shard_dir must win.
    monkeypatch.setenv("NANOPROT_DATA_DIR", str(tmp_path / "BOGUS_climbmix"))

    cfg = _cfg(data, objective)
    loader = build_data_loader(cfg, split="train", device="cpu")
    idx, targets = next(loader)

    assert idx.shape == (2, 16)
    assert int(idx.max()) < 33, "tokens must be within the 33-residue alphabet"
    # AR: targets are shifted inputs (valid ids). MLM: targets are ids or -100.
    if objective == "mlm":
        valid = (targets == -100) | ((targets >= 0) & (targets < 33))
        assert bool(valid.all())


def test_missing_shard_dir_errors_clearly(tmp_path: Path, monkeypatch) -> None:
    """A missing explicit shard_dir must fail loudly and specifically — never
    silently fall back to the legacy/default ClimbMix data."""
    monkeypatch.setenv("NANOPROT_DATA_DIR", str(tmp_path / "also_empty"))
    cfg = _cfg(tmp_path / "does_not_exist", "ar")
    loader = build_data_loader(cfg, split="train", device="cpu")
    with pytest.raises(FileNotFoundError, match="shard_dir"):
        next(loader)
