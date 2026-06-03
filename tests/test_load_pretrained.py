"""Arch-aware checkpoint loading: load_pretrained must round-trip every arch.

The legacy checkpoint.build_model is gpt2+bpe-only; load_pretrained rebuilds from
the embedded config via the registry, so esm2/mamba checkpoints load too. This
guards that contract for all three architectures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nanoprot.config import NanoprotConfig  # noqa: E402
from nanoprot.models import build_model  # noqa: E402
from nanoprot.training.checkpoint import save_checkpoint, load_pretrained  # noqa: E402


def _tiny(arch: str, out: Path) -> NanoprotConfig:
    extra = {"window_pattern": "L"} if arch == "gpt2" else {}
    return NanoprotConfig(
        name=f"{arch}-t",
        model={"arch": arch, "depth": 2, "max_seq_len": 16, "vocab_size": 33,
               "head_dim": 32, **extra},
        data={"shard_dir": "/tmp/d"},
        checkpointing={"output_dir": str(out)},
    )


@pytest.mark.slow
@pytest.mark.parametrize("arch", ["gpt2", "esm2", "mamba"])
def test_load_pretrained_roundtrips_arch(tmp_path: Path, arch: str) -> None:
    out = tmp_path / arch
    cfg = _tiny(arch, out)

    # "train": build + init + save a self-describing checkpoint.
    with torch.device("meta"):
        m = build_model(cfg.model)
    m = m.to_empty(device="cpu")
    m.init_weights()
    meta = {"step": 5, "num_iterations": 5, "config": cfg.model_dump(),
            "model_config": cfg.model.model_dump(), "name": cfg.name}
    save_checkpoint(str(out), 5, m.state_dict(), None, meta, rank=0)

    # load via the arch-aware loader (no arch hint passed — inferred from meta).
    model, cfg2, meta2 = load_pretrained(out, device="cpu")
    assert cfg2.model.arch == arch
    assert meta2["step"] == 5

    # the loaded model runs and produces vocab-sized logits.
    model.eval()
    with torch.no_grad():
        logits = model(torch.randint(0, 33, (1, 16)))
    assert logits.shape[:2] == (1, 16)
    assert logits.shape[2] in (33, 64)  # 33 logical, possibly padded to 64
    assert torch.isfinite(logits).all()


def test_load_pretrained_rejects_pre_v05_checkpoint(tmp_path: Path) -> None:
    """A checkpoint whose meta lacks the embedded config must fail clearly."""
    out = tmp_path / "old"
    cfg = _tiny("gpt2", out)
    with torch.device("meta"):
        m = build_model(cfg.model)
    m = m.to_empty(device="cpu")
    m.init_weights()
    # meta WITHOUT "config" (simulates a pre-v0.5 checkpoint)
    save_checkpoint(str(out), 1, m.state_dict(), None, {"step": 1}, rank=0)
    with pytest.raises(ValueError, match="config"):
        load_pretrained(out, device="cpu")
