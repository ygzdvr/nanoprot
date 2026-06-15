"""Residual-stream extraction: uniform forward-hook capture across all 3 archs."""

from __future__ import annotations

import pytest
import torch

from nanoprot.config import Esm2ModelConfig, Gpt2ModelConfig, MambaModelConfig
from nanoprot.models import build_model
from nanoprot.eval.probe.extract import (
    extract_layers, get_blocks, n_probe_points, relative_depths, token_identity_features,
)


def _build(cfg):
    """meta-device build -> to_empty -> init_weights (the standard nanoprot pattern)."""
    with torch.device("meta"):
        model = build_model(cfg)
    model = model.to_empty(device="cpu")
    model.init_weights()
    return model


def _tiny_cfg(arch: str):
    common = dict(depth=2, head_dim=32, vocab_size=33, max_seq_len=32)
    if arch == "gpt2":
        return Gpt2ModelConfig(window_pattern="L", **common).derive()
    if arch == "esm2":
        return Esm2ModelConfig(**common).derive()
    if arch == "mamba":
        return MambaModelConfig(**common).derive()
    raise ValueError(arch)


@pytest.mark.slow
@pytest.mark.parametrize("arch", ["gpt2", "esm2", "mamba"])
def test_extract_shapes_and_count(arch: str) -> None:
    cfg = _tiny_cfg(arch)
    model = _build(cfg)
    B, T = 2, 8
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    feats = extract_layers(model, idx)
    # n_layer + 1 probe points: embedding + one per block output
    assert len(feats) == cfg.depth + 1
    assert len(feats) == n_probe_points(model)
    for f in feats:
        assert f.shape == (B, T, cfg.d_model)   # (B, T, d_model) per point
        assert f.is_floating_point()
        assert not f.requires_grad              # detached
        assert torch.isfinite(f).all()


@pytest.mark.slow
@pytest.mark.parametrize("arch", ["gpt2", "esm2", "mamba"])
def test_block_list_resolves(arch: str) -> None:
    assert len(get_blocks(_build(_tiny_cfg(arch)))) == 2


@pytest.mark.slow
def test_extract_does_not_leave_model_in_eval() -> None:
    model = _build(_tiny_cfg("gpt2"))
    model.train()
    extract_layers(model, torch.randint(0, 33, (1, 8)))
    assert model.training is True              # train/eval state restored


def test_relative_depths() -> None:
    assert relative_depths(3) == [0.0, 0.5, 1.0]
    assert relative_depths(1) == [0.0]
    rd = relative_depths(5)
    assert rd[0] == 0.0 and rd[-1] == 1.0


def test_token_identity_features() -> None:
    idx = torch.tensor([[1, 2, 0]])
    f = token_identity_features(idx, 5)
    assert f.shape == (1, 3, 5)
    assert torch.equal(f[0, 0], torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(f.sum(-1), torch.ones(1, 3))   # exactly one-hot per position


def test_get_blocks_rejects_non_nanoprot_module() -> None:
    with pytest.raises(ValueError):
        get_blocks(torch.nn.Linear(4, 4))
