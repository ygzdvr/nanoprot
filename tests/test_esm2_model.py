"""Tests for the ESM-2 model: registry, build, forward pass, optimizer setup."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from nanoprot.config import NanoprotConfig  # noqa: E402
from nanoprot.models import build_model, list_archs  # noqa: E402
from nanoprot.models.esm2 import Esm2  # noqa: E402


def _tiny_esm2_config() -> NanoprotConfig:
    return NanoprotConfig(
        name="esm2-tiny",
        model={"arch": "esm2", "depth": 2, "max_seq_len": 16, "vocab_size": 33, "head_dim": 16},
        data={"shard_dir": "/tmp/d"},
        checkpointing={"output_dir": "/tmp/c"},
    )


class TestRegistry:
    def test_esm2_in_list_archs(self) -> None:
        assert "esm2" in list_archs()

    def test_build_returns_esm2_instance(self) -> None:
        m = build_model(_tiny_esm2_config().model)
        assert isinstance(m, Esm2)


class TestBuild:
    def test_param_count_positive(self) -> None:
        m = build_model(_tiny_esm2_config().model)
        assert sum(p.numel() for p in m.parameters()) > 0

    def test_no_value_embeddings(self) -> None:
        """ESM-2 should not have ResFormer value embeddings."""
        m = build_model(_tiny_esm2_config().model)
        counts = m.num_scaling_params()
        assert counts["value_embeds"] == 0

    def test_param_count_scales_with_depth(self) -> None:
        from .conftest import re_derive_model_with
        cfg_small = re_derive_model_with(_tiny_esm2_config(), depth=2)
        cfg_big = re_derive_model_with(_tiny_esm2_config(), depth=6)
        assert cfg_big.model.d_model > cfg_small.model.d_model  # width re-derived
        n_small = sum(p.numel() for p in build_model(cfg_small.model).parameters())
        n_big = sum(p.numel() for p in build_model(cfg_big.model).parameters())
        assert n_big > n_small


@pytest.mark.slow
class TestForward:
    def test_mlm_loss_near_ln_vocab_at_init(self) -> None:
        """At init, ESM-2 outputs uniform predictions; MLM loss on a masked
        position should be ~= ln(V)."""
        cfg = _tiny_esm2_config()
        m = build_model(cfg.model)
        m.init_weights()
        m.eval()

        V = cfg.model.vocab_size
        B, T = 2, 8
        idx = torch.randint(0, V, (B, T), dtype=torch.long)
        # 2 masked positions, rest ignore_index
        tgt = torch.full((B, T), -100, dtype=torch.long)
        tgt[0, 3] = 5
        tgt[1, 5] = 9
        with torch.no_grad():
            loss = m(idx, targets=tgt)
        expected = float(torch.log(torch.tensor(float(V))))
        assert abs(float(loss) - expected) < 0.3, (
            f"loss {float(loss):.4f} too far from uniform expected {expected:.4f}"
        )

    def test_logits_shape_when_targets_none(self) -> None:
        cfg = _tiny_esm2_config()
        m = build_model(cfg.model)
        m.init_weights()
        m.eval()
        idx = torch.randint(0, cfg.model.vocab_size, (1, 8), dtype=torch.long)
        with torch.no_grad():
            logits = m(idx)
        assert logits.shape == (1, 8, cfg.model.vocab_size)

    def test_attention_is_bidirectional(self) -> None:
        """A bidirectional model's output at position i should depend on
        tokens both before AND after i. If we change the LAST token, the
        FIRST position's output must also change."""
        cfg = _tiny_esm2_config()
        m = build_model(cfg.model)
        m.init_weights()
        m.eval()

        idx_a = torch.zeros(1, 8, dtype=torch.long)
        idx_a[0, :] = torch.arange(8) % cfg.model.vocab_size
        idx_b = idx_a.clone()
        idx_b[0, -1] = (idx_b[0, -1] + 7) % cfg.model.vocab_size  # change last token

        with torch.no_grad():
            logits_a = m(idx_a)
            logits_b = m(idx_b)

        # Position 0 logits must differ -> attention reached back from the last token.
        diff = (logits_a[0, 0] - logits_b[0, 0]).abs().max().item()
        assert diff > 1e-4, (
            f"position-0 output identical when last token changed (diff={diff:.2e}); "
            f"attention may be causal, not bidirectional"
        )


class TestOptimizer:
    def test_setup_optimizer_returns_param_groups(self) -> None:
        m = build_model(_tiny_esm2_config().model)
        m.init_weights()
        opt = m.setup_optimizer(matrix_lr=0.01, embedding_lr=0.1)
        assert hasattr(opt, "param_groups")
        assert len(opt.param_groups) > 0
        # Every group must have initial_lr (set by setup_optimizer)
        for g in opt.param_groups:
            assert "initial_lr" in g
