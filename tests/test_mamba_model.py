"""Tests for the Mamba model: registry, build, forward, causality, optimizer."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from nanoprot.config import NanoprotConfig  # noqa: E402
from nanoprot.models import build_model, list_archs  # noqa: E402
from nanoprot.models.mamba import Mamba, selective_scan_ref  # noqa: E402


def _tiny_mamba_config() -> NanoprotConfig:
    return NanoprotConfig(
        name="mamba-tiny",
        model={
            "arch": "mamba",
            "depth": 2,
            "max_seq_len": 16,
            "vocab_size": 64,
            "d_state": 8,
            "d_conv": 4,
            "expand": 2,
            "head_dim": 32,
        },
        data={"shard_dir": "/tmp/d"},
        checkpointing={"output_dir": "/tmp/c"},
    )


class TestRegistry:
    def test_mamba_in_list_archs(self) -> None:
        assert "mamba" in list_archs()

    def test_build_returns_mamba_instance(self) -> None:
        m = build_model(_tiny_mamba_config().model)
        assert isinstance(m, Mamba)


class TestDerivation:
    def test_dt_rank_auto_derived(self) -> None:
        cfg = _tiny_mamba_config()
        # d_model=128, head_dim=32; dt_rank default = max(d_model // 16, 1) = 8.
        assert cfg.model.dt_rank == 8

    def test_dt_rank_respected_when_explicit(self) -> None:
        d = _tiny_mamba_config().model_dump()
        d["model"]["dt_rank"] = 4
        cfg = NanoprotConfig.model_validate(d)
        assert cfg.model.dt_rank == 4


class TestBuild:
    def test_param_count_positive(self) -> None:
        m = build_model(_tiny_mamba_config().model)
        assert sum(p.numel() for p in m.parameters()) > 0

    def test_no_value_embeddings_in_scaling_params(self) -> None:
        m = build_model(_tiny_mamba_config().model)
        assert m.num_scaling_params()["value_embeds"] == 0

    def test_param_count_scales_with_depth(self) -> None:
        cfg_small = _tiny_mamba_config()
        cfg_small.model.depth = 2
        cfg_big = _tiny_mamba_config()
        cfg_big.model.depth = 6
        cfg_small = NanoprotConfig.model_validate(cfg_small.model_dump())
        cfg_big = NanoprotConfig.model_validate(cfg_big.model_dump())
        n_small = sum(p.numel() for p in build_model(cfg_small.model).parameters())
        n_big = sum(p.numel() for p in build_model(cfg_big.model).parameters())
        assert n_big > n_small


@pytest.mark.slow
class TestForward:
    def test_ar_loss_near_ln_vocab_at_init(self) -> None:
        cfg = _tiny_mamba_config()
        m = build_model(cfg.model)
        m.init_weights()
        m.eval()
        V = cfg.model.vocab_size
        B, T = 2, 8
        idx = torch.randint(0, V, (B, T), dtype=torch.long)
        tgt = torch.randint(0, V, (B, T), dtype=torch.long)
        with torch.no_grad():
            loss = m(idx, targets=tgt)
        expected = float(torch.log(torch.tensor(float(V))))
        assert abs(float(loss) - expected) < 0.3, (
            f"loss {float(loss):.4f} too far from uniform expected {expected:.4f}"
        )

    def test_logits_shape_when_targets_none(self) -> None:
        cfg = _tiny_mamba_config()
        m = build_model(cfg.model)
        m.init_weights()
        m.eval()
        idx = torch.randint(0, cfg.model.vocab_size, (1, 8), dtype=torch.long)
        with torch.no_grad():
            logits = m(idx)
        assert logits.shape == (1, 8, cfg.model.vocab_size)

    def test_attention_is_causal(self) -> None:
        """Changing a token at position p must NOT affect output at positions
        < p (causality by construction of the SSM scan + causal depthwise
        conv). This is the headline correctness property for Mamba."""
        cfg = _tiny_mamba_config()
        m = build_model(cfg.model)
        m.init_weights()
        m.eval()

        idx_a = torch.zeros(1, 8, dtype=torch.long)
        idx_a[0, :] = torch.arange(8) % cfg.model.vocab_size
        idx_b = idx_a.clone()
        p = 5
        idx_b[0, p] = (idx_b[0, p] + 17) % cfg.model.vocab_size

        with torch.no_grad():
            logits_a = m(idx_a)
            logits_b = m(idx_b)

        diff_before = (logits_a[0, :p] - logits_b[0, :p]).abs().max().item()
        diff_after = (logits_a[0, p:] - logits_b[0, p:]).abs().max().item()
        assert diff_before < 1e-5, (
            f"non-causal! changing token at position {p} affected output at "
            f"positions < {p} (diff={diff_before:.2e})"
        )
        assert diff_after > 1e-4, (
            f"change at position {p} did not propagate forward (diff={diff_after:.2e})"
        )


class TestSelectiveScan:
    def test_scan_shape(self) -> None:
        B, L, D, N = 2, 5, 4, 3
        x = torch.randn(B, L, D)
        dt = torch.rand(B, L, D) * 0.1
        A = -torch.exp(torch.randn(D, N))
        Bt = torch.randn(B, L, N)
        Ct = torch.randn(B, L, N)
        D_skip = torch.randn(D)
        y = selective_scan_ref(x, dt, A, Bt, Ct, D_skip)
        assert y.shape == (B, L, D)

    def test_scan_first_step_uses_zero_state(self) -> None:
        """At t=0 the recurrence reduces to h_0 = dt_0 * B_0 * x_0; y_0 must
        depend only on x_0, B_0, C_0, D_skip — not on later inputs."""
        B, L, D, N = 1, 4, 2, 3
        x1 = torch.randn(B, L, D)
        x2 = x1.clone()
        x2[:, 1:, :] += 5.0   # perturb everything after t=0
        dt = torch.rand(B, L, D) * 0.1
        A = -torch.exp(torch.randn(D, N))
        Bt = torch.randn(B, L, N)
        Ct = torch.randn(B, L, N)
        D_skip = torch.randn(D)
        y1 = selective_scan_ref(x1, dt, A, Bt, Ct, D_skip)
        y2 = selective_scan_ref(x2, dt, A, Bt, Ct, D_skip)
        # y[:, 0] should be identical.
        assert torch.allclose(y1[:, 0], y2[:, 0], atol=1e-6), (
            "selective_scan_ref violates causality at t=0"
        )


class TestOptimizer:
    def test_setup_optimizer_returns_param_groups(self) -> None:
        m = build_model(_tiny_mamba_config().model)
        m.init_weights()
        opt = m.setup_optimizer(matrix_lr=0.01, embedding_lr=0.1)
        assert hasattr(opt, "param_groups")
        assert len(opt.param_groups) > 0
        for g in opt.param_groups:
            assert "initial_lr" in g

    def test_all_params_covered_by_optimizer(self) -> None:
        """Every parameter in the model must be in exactly one optimizer group."""
        m = build_model(_tiny_mamba_config().model)
        m.init_weights()
        opt = m.setup_optimizer()
        all_model = {id(p) for p in m.parameters()}
        all_opt = set()
        for g in opt.param_groups:
            for p in g["params"]:
                all_opt.add(id(p))
        assert all_model == all_opt, (
            "Mamba.setup_optimizer must cover every parameter exactly once; "
            f"missing={len(all_model - all_opt)} extra={len(all_opt - all_model)}"
        )
