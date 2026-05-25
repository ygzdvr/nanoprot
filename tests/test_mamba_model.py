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
        from .conftest import re_derive_model_with
        cfg_small = re_derive_model_with(_tiny_mamba_config(), depth=2)
        cfg_big = re_derive_model_with(_tiny_mamba_config(), depth=6)
        # Width AND dt_rank must both have re-derived from the new depth.
        assert cfg_big.model.d_model > cfg_small.model.d_model
        assert cfg_big.model.dt_rank >= cfg_small.model.dt_rank
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

    def test_scan_preserves_input_dtype(self) -> None:
        """Output dtype must match input dtype (even though internals upcast
        to fp32). This is the contract the surrounding model relies on when
        composing with bf16 activations."""
        B, L, D, N = 1, 6, 4, 4
        for in_dtype in (torch.float32, torch.bfloat16):
            x = torch.randn(B, L, D, dtype=in_dtype)
            dt = (torch.rand(B, L, D) * 0.1).to(in_dtype)
            A = -torch.exp(torch.randn(D, N)).to(in_dtype)
            Bt = torch.randn(B, L, N, dtype=in_dtype)
            Ct = torch.randn(B, L, N, dtype=in_dtype)
            D_skip = torch.randn(D, dtype=in_dtype)
            y = selective_scan_ref(x, dt, A, Bt, Ct, D_skip)
            assert y.dtype == in_dtype, (
                f"scan changed dtype: in={in_dtype} out={y.dtype}"
            )

    def test_scan_bf16_matches_fp32_internally(self) -> None:
        """The bf16 scan (with internal fp32 upcast) must agree with the
        pure-fp32 scan on the same numerical inputs, within bf16-quantization
        tolerance. This is the test that would have FAILED before C1: if the
        recurrence runs in bf16, ``h`` drifts and outputs diverge as L grows.
        With internal fp32 upcasting, the only error source is the bf16
        casting at function entry / exit."""
        torch.manual_seed(0)
        B, L, D, N = 2, 64, 8, 16
        x = torch.randn(B, L, D)
        dt = torch.rand(B, L, D) * 0.05
        A = -torch.exp(torch.randn(D, N))
        Bt = torch.randn(B, L, N)
        Ct = torch.randn(B, L, N)
        D_skip = torch.randn(D)

        y_fp32 = selective_scan_ref(x, dt, A, Bt, Ct, D_skip)
        y_bf16 = selective_scan_ref(
            x.bfloat16(), dt.bfloat16(), A.bfloat16(),
            Bt.bfloat16(), Ct.bfloat16(), D_skip.bfloat16(),
        ).float()

        rel_err = (y_fp32 - y_bf16).abs().max().item() / (y_fp32.abs().max().item() + 1e-9)
        # bf16 has ~3 decimal digits of mantissa. With the input cast applied
        # twice (in + out) we expect ~1% relative error. A regression that
        # drops the internal fp32 upcast would show O(10%-100%) error.
        assert rel_err < 0.05, (
            f"bf16 scan deviates from fp32 by rel_err={rel_err:.4f}; this "
            f"is the symptom of running the SSM recurrence in bf16 instead "
            f"of upcasting to fp32 internally."
        )


class TestSafeDefaults:
    """Direct construction (no meta + to_empty + init_weights) must yield a
    finite, mathematically valid model. init_weights still gives the canonical
    Mamba init, but these tests guard against latent NaN bugs of the kind
    that hit v0.2.1's smear_gate."""

    def test_a_log_and_d_finite_without_init_weights(self) -> None:
        m = build_model(_tiny_mamba_config().model)
        for resblock in m.transformer.h:
            blk = resblock.mixer
            assert torch.isfinite(blk.A_log).all(), "A_log has non-finite values"
            assert torch.isfinite(blk.D).all(), "D has non-finite values"
            # A = -exp(A_log) with A_log=0 gives A=-1 (canonical S4D-Real
            # starting point). It's not the Mamba paper init, but it's the
            # right sign and won't blow up.
            assert (blk.A_log == 0.0).all(), (
                "A_log default should be zero (so A = -1, stable decay)"
            )
            assert (blk.D == 0.0).all(), "D default should be zero (no skip)"


class TestEstimateParams:
    """The closed-form ``cfg.estimate_params()`` is used to derive the
    Chinchilla training horizon when ``training.total_residues`` is null.
    For Mamba we now use a per-arch formula; this test ensures it stays
    within ~10% of the real instantiated parameter count, so the data
    budget doesn't drift."""

    def test_mamba_estimate_within_tolerance_of_actual(self) -> None:
        cfg = _tiny_mamba_config()
        m = build_model(cfg.model)
        actual = sum(p.numel() for p in m.parameters())
        estimated = cfg.estimate_params()
        # Tolerance: formula omits small terms (conv1d, A_log, D, RMSNorm
        # scalars, vocab padding from 64→64-multiple). Real ≤ formula+small.
        # Should agree to within ~10% at this scale.
        ratio = abs(estimated - actual) / max(actual, 1)
        assert ratio < 0.10, (
            f"Mamba estimate_params drift too large: "
            f"estimated={estimated:,}, actual={actual:,}, ratio={ratio:.3f}. "
            f"The closed-form formula in NanoprotConfig.estimate_params "
            f"is out of sync with nanoprot.models.mamba.Mamba."
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
