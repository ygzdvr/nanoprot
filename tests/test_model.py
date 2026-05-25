"""Tests for the model registry + GPT-2 model instantiation."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from nanoprot.config import NanoprotConfig  # noqa: E402
from nanoprot.models import build_model, list_archs  # noqa: E402


def _tiny_config() -> NanoprotConfig:
    """A config small enough to instantiate + forward on CPU in <1s."""
    return NanoprotConfig(
        name="tiny-test",
        model={
            "depth": 2,
            "max_seq_len": 16,
            "vocab_size": 64,
            "head_dim": 32,
            "window_pattern": "L",
        },
        data={"shard_dir": "/tmp/d"},
        checkpointing={"output_dir": "/tmp/c"},
    )


class TestRegistry:
    def test_gpt2_registered(self) -> None:
        assert "gpt2" in list_archs()

    def test_unknown_arch_rejected(self) -> None:
        cfg = _tiny_config()
        cfg.model.arch = "totally_made_up"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="unknown model arch"):
            build_model(cfg.model)


class TestGpt2Build:
    def test_builds(self) -> None:
        cfg = _tiny_config()
        m = build_model(cfg.model)
        assert m is not None

    def test_param_count_positive(self) -> None:
        cfg = _tiny_config()
        m = build_model(cfg.model)
        n = sum(p.numel() for p in m.parameters())
        assert n > 0

    def test_param_count_scales_with_depth(self) -> None:
        # Round-trip through re_derive_model_with so that d_model also
        # re-derives from the new depth. Naive `cfg.model.depth = 6 +
        # model_validate(model_dump())` is a silent no-op for width.
        from .conftest import re_derive_model_with
        small = re_derive_model_with(_tiny_config(), depth=2)
        big = re_derive_model_with(_tiny_config(), depth=6)
        # Width should actually have scaled — that's the whole point.
        assert big.model.d_model > small.model.d_model
        n_small = sum(p.numel() for p in build_model(small.model).parameters())
        n_big = sum(p.numel() for p in build_model(big.model).parameters())
        # More layers AND wider model => strictly more params.
        assert n_big > n_small


@pytest.mark.slow
class TestGpt2Forward:
    """Forward-pass test on CPU. Marked slow because it actually allocates
    and runs the model; the others are pure shape / config checks."""

    def test_forward_loss_near_uniform_at_init(self) -> None:
        cfg = _tiny_config()
        m = build_model(cfg.model)
        m.init_weights()
        m.eval()
        V = cfg.model.vocab_size
        idx = torch.randint(0, V, (2, 8), dtype=torch.long)
        targets = torch.randint(0, V, (2, 8), dtype=torch.long)
        with torch.no_grad():
            loss = m(idx, targets=targets)
        # At init, the model should predict ~uniformly over vocab => loss ≈ ln(V)
        expected = float(torch.log(torch.tensor(float(V))))
        assert abs(float(loss) - expected) < 0.2, (
            f"loss {float(loss):.4f} too far from uniform expected {expected:.4f}"
        )
