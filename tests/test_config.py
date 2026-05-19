"""Tests for ``nanoprot.config``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nanoprot.config import (
    Esm2ModelConfig,
    Gpt2ModelConfig,
    NanoprotConfig,
    dump_config,
    load_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config_dict() -> dict:
    """Smallest config that satisfies the required fields."""
    return {
        "name": "test-tiny",
        "model": {"depth": 4},
        "data": {"shard_dir": "/tmp/data"},
        "checkpointing": {"output_dir": "/tmp/ckpts"},
    }


# ---------------------------------------------------------------------------
# Schema + derivation
# ---------------------------------------------------------------------------

class TestDerivation:
    def test_d_model_derived_from_depth(self) -> None:
        cfg = NanoprotConfig(**_minimal_config_dict())
        # depth=4, head_dim=128 (default), 4*64=256 rounds up to 256 (mult of 128 already).
        assert cfg.model.d_model == 256

    def test_d_model_rounds_up_to_head_dim_multiple(self) -> None:
        d = _minimal_config_dict()
        d["model"] = {"depth": 3}  # 3*64=192, rounds up to 256 with head_dim=128
        cfg = NanoprotConfig(**d)
        assert cfg.model.d_model == 256

    def test_n_heads_derived(self) -> None:
        cfg = NanoprotConfig(**_minimal_config_dict())
        assert cfg.model.n_heads == cfg.model.d_model // cfg.model.head_dim

    def test_n_kv_heads_defaults_to_n_heads(self) -> None:
        cfg = NanoprotConfig(**_minimal_config_dict())
        assert cfg.model.n_kv_heads == cfg.model.n_heads

    def test_d20_matches_reference(self) -> None:
        """The reference d20 run must compute d_model=1280, n_heads=10."""
        d = _minimal_config_dict()
        d["model"] = {"depth": 20}
        cfg = NanoprotConfig(**d)
        assert cfg.model.d_model == 1280
        assert cfg.model.n_heads == 10


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            NanoprotConfig(model={"depth": 4}, data={"shard_dir": "/tmp/d"})  # no checkpointing

    def test_unknown_field_rejected(self) -> None:
        d = _minimal_config_dict()
        d["model"]["totally_made_up_field"] = 42
        with pytest.raises(ValidationError):
            NanoprotConfig(**d)

    def test_depth_must_be_positive(self) -> None:
        d = _minimal_config_dict()
        d["model"]["depth"] = 0
        with pytest.raises(ValidationError):
            NanoprotConfig(**d)

    def test_invalid_window_pattern_rejected(self) -> None:
        d = _minimal_config_dict()
        d["model"]["window_pattern"] = "SLX"  # X is invalid
        with pytest.raises(ValidationError):
            NanoprotConfig(**d)

    def test_window_pattern_uppercased(self) -> None:
        d = _minimal_config_dict()
        d["model"]["window_pattern"] = "ssl"
        cfg = NanoprotConfig(**d)
        assert cfg.model.window_pattern == "SSL"

    def test_n_heads_not_divisible_by_kv_heads_rejected(self) -> None:
        d = _minimal_config_dict()
        d["model"]["depth"] = 20  # n_heads=10
        d["model"]["n_kv_heads"] = 3  # 10 % 3 != 0
        with pytest.raises(ValidationError):
            NanoprotConfig(**d)


# ---------------------------------------------------------------------------
# Environment-variable expansion
# ---------------------------------------------------------------------------

class TestEnvSubstitution:
    def test_shard_dir_expands_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_DATA", "/scratch/data")
        d = _minimal_config_dict()
        d["data"]["shard_dir"] = "$MY_DATA/uniref50"
        cfg = NanoprotConfig(**d)
        assert cfg.data.shard_dir == "/scratch/data/uniref50"

    def test_output_dir_expands_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_CKPTS", "/scratch/ckpts")
        d = _minimal_config_dict()
        d["checkpointing"]["output_dir"] = "${MY_CKPTS}/run"
        cfg = NanoprotConfig(**d)
        assert cfg.checkpointing.output_dir == "/scratch/ckpts/run"


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_loads_minimal_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "min.yaml"
        path.write_text(yaml.safe_dump(_minimal_config_dict()))
        cfg = load_config(path)
        assert cfg.name == "test-tiny"
        assert cfg.model.depth == 4

    def test_reference_d20_config_loads_and_derives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = CONFIGS_DIR / "gpt2_d20_uniref50.yaml"
        assert path.is_file(), f"reference config missing at {path}"
        monkeypatch.setenv("NANOPROT_BASE_DIR", "/tmp/nanoprot_test")
        cfg = load_config(path)
        # The reference d20 numbers from the docstring.
        assert cfg.model.depth == 20
        assert cfg.model.d_model == 1280
        assert cfg.model.n_heads == 10
        assert cfg.model.n_kv_heads == 10


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------

class TestEstimates:
    def test_estimate_params_grows_with_depth(self) -> None:
        small = _minimal_config_dict()
        small["model"]["depth"] = 4
        big = _minimal_config_dict()
        big["model"]["depth"] = 24
        assert (
            NanoprotConfig(**big).estimate_params()
            > NanoprotConfig(**small).estimate_params()
        )

    def test_total_residues_explicit_overrides_chinchilla(self) -> None:
        d = _minimal_config_dict()
        d["training"] = {"total_residues": 1_000_000}
        cfg = NanoprotConfig(**d)
        assert cfg.total_residues() == 1_000_000

    def test_total_residues_falls_back_to_chinchilla(self) -> None:
        cfg = NanoprotConfig(**_minimal_config_dict())
        n = cfg.total_residues()
        assert n == int(cfg.training.param_data_ratio * cfg.estimate_params())

    def test_estimate_includes_value_embeddings_for_gpt2(self) -> None:
        """gpt2 arch should add ResFormer value embeddings on alternating layers.

        The closed-form estimate for d20 used to be ~522 M (just attention + MLP
        + token embedding + lm_head). With value embeddings on 10 of the 20
        layers, the estimate climbs to ~1.2 B, much closer to the real 1.17 B.
        """
        d = _minimal_config_dict()
        d["model"] = {"depth": 20, "vocab_size": 50_256}
        cfg = NanoprotConfig(**d)
        n = cfg.estimate_params()
        # Lower bound: at least the core (12 d^2 L + 2 V d) ~ 521 M.
        # Upper bound: well under 2 B (would mean we're double-counting).
        assert 900_000_000 < n < 1_500_000_000, f"got {n:,} (~{n / 1e9:.2f} B)"


class TestSeedInheritance:
    def test_training_seed_inherits_from_global_when_unset(self) -> None:
        d = _minimal_config_dict()
        d["seed"] = 99
        cfg = NanoprotConfig(**d)
        assert cfg.training.seed == 99

    def test_training_seed_respected_when_explicit(self) -> None:
        d = _minimal_config_dict()
        d["seed"] = 99
        d["training"] = {"seed": 123}
        cfg = NanoprotConfig(**d)
        # Explicit training seed wins over global seed.
        assert cfg.training.seed == 123

    def test_global_seed_default_propagates(self) -> None:
        cfg = NanoprotConfig(**_minimal_config_dict())
        # Default global seed is 42; training.seed (None) inherits.
        assert cfg.seed == 42
        assert cfg.training.seed == 42


class TestDumpConfigRoundTrip:
    def test_dump_then_load_preserves_values(self, tmp_path: Path) -> None:
        original_dict = _minimal_config_dict()
        original_dict["seed"] = 17
        original_dict["model"]["depth"] = 8
        original_dict["training"] = {"total_batch_size": 131072, "device_batch_size": 4}
        original = NanoprotConfig(**original_dict)

        out = tmp_path / "roundtrip.yaml"
        dump_config(original, out)
        reloaded = load_config(out)

        # Top-level identifiers
        assert reloaded.name == original.name
        assert reloaded.seed == original.seed
        # Model + derivations
        assert reloaded.model.depth == original.model.depth
        assert reloaded.model.d_model == original.model.d_model
        assert reloaded.model.n_heads == original.model.n_heads
        # Training
        assert reloaded.training.total_batch_size == 131072
        assert reloaded.training.device_batch_size == 4
        # Seed inheritance round-trips
        assert reloaded.training.seed == 17

    def test_dump_creates_parent_dirs(self, tmp_path: Path) -> None:
        cfg = NanoprotConfig(**_minimal_config_dict())
        out = tmp_path / "nested" / "dir" / "cfg.yaml"
        dump_config(cfg, out)
        assert out.is_file()


# ---------------------------------------------------------------------------
# Discriminated-union and ESM-2 model config
# ---------------------------------------------------------------------------

class TestDiscriminatedUnion:
    def test_missing_arch_defaults_to_gpt2_via_load_config(self, tmp_path: Path) -> None:
        # Older configs without an explicit arch should still parse as gpt2.
        path = tmp_path / "no_arch.yaml"
        path.write_text(yaml.safe_dump(_minimal_config_dict()))
        cfg = load_config(path)
        assert isinstance(cfg.model, Gpt2ModelConfig)

    def test_explicit_esm2_arch_yields_esm2_config(self, tmp_path: Path) -> None:
        d = _minimal_config_dict()
        d["model"] = {"arch": "esm2", "depth": 6}
        path = tmp_path / "esm2.yaml"
        path.write_text(yaml.safe_dump(d))
        cfg = load_config(path)
        assert isinstance(cfg.model, Esm2ModelConfig)
        # ESM-2 has different defaults
        assert cfg.model.vocab_size == 33
        assert cfg.model.head_dim == 64

    def test_unknown_arch_rejected_by_discriminator(self) -> None:
        d = _minimal_config_dict()
        d["model"] = {"arch": "made_up", "depth": 4}
        with pytest.raises(ValidationError):
            NanoprotConfig(**d)

    def test_gpt2_window_pattern_not_on_esm2(self) -> None:
        # ``window_pattern`` is gpt2-only; ESM-2 config should reject it via extra="forbid".
        d = _minimal_config_dict()
        d["model"] = {"arch": "esm2", "depth": 6, "window_pattern": "SSL"}
        with pytest.raises(ValidationError):
            NanoprotConfig(**d)


class TestEsm2Derivation:
    def test_d_model_uses_40_per_depth_multiplier(self) -> None:
        d = _minimal_config_dict()
        d["model"] = {"arch": "esm2", "depth": 8}
        cfg = NanoprotConfig(**d)
        # 8 * 40 = 320, already a multiple of head_dim=64
        assert cfg.model.d_model == 320

    def test_d_model_rounds_up_to_head_dim(self) -> None:
        d = _minimal_config_dict()
        d["model"] = {"arch": "esm2", "depth": 5, "head_dim": 64}
        cfg = NanoprotConfig(**d)
        # 5 * 40 = 200, rounds up to 256 (next multiple of 64)
        assert cfg.model.d_model == 256
        assert cfg.model.n_heads == 4

    def test_estimate_params_does_not_include_value_embeddings_for_esm2(self) -> None:
        # ESM-2 has no ResFormer value embeddings, so the closed-form estimate
        # is just the core transformer + token embedding + lm_head.
        d = _minimal_config_dict()
        d["model"] = {"arch": "esm2", "depth": 6}
        cfg = NanoprotConfig(**d)
        # Compute expected lower bound (just core + embeddings, no ve term).
        m = cfg.model
        assert m.d_model is not None
        expected = 12 * m.d_model * m.d_model * m.depth + 2 * m.vocab_size * m.d_model
        assert cfg.estimate_params() == expected


# ---------------------------------------------------------------------------
# Training-objective field
# ---------------------------------------------------------------------------

class TestTrainingObjective:
    def test_default_objective_is_ar(self) -> None:
        cfg = NanoprotConfig(**_minimal_config_dict())
        assert cfg.training.objective == "ar"

    def test_objective_mlm_accepted(self) -> None:
        d = _minimal_config_dict()
        d["training"] = {"objective": "mlm", "mlm_probability": 0.2}
        cfg = NanoprotConfig(**d)
        assert cfg.training.objective == "mlm"
        assert abs(cfg.training.mlm_probability - 0.2) < 1e-9

    def test_invalid_objective_rejected(self) -> None:
        d = _minimal_config_dict()
        d["training"] = {"objective": "contrastive"}
        with pytest.raises(ValidationError):
            NanoprotConfig(**d)

    def test_mlm_probability_must_be_in_open_unit_interval(self) -> None:
        for bad in (0.0, 1.0, 1.5, -0.1):
            d = _minimal_config_dict()
            d["training"] = {"objective": "mlm", "mlm_probability": bad}
            with pytest.raises(ValidationError):
                NanoprotConfig(**d)


class TestTokenizerDispatch:
    def test_default_tokenizer_is_bpe(self) -> None:
        cfg = NanoprotConfig(**_minimal_config_dict())
        assert cfg.tokenizer.name == "bpe"

    def test_esm2_tokenizer_accepted(self) -> None:
        d = _minimal_config_dict()
        d["tokenizer"] = {"name": "esm2"}
        cfg = NanoprotConfig(**d)
        assert cfg.tokenizer.name == "esm2"

    def test_unknown_tokenizer_rejected(self) -> None:
        d = _minimal_config_dict()
        d["tokenizer"] = {"name": "sentencepiece"}
        with pytest.raises(ValidationError):
            NanoprotConfig(**d)
