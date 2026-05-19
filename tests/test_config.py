"""Tests for ``nanoprot.config``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nanoprot.config import (
    NanoprotConfig,
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

    def test_reference_d20_config_loads_and_derives(self) -> None:
        path = CONFIGS_DIR / "gpt2_d20_uniref50.yaml"
        assert path.is_file(), f"reference config missing at {path}"
        os.environ.setdefault("NANOPROT_BASE_DIR", "/tmp/nanoprot_test")
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
