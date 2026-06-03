"""Data-budget sweep: fixed N, varied D via param_data_ratio."""

from __future__ import annotations

from nanoprot.config import NanoprotConfig
from scripts.gen_sweep_configs import _sweep_config


def test_ratio_varies_data_at_fixed_model() -> None:
    base = None
    residues = {}
    for ratio in (3, 12, 96):
        cfg_dict = _sweep_config("esm2", "S", ratio, seed=0)
        assert cfg_dict["training"]["param_data_ratio"] == ratio
        assert cfg_dict["training"]["total_residues"] is None  # derived from ratio
        # model size is identical across the sweep; only the budget changes
        if base is None:
            base = cfg_dict["model"]
        assert cfg_dict["model"] == base
        # resolve env vars so derivation runs, then read the derived data budget
        d = cfg_dict.copy()
        d["data"] = {**d["data"], "shard_dir": "/tmp"}
        d["checkpointing"] = {**d["checkpointing"], "output_dir": "/tmp"}
        residues[ratio] = NanoprotConfig.model_validate(d).total_residues()
    # D scales linearly with the ratio at fixed N
    assert residues[12] == 4 * residues[3]
    assert residues[96] == 8 * residues[12]


def test_name_encodes_ratio() -> None:
    cfg = _sweep_config("mamba", "S", 24, seed=1)
    assert cfg["name"] == "nanoprot-sweep-mamba-S-r24-s1"
    assert "sweep/nanoprot-sweep-mamba-S-r24-s1" in cfg["checkpointing"]["output_dir"]
