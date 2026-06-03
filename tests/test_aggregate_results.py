"""Results aggregation: only-done filtering + per-(arch,scale) mean/std."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_results import collect_cells, aggregate


def _cell(root: Path, arch: str, scale: str, seed: int, *, bpr: float,
          step: int, num_iter: int, objective: str = "mlm", n_params: int = 1_000_000) -> None:
    d = root / f"nanoprot-{arch}-{scale}-s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "step": step, "num_iterations": num_iter,
        "last_val_bpr": bpr, "best_val_bpr": bpr,
        "n_params": n_params, "total_residues": 12 * n_params,
        "total_flops": 72.0 * n_params ** 2, "train_time_sec": 100.0,
        "config": {"training": {"objective": objective}},
        "name": f"nanoprot-{arch}-{scale}-s{seed}",
    }
    (d / f"meta_{step:06d}.json").write_text(json.dumps(meta))


def test_only_done_cells_are_collected(tmp_path: Path) -> None:
    _cell(tmp_path, "esm2", "S", 0, bpr=3.6, step=761, num_iter=761)   # done
    _cell(tmp_path, "esm2", "S", 1, bpr=3.7, step=400, num_iter=761)   # partial
    done = collect_cells(tmp_path, only_done=True)
    allc = collect_cells(tmp_path, only_done=False)
    assert len(done) == 1 and done[0]["seed"] == 0
    assert len(allc) == 2


def test_aggregate_mean_std_across_seeds(tmp_path: Path) -> None:
    for seed, bpr in [(0, 3.30), (1, 3.32), (2, 3.34)]:
        _cell(tmp_path, "esm2", "L", seed, bpr=bpr, step=14857, num_iter=14857)
    agg = aggregate(collect_cells(tmp_path))
    assert len(agg) == 1
    row = agg[0]
    assert row["arch"] == "esm2" and row["scale"] == "L" and row["n_seeds"] == 3
    assert abs(row["val_bpr_mean"] - 3.32) < 1e-9        # mean of 3.30/3.32/3.34
    assert row["val_bpr_std"] == pytest.approx(0.02, abs=1e-9)  # sample std


def test_aggregate_sorted_by_arch_then_scale(tmp_path: Path) -> None:
    _cell(tmp_path, "esm2", "L", 0, bpr=3.3, step=1, num_iter=1)
    _cell(tmp_path, "esm2", "XS", 0, bpr=3.7, step=1, num_iter=1)
    _cell(tmp_path, "gpt2", "S", 0, bpr=3.6, step=1, num_iter=1, objective="ar")
    agg = aggregate(collect_cells(tmp_path))
    order = [(r["arch"], r["scale"]) for r in agg]
    assert order == [("esm2", "XS"), ("esm2", "L"), ("gpt2", "S")]  # XS before L
