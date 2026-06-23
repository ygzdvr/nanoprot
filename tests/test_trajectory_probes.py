"""Tests for the trajectory probe driver's pure helpers (scripts/run_trajectory_probes.py).

The GPU path is exercised by the real runs; here we lock the load-bearing pure logic:
step enumeration, validation-based layer selection, and the FIXED-baseline delta.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_trajectory_probes import (  # noqa: E402
    best_layer_by_val, delta_per_layer, list_checkpoint_steps,
)


def test_list_checkpoint_steps(tmp_path):
    for s in [1, 5, 88, 826, 2]:
        (tmp_path / f"model_{s:06d}.pt").write_text("x")
    (tmp_path / "meta_000001.json").write_text("{}")     # not a model_*.pt -> ignored
    assert list_checkpoint_steps(tmp_path) == [1, 2, 5, 88, 826]


def test_best_layer_selected_on_validation_not_test():
    per_layer = [
        {"layer": 0, "val": {"macro_f1": 0.3}, "test": {"macro_f1": 0.9}},  # high test, low val
        {"layer": 1, "val": {"macro_f1": 0.7}, "test": {"macro_f1": 0.5}},  # best val
        {"layer": 2, "val": {"macro_f1": 0.6}, "test": {"macro_f1": 0.6}},
    ]
    assert best_layer_by_val(per_layer, "macro_f1") == 1   # picks on VAL (no test peeking)


def test_delta_subtracts_fixed_baseline_per_layer():
    learned = [{"layer": 0, "test": {"macro_f1": 0.60}}, {"layer": 1, "test": {"macro_f1": 0.50}}]
    baseline = [{"layer": 0, "test": {"macro_f1": 0.40}}, {"layer": 1, "test": {"macro_f1": 0.45}}]
    d = delta_per_layer(learned, baseline, "macro_f1")
    assert abs(d[0] - 0.20) < 1e-9
    assert abs(d[1] - 0.05) < 1e-9
