"""Tests for emergence metrics (scripts/analyze_emergence.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_emergence import auec, emergence_times, layer_star_over_time  # noqa: E402


def test_emergence_times_onset_and_fractions():
    steps = [1, 2, 4, 8, 16, 32]
    deltas = [0.00, 0.01, 0.05, 0.12, 0.19, 0.20]    # Δ_final = 0.20 (values off the 50/90% boundary)
    et = emergence_times(steps, deltas, eps=0.02)
    assert et["delta_final"] == 0.20
    assert et["t_onset"] == 4     # first Δ ≥ 0.02 (0.05 at step 4)
    assert et["t50"] == 8         # first Δ ≥ 0.10 (0.12 at step 8)
    assert et["t90"] == 16        # first Δ ≥ 0.18 (0.19 at step 16)


def test_emergence_times_never_onset():
    et = emergence_times([1, 2, 3], [0.0, 0.005, 0.01], eps=0.02)
    assert et["t_onset"] is None  # never crosses ε


def test_auec_ranks_faster_acquisition_higher():
    steps = [1, 2, 4, 8]
    fast = [0.2, 0.2, 0.2, 0.2]   # decodable early
    slow = [0.0, 0.0, 0.0, 0.2]   # only at the end; same Δ_final
    assert auec(steps, fast) > auec(steps, slow)


def test_layer_star_tracks_migration_deeper():
    side = {"checkpoints": [
        {"step": 1, "per_layer": [{"layer": 0, "rel_depth": 0.0, "delta": 0.10},
                                  {"layer": 1, "rel_depth": 1.0, "delta": 0.05}]},
        {"step": 8, "per_layer": [{"layer": 0, "rel_depth": 0.0, "delta": 0.05},
                                  {"layer": 1, "rel_depth": 1.0, "delta": 0.20}]},
    ]}
    assert layer_star_over_time(side) == [(1, 0.0), (8, 1.0)]   # peak migrates shallow→deep
