"""Tests for P3.2 rank-reversal analysis (synthetic fixtures; pure stdlib, CPU).

Guards the load-bearing logic that produced two bugs this session: the iso-FLOP interpolation,
the crossover detection, and the reversal@budget classification. No data/cluster needed.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rank_reversal import interp, analyze  # noqa: E402


def test_interp_clamps_and_interpolates():
    c = [(0.0, 0.0), (1.0, 2.0), (2.0, 2.0)]
    assert interp(c, -5.0) == 0.0          # clamp below
    assert interp(c, 9.0) == 2.0           # clamp above
    assert abs(interp(c, 0.5) - 1.0) < 1e-9
    assert abs(interp(c, 1.5) - 2.0) < 1e-9


def _curves(score, scale="S", seeds=("0", "1", "2")):
    """Build (arch,scale,seed) -> [(logC, score)] over logC in [0,10] from score(arch, logC)."""
    grid = [i * (10.0 / 40) for i in range(41)]
    return {(arch, scale, s): [(x, score(arch, x)) for x in grid]
            for arch in ("gpt2", "mamba") for s in seeds}


def test_reversal_detected_with_correct_crossover():
    # mamba ahead early, gpt2 overtakes: Δ = gpt2 - mamba = c - (5 + 0.1c) = 0.9c - 5
    # zero-crossing at c = 5/0.9 ≈ 5.556  ->  C×/C_f = exp(5.556 - 10) ≈ 0.0118
    d = _curves(lambda a, c: c if a == "gpt2" else 5.0 + 0.1 * c)
    r = analyze(d, "S", n_boot=300, rng=random.Random(0))
    assert r is not None
    assert r["reversal"] == 1
    assert r["d_early"] < 0 < r["d_final"]          # mamba early, gpt2 converged
    assert r["p_reversal"] > 0.99                    # identical seeds -> unanimous
    expected = math.exp(5.0 / 0.9 - 10.0)
    assert 0.7 * expected < r["cross_over_frac"] < 1.4 * expected   # grid-resolution slack


def test_no_reversal_when_gpt2_monotone_ahead():
    d = _curves(lambda a, c: (c + 1.0) if a == "gpt2" else 0.0)   # Δ = c+1 > 0 everywhere
    r = analyze(d, "S", n_boot=300, rng=random.Random(0))
    assert r["reversal"] == 0
    assert r["p_reversal"] == 0.0
    assert r["d_early"] > 0 and r["d_final"] > 0


def test_requires_two_seeds():
    d = _curves(lambda a, c: c, seeds=("0",))
    assert analyze(d, "S", n_boot=10, rng=random.Random(0)) is None
