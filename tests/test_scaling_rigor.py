"""Statistical hardening of the cross-arch scaling result (bootstrap CIs, gap test)."""

from __future__ import annotations

import numpy as np

from scripts.scaling_rigor import (
    analyze, per_scale, loglog_alpha, bootstrap_alpha,
)

_N = {"XS": 1e7, "S": 3e7, "M": 1e8, "L": 6e8}


def _cells(alpha_by_arch: dict, noise: float = 1e-4, E: float = 1.0, A: float = 5.0):
    """Synthetic per-seed cells: L = E + A*N^-alpha, tiny per-seed noise."""
    rng = np.random.default_rng(123)
    cells = []
    for arch, alpha in alpha_by_arch.items():
        for scale, N in _N.items():
            base = E + A * N ** (-alpha)
            for seed in range(3):
                cells.append({
                    "arch": arch, "scale": scale, "seed": seed,
                    "n_params": N, "total_flops": 6 * N * (12 * N),
                    "val_bpr": base + rng.normal(0, noise),
                })
    return cells


def test_per_scale_groups_and_sorts() -> None:
    rows = per_scale(_cells({"gpt2": 0.04}), "gpt2")
    assert [r["scale"] for r in rows] == ["XS", "S", "M", "L"]   # ascending N
    assert all(len(r["Ls"]) == 3 for r in rows)                  # 3 seeds each
    assert rows[0]["C"] is not None                              # total_flops carried


def test_loglog_alpha_recovers_slope() -> None:
    # Pure power law L = A*N^-alpha (E=0) -> log-log slope is exactly alpha.
    N = np.array(list(_N.values()))
    L = 5.0 * N ** (-0.04)
    assert abs(loglog_alpha(N, L) - 0.04) < 1e-6


def test_steeper_arch_has_higher_alpha_with_separated_cis() -> None:
    cells = _cells({"gpt2": 0.04, "mamba": 0.02})
    res = analyze(cells, B=400, seed=0)
    ag, am = res["alpha"]["gpt2"], res["alpha"]["mamba"]
    assert ag["point"] > am["point"]               # steeper arch -> larger alpha
    assert ag["ci"][0] > am["ci"][1]               # non-overlapping 95% CIs


def test_cross_arch_delta_excludes_zero() -> None:
    res = analyze(_cells({"gpt2": 0.04, "mamba": 0.02}), B=400, seed=0)
    d = res["delta"]
    assert d["ci"][0] > 0                           # d_alpha CI lower bound positive
    assert d["p_gpt2_steeper"] > 0.95


def test_per_scale_gap_positive_everywhere() -> None:
    res = analyze(_cells({"gpt2": 0.04, "mamba": 0.02}), B=400, seed=0)
    assert len(res["gap"]) == 4
    for g in res["gap"]:
        assert g["gap"] > 0                          # mamba (shallower) loses at every scale
        assert g["ci"][0] > 0                         # CI excludes 0
    # gap grows with N (wider at L than XS) for this power-law pair
    assert res["gap"][-1]["gap"] > res["gap"][0]["gap"]


def test_bootstrap_is_deterministic() -> None:
    rows = per_scale(_cells({"gpt2": 0.04}), "gpt2")
    a = bootstrap_alpha(rows, np.random.default_rng(0), B=200)
    b = bootstrap_alpha(rows, np.random.default_rng(0), B=200)
    assert np.allclose(a, b)                          # same seed -> identical draws
