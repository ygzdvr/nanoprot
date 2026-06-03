"""Scaling-law fits must recover known exponents from synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from scripts.scaling_laws import fit_power_law


def test_pure_power_law_recovers_exponent() -> None:
    N = np.array([1e7, 3e7, 1.5e8, 6.5e8])
    L = 5.0 * N ** (-0.08)                 # pure power law, no irreducible term
    fit = fit_power_law(N, L)
    assert abs(fit["pl_alpha"] - 0.08) < 5e-3
    assert fit["pl_r2"] > 0.999


def test_chinchilla_form_recovers_irreducible_loss() -> None:
    N = np.array([1e7, 3e7, 1.5e8, 6.5e8, 2e9])  # 5 points -> E identifiable
    L = 0.30 + 5.0 * N ** (-0.10)
    fit = fit_power_law(N, L)
    assert "alpha" in fit and "E" in fit
    assert abs(fit["alpha"] - 0.10) < 0.02
    assert abs(fit["E"] - 0.30) < 0.10
    assert fit["r2"] > 0.999


def test_two_points_gives_pure_power_law_only() -> None:
    N = np.array([1e7, 1e8])
    L = 2.0 * N ** (-0.05)
    fit = fit_power_law(N, L)
    assert "pl_alpha" in fit          # 2-param always available
    assert "alpha" not in fit          # 3-param needs >= 3 points
