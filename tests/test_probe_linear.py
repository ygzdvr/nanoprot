"""Linear probe: standardized fit, metrics, and the validation-selected best layer."""

from __future__ import annotations

import torch

from nanoprot.eval.probe.linear import (
    accuracy, evaluate_probe, evaluate_regression, fit_linear_probe,
    fit_linear_regression, flatten_residues, macro_f1, probe_layer_sweep,
    r2_score, spearman_corr, valid_mask,
)


def _blobs(n_per: int, d: int, centers, noise: float = 0.5, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    Xs, ys = [], []
    for c, center in enumerate(centers):
        Xs.append(center + noise * torch.randn(n_per, d, generator=g))
        ys.append(torch.full((n_per,), c))
    return torch.cat(Xs), torch.cat(ys)


def test_linear_probe_learns_separable_classes() -> None:
    d = 8
    centers = [torch.zeros(d), torch.ones(d) * 4, -torch.ones(d) * 4]
    X, y = _blobs(100, d, centers, seed=1)
    probe = fit_linear_probe(X, y, 3, epochs=300, seed=0, device="cpu")
    assert evaluate_probe(probe, X, y, 3)["accuracy"] > 0.95


def test_standardization_handles_large_feature_scale() -> None:
    d = 6
    X, y = _blobs(80, d, [torch.zeros(d), torch.ones(d) * 4], seed=2)
    X = X * 1000.0   # blow up the scale; z-scoring inside fit must absorb it
    probe = fit_linear_probe(X, y, 2, epochs=300, seed=0, device="cpu")
    assert evaluate_probe(probe, X, y, 2)["accuracy"] > 0.95


def test_macro_f1_matches_hand_computation() -> None:
    y_true = torch.tensor([0, 0, 1, 1, 2, 2])
    y_pred = torch.tensor([0, 0, 1, 1, 2, 0])     # one class-2 sample misread as 0
    # class0 F1 = 2*2/(2*2+1+0)=0.8 ; class1 = 1.0 ; class2 = 2*1/(2*1+0+1)=2/3
    expected = (0.8 + 1.0 + 2 / 3) / 3
    assert abs(macro_f1(y_true, y_pred, 3) - expected) < 1e-6
    assert abs(accuracy(y_true, y_pred) - 5 / 6) < 1e-6


def test_layer_sweep_selects_best_layer_on_validation() -> None:
    d, N = 6, 240
    g = torch.Generator().manual_seed(3)
    y = torch.randint(0, 2, (N,), generator=g)
    noise0 = torch.randn(N, d, generator=g)                          # uninformative
    sep = (y.unsqueeze(1).float() * 6 - 3) + 0.3 * torch.randn(N, d, generator=g)  # separable
    noise2 = torch.randn(N, d, generator=g)                          # uninformative
    perm = torch.randperm(N, generator=g)
    split = {"train": perm[:120], "val": perm[120:180], "test": perm[180:]}

    out = probe_layer_sweep([noise0, sep, noise2], y, split, 2,
                            epochs=200, seed=0, device="cpu")
    assert out["best_layer"] == 1                       # the separable layer, chosen on val
    assert out["best_test"]["accuracy"] > 0.9           # and it generalizes on test
    assert len(out["per_layer"]) == 3
    # the noise layers should be near chance on test
    noise_layers = [r for r in out["per_layer"] if r["layer"] != 1]
    assert all(r["test"]["accuracy"] < 0.75 for r in noise_layers)


def test_linear_regression_recovers_linear_target() -> None:
    d, n = 8, 200
    g = torch.Generator().manual_seed(5)
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, generator=g)
    y = X @ w + 0.05 * torch.randn(n, generator=g)        # linear target + small noise
    reg = fit_linear_regression(X, y, epochs=500, seed=0, device="cpu")
    m = evaluate_regression(reg, X, y)
    assert m["r2"] > 0.95 and m["spearman"] > 0.95


def test_r2_and_spearman_edge_cases() -> None:
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert abs(r2_score(y, y) - 1.0) < 1e-6
    assert abs(spearman_corr(y, y) - 1.0) < 1e-6
    assert spearman_corr(y, -y) < -0.99                   # reversed ranks -> -1


def test_valid_mask_handles_nan_and_ignore() -> None:
    labels = torch.tensor([0.5, float("nan"), -100.0, 0.7])   # value / NaN / -100 / value
    assert valid_mask(labels, ignore_index=-100).tolist() == [True, False, False, True]


def test_flatten_residues_drops_ignore_positions() -> None:
    feats = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    labels = torch.tensor([[0, -100, 1], [2, 1, -100]])
    X, y = flatten_residues(feats, labels, ignore_index=-100)
    assert X.shape == (4, 4)                  # 4 of 6 positions are valid
    assert y.tolist() == [0, 1, 2, 1]
    # row-major order preserved: first valid row is feats[0,0]
    assert torch.equal(X[0], feats[0, 0])
