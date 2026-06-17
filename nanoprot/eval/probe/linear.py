"""
Linear probes for cross-architecture representation analysis.

Frozen-model probing: fit a standardized linear classifier on residual-stream
features and measure how linearly decodable a per-residue concept is. Two methodology
points from docs/probing_harness_plan.md are baked in here:

- **Standardize** features (z-score) before fitting — layers/archs differ in scale.
- **Select the best layer on VALIDATION, report it on TEST** — sweeping every layer
  and reporting the test max is multiple-comparisons peeking.

The probe is a plain ``nn.Linear`` trained with Adam (convex multinomial logistic
regression), so the only dependency is torch and the fit is deterministic given a
seed. ``learned − baseline`` (baseline = same probe on a random-init model, handled by
the caller) is the quantity that actually means something.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data shaping
# ---------------------------------------------------------------------------

def flatten_residues(feats: torch.Tensor, labels: torch.Tensor,
                     ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B,T,d)`` features + ``(B,T)`` labels -> ``(N,d)`` X, ``(N,)`` y over the valid
    (``label != ignore_index``) positions, in row-major order."""
    feats = feats.reshape(-1, feats.shape[-1])
    labels = labels.reshape(-1)
    mask = labels != ignore_index
    return feats[mask], labels[mask]


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

@dataclass
class LinearProbe:
    """A fitted linear probe with its standardization stats baked in."""

    mean: torch.Tensor      # (1, d)
    std: torch.Tensor       # (1, d)
    weight: torch.Tensor    # (C, d)
    bias: torch.Tensor      # (C,)
    n_classes: int

    def logits(self, X) -> torch.Tensor:
        X = torch.as_tensor(X, dtype=torch.float32, device=self.weight.device)
        Xs = (X - self.mean) / self.std
        return Xs @ self.weight.t() + self.bias

    def predict(self, X) -> torch.Tensor:
        return self.logits(X).argmax(dim=-1)


def fit_linear_probe(X, y, n_classes: int, *, standardize: bool = True,
                     epochs: int = 300, lr: float = 0.05, weight_decay: float = 1e-4,
                     seed: int = 0, device: Optional[str] = None) -> LinearProbe:
    """Fit a standardized multinomial-logistic linear probe. Deterministic given ``seed``."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    d = X.shape[1]
    if standardize:
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    else:
        mean = torch.zeros(1, d, device=device)
        std = torch.ones(1, d, device=device)
    Xs = (X - mean) / std

    torch.manual_seed(seed)                         # deterministic nn.Linear init
    lin = nn.Linear(d, n_classes).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(lin(Xs), y).backward()
        opt.step()
    return LinearProbe(mean.detach(), std.detach(),
                       lin.weight.detach(), lin.bias.detach(), n_classes)


@dataclass
class LinearRegressor:
    """A fitted standardized linear-regression probe (for continuous targets)."""

    mean: torch.Tensor      # (1, d)
    std: torch.Tensor       # (1, d)
    weight: torch.Tensor    # (1, d)
    bias: torch.Tensor      # (1,)

    def predict(self, X) -> torch.Tensor:
        X = torch.as_tensor(X, dtype=torch.float32, device=self.weight.device)
        Xs = (X - self.mean) / self.std
        return (Xs @ self.weight.t() + self.bias).squeeze(-1)


def fit_linear_regression(X, y, *, standardize: bool = True, epochs: int = 300,
                          lr: float = 0.05, weight_decay: float = 1e-4, seed: int = 0,
                          device: Optional[str] = None) -> LinearRegressor:
    """Fit a standardized linear-regression probe (MSE). Deterministic given ``seed``."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    d = X.shape[1]
    if standardize:
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    else:
        mean = torch.zeros(1, d, device=device)
        std = torch.ones(1, d, device=device)
    Xs = (X - mean) / std
    torch.manual_seed(seed)
    lin = nn.Linear(d, 1).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        F.mse_loss(lin(Xs).squeeze(-1), y).backward()
        opt.step()
    return LinearRegressor(mean.detach(), std.detach(), lin.weight.detach(), lin.bias.detach())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy(y_true, y_pred) -> float:
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred, device=y_true.device)
    return float((y_true == y_pred).float().mean()) if y_true.numel() else float("nan")


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    """Unweighted mean per-class F1 (handles class imbalance better than accuracy)."""
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred, device=y_true.device)
    f1s = []
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().float()
        fp = ((y_pred == c) & (y_true != c)).sum().float()
        fn = ((y_pred != c) & (y_true == c)).sum().float()
        denom = 2 * tp + fp + fn
        f1s.append(float(2 * tp / denom) if denom > 0 else 0.0)
    return sum(f1s) / len(f1s)


def evaluate_probe(probe: LinearProbe, X, y, n_classes: int) -> Dict[str, float]:
    y = torch.as_tensor(y, dtype=torch.long)
    pred = probe.predict(X).to(y.device)
    return {"accuracy": accuracy(y, pred), "macro_f1": macro_f1(y, pred, n_classes)}


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination (1 − SS_res/SS_tot)."""
    y_true = torch.as_tensor(y_true, dtype=torch.float32)
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32, device=y_true.device)
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if float(ss_tot) <= 0 or y_true.numel() == 0:
        return float("nan")
    return float(1 - ((y_true - y_pred) ** 2).sum() / ss_tot)


def spearman_corr(y_true, y_pred) -> float:
    """Spearman rank correlation (Pearson correlation of ranks; ties broken by argsort)."""
    y_true = torch.as_tensor(y_true, dtype=torch.float32)
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32, device=y_true.device)
    if y_true.numel() < 2:
        return float("nan")
    rt = y_true.argsort().argsort().float()
    rp = y_pred.argsort().argsort().float()
    rt = rt - rt.mean()
    rp = rp - rp.mean()
    denom = (rt.norm() * rp.norm()).clamp_min(1e-12)      # Pearson r = centered cosine
    return float((rt * rp).sum() / denom)


def evaluate_regression(reg: LinearRegressor, X, y) -> Dict[str, float]:
    y = torch.as_tensor(y, dtype=torch.float32)
    pred = reg.predict(X).to(y.device)
    return {"r2": r2_score(y, pred), "spearman": spearman_corr(y, pred)}


def valid_mask(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Positions to keep: not NaN (regression ignore) AND != ignore_index (classification
    ignore). Works for both int-label and float-target tensors."""
    return (labels == labels) & (labels != ignore_index)


# ---------------------------------------------------------------------------
# Layer sweep with validation-selected best layer
# ---------------------------------------------------------------------------

def probe_layer_sweep(layer_X: Sequence, y, split: Dict[str, torch.Tensor],
                      n_classes: int, *, metric: str = "macro_f1", **fit_kw) -> dict:
    """Fit a probe per layer, then pick the best layer ON VALIDATION and report it ON TEST.

    Parameters
    ----------
    layer_X : list of ``(N, d_l)`` feature arrays, one per probe point, all aligned to ``y``.
    y       : ``(N,)`` integer labels.
    split   : ``{"train", "val", "test"}`` -> index or boolean-mask tensors over the N rows.
    metric  : which metric to maximize on validation when selecting the best layer.
    **fit_kw: forwarded to :func:`fit_linear_probe` (epochs, lr, seed, device, ...).

    Returns ``{"per_layer", "best_layer", "best_val", "best_test"}``.
    """
    tr, va, te = split["train"], split["val"], split["test"]
    y = torch.as_tensor(y, dtype=torch.long)
    per_layer: List[dict] = []
    for layer_idx, X in enumerate(layer_X):
        X = torch.as_tensor(X, dtype=torch.float32)
        probe = fit_linear_probe(X[tr], y[tr], n_classes, **fit_kw)
        per_layer.append({
            "layer": layer_idx,
            "val": evaluate_probe(probe, X[va], y[va], n_classes),
            "test": evaluate_probe(probe, X[te], y[te], n_classes),
        })
    best = max(per_layer, key=lambda r: r["val"][metric])     # SELECT ON VAL
    return {
        "per_layer": per_layer,
        "best_layer": best["layer"],
        "best_val": best["val"],
        "best_test": best["test"],                            # REPORT ON TEST
    }
