"""
Probe driver: gather residual-stream features for a checkpoint, fit a per-layer
linear probe (validation-selected), and compare against a random-init baseline.

Memory-aware by construction (docs/probing_harness_plan.md §3, §5): features are
gathered one protein at a time and moved to CPU immediately, accumulated per layer;
``max_residues`` caps the set so a layer's matrix stays in RAM. The headline number is
``learned − baseline`` — the trained model's best-layer test metric minus the same
probe on an untrained model of the identical architecture.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from nanoprot.eval.probe.extract import extract_layers, n_probe_points
from nanoprot.eval.probe.labels import (
    ProbeDataset, ProbeProtein, default_tokenizer, tokenize_and_align,
)
from nanoprot.eval.probe.linear import evaluate_probe, fit_linear_probe


def build_random_init(model_cfg, device: str = "cpu"):
    """An untrained model of the same architecture (the probe's random-init baseline)."""
    from nanoprot.models import build_model
    with torch.device("meta"):
        model = build_model(model_cfg)
    model = model.to_empty(device=device)
    model.init_weights()
    model.eval()
    return model


@torch.no_grad()
def gather_features(model, proteins: Sequence[ProbeProtein], tokenizer=None, *,
                    add_bos: bool = True, ignore_index: int = -100,
                    max_residues: Optional[int] = None,
                    device: Optional[str] = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Residual-stream features at every layer for the labelled residues of ``proteins``.

    Streams one protein at a time; each protein's per-layer features are masked to the
    labelled positions and moved to CPU before the next, so GPU memory stays flat.

    Returns ``(X_by_layer, y)`` where ``X_by_layer[k]`` is ``(N, d)`` for probe point
    ``k`` and ``y`` is ``(N,)`` — the same N rows, aligned, across all layers.
    """
    tokenizer = tokenizer or default_tokenizer()
    if device is None:
        device = str(next(model.parameters()).device)
    n_points = n_probe_points(model)
    chunks: List[List[torch.Tensor]] = [[] for _ in range(n_points)]
    ys: List[torch.Tensor] = []
    total = 0
    for p in proteins:
        ids, aligned = tokenize_and_align(
            p.sequence, p.labels, tokenizer, add_bos=add_bos, ignore_index=ignore_index)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        labels = torch.tensor(aligned, dtype=torch.long)          # (T,)
        mask = labels != ignore_index
        if not bool(mask.any()):
            continue
        feats = extract_layers(model, idx)                        # list of (1, T, d)
        for k, f in enumerate(feats):
            chunks[k].append(f[0][mask].float().cpu())            # (n_valid, d) on CPU
        ys.append(labels[mask])
        total += int(mask.sum())
        if max_residues is not None and total >= max_residues:
            break
    if not ys:
        raise ValueError("gathered no labelled residues (empty split or all-ignore labels)")
    X_by_layer = [torch.cat(c, dim=0) for c in chunks]
    return X_by_layer, torch.cat(ys, dim=0)


def _sweep(features: Dict[str, Tuple[List[torch.Tensor], torch.Tensor]],
           n_classes: int, metric: str, fit_kw: dict) -> dict:
    """Fit a probe per layer on train, select the best layer on val, report on test."""
    Xtr, ytr = features["train"]
    Xva, yva = features["val"]
    Xte, yte = features["test"]
    per_layer = []
    for k in range(len(Xtr)):
        probe = fit_linear_probe(Xtr[k], ytr, n_classes, **fit_kw)
        per_layer.append({
            "layer": k,
            "val": evaluate_probe(probe, Xva[k], yva, n_classes),
            "test": evaluate_probe(probe, Xte[k], yte, n_classes),
        })
    best = max(per_layer, key=lambda r: r["val"][metric])          # SELECT ON VAL
    return {"per_layer": per_layer, "best_layer": best["layer"],
            "best_val": best["val"], "best_test": best["test"]}     # REPORT ON TEST


def run_probe(model, baseline_model, dataset: ProbeDataset, tokenizer=None, *,
              metric: str = "macro_f1", max_residues: Optional[int] = None,
              device: Optional[str] = None, fit_kw: Optional[dict] = None) -> dict:
    """Full probe of one checkpoint vs. its random-init baseline over a :class:`ProbeDataset`.

    Returns the trained sweep, the baseline sweep, and ``learned − baseline`` on the
    val-selected best layer's test metric.
    """
    tokenizer = tokenizer or default_tokenizer()
    fit_kw = fit_kw or {}
    n_classes = dataset.n_classes
    ig = dataset.ignore_index

    def features_for(m):
        return {sp: gather_features(m, dataset.split(sp), tokenizer, ignore_index=ig,
                                    max_residues=max_residues, device=device)
                for sp in ("train", "val", "test")}

    learned = _sweep(features_for(model), n_classes, metric, fit_kw)
    baseline = _sweep(features_for(baseline_model), n_classes, metric, fit_kw)
    delta = learned["best_test"][metric] - baseline["best_test"][metric]
    return {
        "concept": dataset.concept, "source": dataset.source, "metric": metric,
        "learned": learned, "baseline": baseline,
        "best_layer": learned["best_layer"],
        "learned_test": learned["best_test"][metric],
        "baseline_test": baseline["best_test"][metric],
        "learned_minus_baseline": delta,
    }
