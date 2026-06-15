"""Probe driver: per-layer feature gathering + the learned-vs-baseline sweep."""

from __future__ import annotations

import pytest
import torch

from nanoprot.config import Gpt2ModelConfig
from nanoprot.eval.probe.extract import n_probe_points
from nanoprot.eval.probe.labels import ProbeDataset, ProbeProtein, default_tokenizer
from nanoprot.eval.probe.run import build_random_init, gather_features, run_probe


def _model():
    from nanoprot.models import build_model
    cfg = Gpt2ModelConfig(depth=2, head_dim=32, vocab_size=33,
                          max_seq_len=64, window_pattern="L").derive()
    with torch.device("meta"):
        m = build_model(cfg)
    m = m.to_empty(device="cpu")
    m.init_weights()
    m.eval()
    return m, cfg


def _dataset():
    seqs = ["MKTAYIAK", "GVSERTID", "PKQNFYMH", "WCLAGVSE", "RTIDPKQN", "FYMHWCLA"]
    splits = ["train", "train", "train", "val", "test", "test"]
    proteins = [
        ProbeProtein(f"p{i}", s, [j % 3 for j in range(len(s))], sp)   # labels cycle 0,1,2
        for i, (s, sp) in enumerate(zip(seqs, splits))
    ]
    return ProbeDataset(concept="ss3", source="synthetic", n_classes=3,
                        class_names=["H", "E", "C"], proteins=proteins)


@pytest.mark.slow
def test_gather_features_shapes_and_cpu() -> None:
    m, cfg = _model()
    ds = _dataset()
    X, y = gather_features(m, ds.split("train"), default_tokenizer(), device="cpu")
    assert len(X) == n_probe_points(m) == cfg.depth + 1
    N = int(y.shape[0])
    assert N == sum(len(p.sequence) for p in ds.split("train"))     # every residue labelled
    for xl in X:
        assert xl.shape == (N, cfg.d_model)                          # same N across layers
        assert xl.device.type == "cpu"                               # moved off-device


@pytest.mark.slow
def test_gather_features_respects_residue_cap() -> None:
    m, _ = _model()
    X, y = gather_features(m, _dataset().split("train"), device="cpu", max_residues=5)
    # stops after the first protein crosses the cap (whole-protein granularity)
    assert 5 <= int(y.shape[0]) <= 5 + 8


@pytest.mark.slow
def test_gather_skips_fully_unlabelled_protein() -> None:
    m, _ = _model()
    proteins = [
        ProbeProtein("masked", "MKTA", [-100, -100, -100, -100], "train"),
        ProbeProtein("ok", "GVSE", [0, 1, 2, 0], "train"),
    ]
    _, y = gather_features(m, proteins, device="cpu")
    assert int(y.shape[0]) == 4                                      # only "ok" contributes


@pytest.mark.slow
def test_run_probe_end_to_end_structure() -> None:
    m, cfg = _model()
    # _model() returns the bare model config; load_pretrained (in the CLI) returns a
    # NanoprotConfig, so the CLI passes cfg.model — here cfg already IS the model config.
    baseline = build_random_init(cfg, device="cpu")
    out = run_probe(m, baseline, _dataset(), device="cpu",
                    fit_kw={"epochs": 50, "seed": 0, "device": "cpu"})
    assert out["concept"] == "ss3" and out["source"] == "synthetic"
    assert 0 <= out["best_layer"] <= cfg.depth
    assert 0.0 <= out["learned_test"] <= 1.0
    assert 0.0 <= out["baseline_test"] <= 1.0
    assert out["learned_minus_baseline"] == pytest.approx(
        out["learned_test"] - out["baseline_test"])
    # both sweeps probed every layer; best layer chosen on val
    assert len(out["learned"]["per_layer"]) == cfg.depth + 1
    assert len(out["baseline"]["per_layer"]) == cfg.depth + 1
