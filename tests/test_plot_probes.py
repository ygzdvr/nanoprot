"""Probe figures: scaling-transfer aggregation + layer-wise rendering from sidecars."""

from __future__ import annotations

import csv
import json

from scripts.plot_probes import (
    _largest_scale_sidecars, aggregate_scaling, figure, load_results, load_sidecars,
)


def _rows():
    rows = []
    for arch, n_params in [("gpt2", 1e7), ("gpt2", 6e8), ("mamba", 1e7), ("mamba", 6e8)]:
        scale = "S" if n_params < 1e8 else "L"
        for seed in (0, 1):
            rows.append({
                "arch": arch, "scale": scale, "seed": str(seed), "n_params": str(n_params),
                "source": "netsurfp", "metric": "macro_f1",
                "learned_test": "0.7", "baseline_test": "0.4",
                "learned_minus_baseline": "0.30" if arch == "gpt2" else "0.25",
                "best_layer": "3", "n_layers": "5",
            })
    return rows


def _sidecar(arch, scale):
    return {
        "row": {"arch": arch, "scale": scale, "source": "netsurfp"},
        "learned_per_layer": [
            {"layer": 0, "rel_depth": 0.0, "test": {"macro_f1": 0.4, "accuracy": 0.5}},
            {"layer": 1, "rel_depth": 1.0, "test": {"macro_f1": 0.7, "accuracy": 0.8}},
        ],
    }


def test_aggregate_scaling_means_over_seeds() -> None:
    agg = aggregate_scaling(_rows())
    pts = agg[("netsurfp", "gpt2")]            # list of (n_params, scale, mean, std)
    assert [p[1] for p in pts] == ["S", "L"]   # sorted by params
    assert abs(pts[0][2] - 0.30) < 1e-9        # mean of the two seeds
    assert pts[0][3] == 0.0                     # identical seeds -> zero std


def test_largest_scale_sidecar_per_arch() -> None:
    best = _largest_scale_sidecars([_sidecar("gpt2", "S"), _sidecar("gpt2", "L")], "netsurfp")
    assert best["gpt2"]["row"]["scale"] == "L"


def test_figure_writes_png_and_pdf(tmp_path) -> None:
    out = tmp_path / "probes"
    figure(_rows(), [_sidecar("gpt2", "L"), _sidecar("mamba", "L")], out)
    assert out.with_suffix(".png").exists() and out.with_suffix(".pdf").exists()


def test_figure_handles_empty_results(tmp_path) -> None:
    out = tmp_path / "empty"
    figure([], [], out)                         # placeholders, must not crash
    assert out.with_suffix(".png").exists()


def test_load_results_and_sidecars_roundtrip(tmp_path) -> None:
    rows = _rows()
    csv_path = tmp_path / "r.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    assert len(load_results(csv_path)) == len(rows)
    (tmp_path / "s.json").write_text(json.dumps(_sidecar("gpt2", "L")))
    assert len(load_sidecars(tmp_path)) == 1
