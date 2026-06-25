"""Smoke test for plot_emergence on synthetic trajectory data (no GPU, no real run).

Validates that the four figure families read the run_trajectory_probes / analyze_emergence
/ posthoc_val_loss_eval schemas and render without error, since there is no real
trajectory data yet.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.plot_emergence as pe  # noqa: E402
from scripts.analyze_emergence import load_curves  # noqa: E402

STEPS = [1, 4, 16, 64, 256]


def _synthesize(results_dir: Path) -> None:
    results_dir.mkdir(parents=True)
    # probe deltas (best-by-val per checkpoint): gpt2 > mamba, both rising
    rows = []
    for arch, base in [("gpt2", 0.0), ("mamba", -0.02), ("esm2", -0.04)]:
        for seed in (0, 1):
            for k, st in enumerate(STEPS):
                delta = base + 0.05 * k + 0.005 * seed
                rows.append(dict(arch=arch, scale="S", seed=seed, step=st,
                                 train_residues=st * 524288, train_flops=st * 1e12,
                                 concept="ss3", source="netsurfp", metric="macro_f1",
                                 best_layer=2, learned_test=0.5 + delta, delta=round(delta, 6)))
    with open(results_dir / "ss3_netsurfp.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # full layer x checkpoint sidecar (heatmap)
    side = {"arch": "gpt2", "scale": "S", "seed": 0, "concept": "ss3", "source": "netsurfp",
            "metric": "macro_f1", "class_names": ["helix", "strand", "coil"],
            "checkpoints": [{"step": st, "per_layer": [{"layer": l, "rel_depth": l / 3,
                                                        "delta": 0.02 * k + 0.01 * l}
                                                       for l in range(4)]}
                            for k, st in enumerate(STEPS)]}
    (results_dir / "traj_gpt2-S-s0_netsurfp_ss3.json").write_text(json.dumps(side))

    # post-hoc val_loss.csv (exact per-checkpoint val_loss; falls over training)
    with open(results_dir / "val_loss.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arch", "scale", "seed", "step", "val_loss", "val_bpr", "train_residues"])
        for arch in ("gpt2", "mamba", "esm2"):
            for seed in (0, 1):
                for k, st in enumerate(STEPS):
                    vl = 5.0 - 0.4 * k
                    w.writerow([arch, "S", seed, st, vl, vl / 0.693, st * 524288])


def test_plot_emergence_smoke(tmp_path):
    results = tmp_path / "trajectory_results"
    _synthesize(results)

    curves, metric = load_curves(results)
    assert curves
    assert any(k[2] == "esm2" for k in curves)           # esm2 present -> exercises the AR-only filter
    valloss = pe.load_val_loss_csv(results)
    assert valloss                                       # matched-loss join is populated

    out = tmp_path / "fig" / "emergence"
    pe.fig_vs_step(curves, Path(f"{out}_vs_step"))
    pe.fig_vs_valloss(curves, valloss, Path(f"{out}_vs_valloss"))                  # aggregated main
    pe.fig_vs_valloss_seeds(curves, valloss, Path(f"{out}_vs_valloss_seeds"))      # per-seed supp.
    pe.fig_layer_heatmaps(pe.load_sidecars(results), Path(f"{out}_heatmaps"))
    pe.fig_emergence_times(curves, metric, Path(f"{out}_times"))
    for suffix in ("_vs_step", "_vs_valloss", "_vs_valloss_seeds", "_heatmaps", "_times"):
        assert (tmp_path / "fig" / f"emergence{suffix}.png").exists(), suffix
