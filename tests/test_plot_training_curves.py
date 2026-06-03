"""Trajectory parsing: pull (step, val_bpr) per cell from array stdout logs."""

from __future__ import annotations

from pathlib import Path

from scripts.plot_training_curves import parse_trajectories

_LOG = """\
Loaded config: nanoprot-esm2-L-s0  (/x/configs/release/nanoprot-esm2-L-s0.yaml)
Starting training: 14857 iterations on 4 ranks
step      0/14857 | loss 3.66 | tok/s 2.6e+04
  [eval] step 250: val_loss=2.6031 val_bpr=3.7554 (best=3.7554)
  [eval] step 500: val_loss=2.5579 val_bpr=3.6902 (best=3.6902)
  [final eval] val_loss=2.23 val_bpr=3.2232 (best=3.2232)
"""


def test_parses_cell_and_trajectory(tmp_path: Path) -> None:
    (tmp_path / "9_0.nanoprot-release.out").write_text(_LOG)
    traj = parse_trajectories(tmp_path, "*.nanoprot-release.out")
    assert "nanoprot-esm2-L-s0" in traj
    pts = traj["nanoprot-esm2-L-s0"]
    assert (250, 3.7554) in pts and (500, 3.6902) in pts
    assert pts == sorted(pts)  # returned in step order


def test_longest_trajectory_wins_on_resubmit(tmp_path: Path) -> None:
    # same cell logged twice (e.g. a resubmit); keep the longer trajectory.
    (tmp_path / "a.nanoprot-release.out").write_text(_LOG)
    short = _LOG.replace("  [eval] step 500: val_loss=2.5579 val_bpr=3.6902 (best=3.6902)\n", "")
    (tmp_path / "b.nanoprot-release.out").write_text(short)
    traj = parse_trajectories(tmp_path, "*.nanoprot-release.out")
    assert len(traj["nanoprot-esm2-L-s0"]) == 2  # the 2-point version, not the 1-point


def test_ignores_logs_without_a_loaded_config(tmp_path: Path) -> None:
    (tmp_path / "noise.out").write_text("just some unrelated stderr\n")
    assert parse_trajectories(tmp_path, "*.out") == {}
