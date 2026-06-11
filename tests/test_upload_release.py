"""Upload plan: ship the model + card, NEVER the optimizer state; seed layout."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.upload_release import plan


def _cell(root: Path, arch: str, scale: str, seed: int, step: int = 5, done: bool = True) -> None:
    d = root / f"nanoprot-{arch}-{scale}-s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"model_{step:06d}.pt").write_bytes(b"w")
    (d / f"meta_{step:06d}.json").write_text(json.dumps(
        {"step": step, "num_iterations": step if done else step + 1}))
    # optimizer shards — must NEVER be uploaded
    for r in range(4):
        (d / f"optim_{step:06d}_rank{r}.pt").write_bytes(b"opt")
    (d / "config.yaml").write_text("x: 1")
    (d / "provenance.json").write_text("{}")
    (d / "README.md").write_text("# card")


def test_plan_excludes_optimizer_state(tmp_path: Path) -> None:
    _cell(tmp_path, "esm2", "S", 0)
    repos = plan(tmp_path, "yagizdevre", only=None)
    assert len(repos) == 1
    repo, files = repos[0]
    assert repo == "yagizdevre/nanoprot-esm2-S"
    repo_paths = [p for _, p in files]
    assert not any("optim_" in p for p in repo_paths), "optimizer state must not be uploaded"
    assert "README.md" in repo_paths and "config.yaml" in repo_paths
    assert any(p.startswith("model_") for p in repo_paths)


def test_plan_seed_layout(tmp_path: Path) -> None:
    for s in (0, 1, 2):
        _cell(tmp_path, "esm2", "L", s)
    (repo, files) = plan(tmp_path, "yagizdevre", only=None)[0]
    paths = [p for _, p in files]
    # seed 0 at root, seeds 1/2 under seedN/
    assert any(p == "model_000005.pt" for p in paths)         # headline at root
    assert any(p == "seed1/model_000005.pt" for p in paths)
    assert any(p == "seed2/model_000005.pt" for p in paths)
    # only the headline ships config/provenance/README (not duplicated per seed)
    assert sum(p.endswith("README.md") for p in paths) == 1


def test_plan_skips_partial_cells(tmp_path: Path) -> None:
    _cell(tmp_path, "gpt2", "M", 0, done=False)  # not finished
    assert plan(tmp_path, "yagizdevre", only=None) == []
