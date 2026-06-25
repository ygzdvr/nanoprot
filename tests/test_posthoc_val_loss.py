"""Tests for the post-hoc val-loss eval's pure helpers (the GPU path is smoke-tested live)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from scripts.posthoc_val_loss_eval import list_checkpoint_steps, val_loss_on_batches  # noqa: E402


def test_list_checkpoint_steps(tmp_path):
    for s in [1, 9, 826, 5]:
        (tmp_path / f"model_{s:06d}.pt").write_text("x")
    (tmp_path / "meta_000001.json").write_text("{}")     # ignored
    assert list_checkpoint_steps(tmp_path) == [1, 5, 9, 826]


class _DummyModel:
    training = False

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, idx, targets=None):
        return idx.sum()                                  # deterministic "loss"


def test_val_loss_on_batches_is_mean_over_fixed_batches():
    batches = [(torch.tensor([2.0]), None), (torch.tensor([4.0]), None)]
    assert abs(val_loss_on_batches(_DummyModel(), batches) - 3.0) < 1e-9
