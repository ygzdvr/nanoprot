"""Integration smoke test for :func:`nanoprot.training.loop.train`.

Runs a few optimizer steps end-to-end on CPU using a tiny model, an injected
synthetic data loader, and a temp checkpoint dir. This test exists so that
signature regressions in ``save_checkpoint`` or the data-loader contract get
caught without needing a GPU or a real UniRef50 dataset.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from nanoprot.config import NanoprotConfig  # noqa: E402
from nanoprot.training.loop import lr_multiplier, train  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config(tmp_path: Path) -> NanoprotConfig:
    """A config small enough to actually run on CPU in a few seconds.

    We pick ``total_batch_size == device_batch_size * max_seq_len * world_size``
    so ``grad_accum_steps == 1`` and each optimizer step is one fwd+bwd.
    """
    DEVICE_BATCH = 2
    SEQ_LEN = 16
    TOTAL_BATCH = DEVICE_BATCH * SEQ_LEN  # world_size=1 in pytest, so grad_accum=1
    N_ITERS = 3
    return NanoprotConfig(
        name="loop-smoke",
        seed=0,
        model={
            "depth": 2,
            "max_seq_len": SEQ_LEN,
            "vocab_size": 64,
            "head_dim": 32,
            "window_pattern": "L",
        },
        data={"shard_dir": str(tmp_path / "data")},
        optimizer={
            # Small LRs so a tiny model doesn't explode in 3 steps.
            "matrix_lr": 0.001,
            "embedding_lr": 0.01,
            "unembedding_lr": 0.001,
            "scalar_lr": 0.01,
            "weight_decay": 0.0,
        },
        training={
            "total_residues": N_ITERS * TOTAL_BATCH,
            "total_batch_size": TOTAL_BATCH,
            "device_batch_size": DEVICE_BATCH,
            "warmup_steps": 1,
            "warmdown_ratio": 0.5,
            "final_lr_frac": 0.5,
            "precision": "fp32",
            "flash_attention": False,
        },
        eval={"eval_every": -1},
        logging={"wandb_mode": "disabled"},
        checkpointing={"output_dir": str(tmp_path / "ckpt"), "save_every": -1},
    )


def _synthetic_loader(vocab_size: int, batch: int, seq_len: int, device: str = "cpu"):
    """Infinite iterator of ``(idx, targets)`` pairs of random tokens."""
    g = torch.Generator(device=device).manual_seed(7)
    while True:
        idx = torch.randint(0, vocab_size, (batch, seq_len), generator=g, device=device)
        tgt = torch.randint(0, vocab_size, (batch, seq_len), generator=g, device=device)
        yield idx, tgt


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

class TestLrSchedule:
    def test_warmup_starts_below_one(self) -> None:
        assert lr_multiplier(0, 100, warmup=10, warmdown_ratio=0.5, final_frac=0.1) < 1.0

    def test_warmup_finishes_at_one(self) -> None:
        # After warmup, before warmdown -> flat at 1.0
        assert lr_multiplier(20, 100, warmup=10, warmdown_ratio=0.5, final_frac=0.1) == 1.0

    def test_warmdown_ends_at_final_frac(self) -> None:
        # At the very last step the multiplier should equal final_frac.
        mult = lr_multiplier(99, 100, warmup=10, warmdown_ratio=0.5, final_frac=0.1)
        assert abs(mult - 0.1) < 0.05

    def test_warmdown_is_monotonic(self) -> None:
        steps = list(range(50, 100))
        vals = [lr_multiplier(s, 100, warmup=10, warmdown_ratio=0.5, final_frac=0.1) for s in steps]
        # non-increasing
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


# ---------------------------------------------------------------------------
# End-to-end loop
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTrainLoopEndToEnd:
    """Actually runs the loop on CPU. Catches signature regressions in
    save_checkpoint / dataloader / optimizer that the unit tests cannot."""

    def test_train_runs_and_saves_checkpoint(self, tmp_path: Path) -> None:
        cfg = _tiny_config(tmp_path)
        loader = _synthetic_loader(
            vocab_size=cfg.model.vocab_size,
            batch=cfg.training.device_batch_size,
            seq_len=cfg.model.max_seq_len,
            device="cpu",
        )

        state = train(cfg, device_type="cpu", train_loader=loader)

        # Final state sanity
        n_iter = cfg.total_residues() // cfg.training.total_batch_size
        assert state.step == n_iter - 1
        assert state.smooth_loss > 0
        # Checkpoint files were actually written
        ckpt_dir = Path(cfg.checkpointing.output_dir)
        assert ckpt_dir.is_dir()
        model_files = list(ckpt_dir.glob("model_*.pt"))
        meta_files = list(ckpt_dir.glob("meta_*.json"))
        assert len(model_files) >= 1, f"no model checkpoints in {ckpt_dir}"
        assert len(meta_files) >= 1, f"no meta files in {ckpt_dir}"

    def test_train_loss_finite(self, tmp_path: Path) -> None:
        cfg = _tiny_config(tmp_path)
        loader = _synthetic_loader(
            vocab_size=cfg.model.vocab_size,
            batch=cfg.training.device_batch_size,
            seq_len=cfg.model.max_seq_len,
            device="cpu",
        )
        state = train(cfg, device_type="cpu", train_loader=loader)
        assert state.smooth_loss == state.smooth_loss  # not NaN
        assert state.smooth_loss < 1e3, f"loss exploded: {state.smooth_loss}"

    def test_eval_loop_runs_and_records_bpr(self, tmp_path: Path) -> None:
        """With a val_loader supplied + eval_every<=num_iterations, the loop
        should run at least one eval pass and populate state.last_val_bpr."""
        cfg = _tiny_config(tmp_path)
        # Override eval cadence so we actually trigger one eval pass.
        cfg.eval.eval_every = 1                   # eval every step (after step 0)
        cfg.eval.eval_tokens = 32                 # 1 batch worth
        train_loader = _synthetic_loader(
            cfg.model.vocab_size,
            cfg.training.device_batch_size,
            cfg.model.max_seq_len,
        )
        val_loader = _synthetic_loader(
            cfg.model.vocab_size,
            cfg.training.device_batch_size,
            cfg.model.max_seq_len,
        )
        state = train(
            cfg, device_type="cpu",
            train_loader=train_loader, val_loader=val_loader,
        )
        assert state.last_val_loss is not None, "eval loop did not run"
        assert state.last_val_bpr is not None
        assert math.isfinite(state.last_val_loss)
        assert state.best_val_bpr <= state.last_val_bpr  # best is min over passes

    def test_train_works_for_esm2_mlm_objective(self, tmp_path: Path) -> None:
        """Same end-to-end loop, but with arch=esm2 and objective=mlm."""
        DEVICE_BATCH = 2
        SEQ_LEN = 16
        TOTAL_BATCH = DEVICE_BATCH * SEQ_LEN
        N_ITERS = 2
        cfg = NanoprotConfig(
            name="esm2-mlm-smoke",
            seed=0,
            model={
                "arch": "esm2",
                "depth": 2,
                "max_seq_len": SEQ_LEN,
                "vocab_size": 33,
                "head_dim": 16,
            },
            tokenizer={"name": "esm2"},
            data={"shard_dir": str(tmp_path / "data")},
            optimizer={
                "matrix_lr": 0.001,
                "embedding_lr": 0.01,
                "unembedding_lr": 0.001,
                "scalar_lr": 0.01,
                "weight_decay": 0.0,
            },
            training={
                "objective": "mlm",
                "mlm_probability": 0.15,
                "total_residues": N_ITERS * TOTAL_BATCH,
                "total_batch_size": TOTAL_BATCH,
                "device_batch_size": DEVICE_BATCH,
                "warmup_steps": 1,
                "warmdown_ratio": 0.5,
                "final_lr_frac": 0.5,
                "precision": "fp32",
                "flash_attention": False,
            },
            eval={"eval_every": -1},
            logging={"wandb_mode": "disabled"},
            checkpointing={"output_dir": str(tmp_path / "ckpt"), "save_every": -1},
        )
        # MLM-style synthetic loader: random ids, random masked targets.
        def _mlm_synth():
            g = torch.Generator().manual_seed(11)
            V = cfg.model.vocab_size
            B, T = cfg.training.device_batch_size, cfg.model.max_seq_len
            while True:
                idx = torch.randint(0, V, (B, T), generator=g)
                tgt = torch.full((B, T), -100, dtype=torch.long)
                # Pick 15% positions to be supervised.
                mask = torch.bernoulli(torch.full((B, T), 0.15), generator=g).bool()
                tgt[mask] = idx[mask]
                yield idx, tgt

        state = train(cfg, device_type="cpu", train_loader=_mlm_synth())
        assert math.isfinite(state.smooth_loss)
        # MLM loss at start should be ~ ln(V); ours has ~ln(33) ≈ 3.5.
        assert state.smooth_loss < 5.0
