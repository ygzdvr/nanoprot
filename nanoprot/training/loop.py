"""
nanoprot.training.loop — the end-to-end training loop.

Takes a fully-derived :class:`nanoprot.config.NanoprotConfig` and runs a
training job: builds the model, sets up the optimizer, opens the train +
val data loaders (objective-aware), drives the schedule, runs periodic
evaluation, and writes checkpoints. The same function works under both
single-process and torchrun-driven DDP launches.

For tests, callers may supply ``train_loader=`` and/or ``val_loader=``
directly (iterators yielding ``(idx, targets)`` tensors on ``device``);
when omitted, the real loaders are constructed via
:func:`nanoprot.data.builder.build_data_loader` from the config.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch

from nanoprot.config import NanoprotConfig
from nanoprot.data.builder import build_data_loader
from nanoprot.models import build_model
from nanoprot.runtime import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    print0,
)
from nanoprot.training.checkpoint import save_checkpoint


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def lr_multiplier(
    step: int,
    num_iterations: int,
    warmup: int,
    warmdown_ratio: float,
    final_frac: float,
) -> float:
    """Trapezoidal LR schedule: warmup -> flat -> linear warmdown to ``final_frac``."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    warmdown_start = int(num_iterations * (1.0 - warmdown_ratio))
    if step < warmdown_start:
        return 1.0
    frac_done = (step - warmdown_start) / max(num_iterations - warmdown_start, 1)
    return 1.0 - frac_done * (1.0 - final_frac)


# ---------------------------------------------------------------------------
# Public state object
# ---------------------------------------------------------------------------

@dataclass
class TrainState:
    """End-of-training state returned by :func:`train`."""

    step: int = 0
    smooth_loss: float = 0.0
    last_val_loss: Optional[float] = None
    last_val_bpr: Optional[float] = None
    best_val_bpr: float = float("inf")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    num_batches: int,
) -> float:
    """Run ``num_batches`` validation steps and return mean loss."""
    if num_batches <= 0:
        return float("nan")
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(num_batches):
        idx, targets = next(val_loader)
        loss = model(idx, targets=targets)
        losses.append(float(loss.detach()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def train(
    cfg: NanoprotConfig,
    *,
    device_type: Optional[str] = None,
    train_loader: Optional[Iterator[Tuple[torch.Tensor, torch.Tensor]]] = None,
    val_loader: Optional[Iterator[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> TrainState:
    """Run a nanoprot training job described by ``cfg``.

    Parameters
    ----------
    cfg : NanoprotConfig
        Validated, fully-derived nanoprot config. The simplest way to obtain
        one is :func:`nanoprot.config.load_config`.
    device_type : str, optional
        ``cuda``, ``mps``, or ``cpu``. If ``None``, autodetected.
    train_loader, val_loader : iterator, optional
        Iterators yielding ``(idx, targets)`` tensors already on the training
        device. When ``None``, the real loaders are constructed via the
        data-builder dispatch (objective- and tokenizer-aware). Used by
        tests to inject synthetic data without touching the disk.

    Returns
    -------
    TrainState
        Final training state (step, smoothed loss, latest val loss/bpr).
    """
    # ---- Device + DDP init -------------------------------------------------
    if device_type is None:
        device_type = autodetect_device_type()
    is_ddp, rank, local_rank, world_size, device = compute_init(device_type=device_type)
    master = rank == 0

    # ---- Seed --------------------------------------------------------------
    torch.manual_seed(cfg.seed)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    # ---- Build model on meta device, then materialise ---------------------
    print0(
        f"Building model: arch={cfg.model.arch}, depth={cfg.model.depth}, "
        f"d_model={cfg.model.d_model}, n_heads={cfg.model.n_heads}, "
        f"objective={cfg.training.objective}"
    )
    print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
    with torch.device("meta"):
        model = build_model(cfg.model)
    model = model.to_empty(device=device)
    model.init_weights()

    # Parameter counts (real numbers, after instantiation)
    param_counts = model.num_scaling_params()
    print0("Parameter counts:")
    for k, v in param_counts.items():
        print0(f"  {k:24s}: {v:,}")
    n_params = param_counts["total"]
    flops_per_token = model.estimate_flops()
    print0(f"Estimated FLOPs per token: {flops_per_token:e}")

    # ---- Determine the iteration count -------------------------------------
    total_residues = cfg.training.total_residues
    if total_residues is None:
        total_residues = int(cfg.training.param_data_ratio * n_params)
        print0(
            f"Auto-derived total residues "
            f"(Chinchilla, ratio={cfg.training.param_data_ratio}): {total_residues:,}"
        )
    else:
        total_residues = int(total_residues)
    num_iterations = total_residues // cfg.training.total_batch_size
    print0(f"Number of optimization iterations: {num_iterations:,}")

    # ---- Optimizer ---------------------------------------------------------
    optimizer = model.setup_optimizer(
        unembedding_lr=cfg.optimizer.unembedding_lr,
        embedding_lr=cfg.optimizer.embedding_lr,
        matrix_lr=cfg.optimizer.matrix_lr,
        weight_decay=cfg.optimizer.weight_decay,
        scalar_lr=cfg.optimizer.scalar_lr,
    )

    # ---- Data loaders (real or injected) -----------------------------------
    if train_loader is None:
        train_loader = build_data_loader(cfg, split="train", device=str(device))
    if val_loader is None and cfg.eval.eval_every > 0:
        val_loader = build_data_loader(cfg, split="val", device=str(device))

    # ---- Wandb (optional) --------------------------------------------------
    if master and cfg.logging.wandb_mode != "disabled":
        import wandb
        wandb_run = wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.run_name,
            mode=cfg.logging.wandb_mode,
            config=cfg.model_dump(),
        )
    else:
        wandb_run = DummyWandb()

    # ---- Training loop -----------------------------------------------------
    print0(f"Starting training: {num_iterations:,} iterations on {world_size} ranks")
    state = TrainState()
    t_start = time.time()

    world_tokens_per_fwdbwd = (
        cfg.training.device_batch_size * cfg.model.max_seq_len * world_size
    )
    grad_accum_steps = max(cfg.training.total_batch_size // world_tokens_per_fwdbwd, 1)
    print0(
        f"World tokens per fwd+bwd: {world_tokens_per_fwdbwd:,}; "
        f"gradient accumulation = {grad_accum_steps}"
    )

    # Number of val batches per eval pass (derived from eval_tokens).
    eval_batches = max(cfg.eval.eval_tokens // max(world_tokens_per_fwdbwd, 1), 1)

    model.train()
    for step in range(num_iterations):
        state.step = step

        # LR schedule
        mult = lr_multiplier(
            step,
            num_iterations,
            cfg.training.warmup_steps,
            cfg.training.warmdown_ratio,
            cfg.training.final_lr_frac,
        )
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * mult

        # Gradient-accumulated forward + backward
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum_steps):
            idx, targets = next(train_loader)
            loss = model(idx, targets=targets)
            (loss / grad_accum_steps).backward()
            loss_accum += float(loss.detach())
        loss_accum /= grad_accum_steps
        optimizer.step()

        # Smoothed loss (EMA)
        state.smooth_loss = (
            0.9 * state.smooth_loss + 0.1 * loss_accum if step > 0 else loss_accum
        )

        if step % 10 == 0 and master:
            dt = time.time() - t_start
            tok_per_sec = (step + 1) * cfg.training.total_batch_size / max(dt, 1e-6)
            print0(
                f"step {step:6d}/{num_iterations} | loss {loss_accum:.4f} "
                f"| smooth {state.smooth_loss:.4f} | lr_mult {mult:.3f} "
                f"| tok/s {tok_per_sec:.2e}"
            )
            wandb_run.log(
                {
                    "step": step,
                    "loss": loss_accum,
                    "smooth_loss": state.smooth_loss,
                    "lr_mult": mult,
                    "tok_per_sec": tok_per_sec,
                }
            )

        # Periodic evaluation
        if (
            cfg.eval.eval_every > 0
            and val_loader is not None
            and step > 0
            and step % cfg.eval.eval_every == 0
        ):
            val_loss = evaluate(model, val_loader, eval_batches)
            # bits-per-residue: loss is natural-log cross-entropy; convert to bits.
            val_bpr = val_loss / math.log(2)
            state.last_val_loss = val_loss
            state.last_val_bpr = val_bpr
            if val_bpr < state.best_val_bpr:
                state.best_val_bpr = val_bpr
            if master:
                print0(
                    f"  [eval] step {step}: val_loss={val_loss:.4f} "
                    f"val_bpr={val_bpr:.4f} (best={state.best_val_bpr:.4f})"
                )
                wandb_run.log(
                    {
                        "step": step,
                        "val_loss": val_loss,
                        "val_bpr": val_bpr,
                        "best_val_bpr": state.best_val_bpr,
                    }
                )

        # Periodic checkpoint
        if (
            cfg.checkpointing.save_every > 0
            and step > 0
            and step % cfg.checkpointing.save_every == 0
        ):
            _save(cfg, step, model, optimizer, state, rank, num_iterations)

    # ---- Final eval + checkpoint -------------------------------------------
    if val_loader is not None and cfg.eval.eval_every > 0:
        val_loss = evaluate(model, val_loader, eval_batches)
        val_bpr = val_loss / math.log(2)
        state.last_val_loss = val_loss
        state.last_val_bpr = val_bpr
        if val_bpr < state.best_val_bpr:
            state.best_val_bpr = val_bpr
        if master:
            print0(
                f"  [final eval] val_loss={val_loss:.4f} val_bpr={val_bpr:.4f} "
                f"(best={state.best_val_bpr:.4f})"
            )
            wandb_run.log(
                {
                    "step": num_iterations,
                    "val_loss": val_loss,
                    "val_bpr": val_bpr,
                    "best_val_bpr": state.best_val_bpr,
                }
            )

    _save(cfg, num_iterations, model, optimizer, state, rank, num_iterations)

    if not isinstance(wandb_run, DummyWandb):
        wandb_run.finish()

    compute_cleanup()
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(
    cfg: NanoprotConfig,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainState,
    rank: int,
    num_iterations: int,
) -> None:
    """Save model + optimizer + meta in the format ``checkpoint.save_checkpoint`` expects."""
    meta = {
        "step": step,
        "smooth_loss": state.smooth_loss,
        "last_val_loss": state.last_val_loss,
        "last_val_bpr": state.last_val_bpr,
        "best_val_bpr": state.best_val_bpr,
        "num_iterations": num_iterations,
        "model_config": cfg.model.model_dump(),
        "training_config": cfg.training.model_dump(),
        "name": cfg.name,
        "version": "0.3.0",
    }
    save_checkpoint(
        cfg.checkpointing.output_dir,
        step,
        model.state_dict(),
        optimizer.state_dict(),
        meta,
        rank=rank,
    )
