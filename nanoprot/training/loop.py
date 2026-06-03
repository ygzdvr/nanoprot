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

import json
import math
import os
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


@torch.no_grad()
def evaluate_metrics(
    model: torch.nn.Module,
    val_loader: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    num_batches: int,
    compute_accuracy: bool = False,
) -> dict:
    """Validation loss (+ optional argmax token accuracy) over ``num_batches``.

    With ``compute_accuracy=False`` this returns exactly the same loss as
    :func:`evaluate` — the loop uses it as a drop-in so the default path is
    unchanged. Accuracy is next-token (AR) or masked-token (MLM) argmax accuracy
    over the supervised positions (``targets != -100``), at the cost of one extra
    logits pass per batch.
    """
    if num_batches <= 0:
        return {"loss": float("nan"), "accuracy": None}
    was_training = model.training
    model.eval()
    losses = []
    n_correct = n_total = 0
    for _ in range(num_batches):
        idx, targets = next(val_loader)
        losses.append(float(model(idx, targets=targets).detach()))
        if compute_accuracy:
            preds = model(idx).argmax(dim=-1)
            mask = targets != -100
            n_correct += int((preds[mask] == targets[mask]).sum())
            n_total += int(mask.sum())
    if was_training:
        model.train()
    acc = (n_correct / n_total) if (compute_accuracy and n_total > 0) else None
    return {"loss": sum(losses) / len(losses), "accuracy": acc}


def _write_history(path: str, records: list) -> None:
    """Write the per-step/-eval training history as JSON Lines (best-effort)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    except Exception as e:  # pragma: no cover - logging must never crash training
        print0(f"  [history] failed to write {path}: {e}")


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

    # Optional per-step training history (off by default; the analysis tools read
    # history.jsonl). Timing is tracked regardless (cheap) but only recorded when
    # logging.history is on, so the default path is unchanged.
    history: list = []
    history_path = os.path.join(cfg.checkpointing.output_dir, "history.jsonl")
    write_history = cfg.logging.history and master
    t_prev = t_start
    cum_tokens = 0

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

        # Per-step timing (for compute-time analysis).
        now = time.time()
        dt_step = now - t_prev
        t_prev = now
        cum_tokens += cfg.training.total_batch_size
        step_tok_per_sec = cfg.training.total_batch_size / max(dt_step, 1e-6)

        # Full per-step training history (off by default).
        if write_history:
            history.append({
                "type": "train", "step": step, "loss": loss_accum,
                "smooth_loss": state.smooth_loss, "lr_mult": mult,
                "dt_sec": dt_step, "tok_per_sec": step_tok_per_sec,
                "tokens": cum_tokens, "wall_sec": now - t_start,
            })

        if step % cfg.logging.log_every == 0 and master:
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
            vm = evaluate_metrics(model, val_loader, eval_batches, cfg.eval.compute_accuracy)
            val_loss = vm["loss"]
            # bits-per-residue: loss is natural-log cross-entropy; convert to bits.
            val_bpr = val_loss / math.log(2)
            val_ppl = math.exp(val_loss)
            state.last_val_loss = val_loss
            state.last_val_bpr = val_bpr
            if val_bpr < state.best_val_bpr:
                state.best_val_bpr = val_bpr
            if master:
                acc_s = f" acc={vm['accuracy']:.4f}" if vm["accuracy"] is not None else ""
                print0(
                    f"  [eval] step {step}: val_loss={val_loss:.4f} "
                    f"val_bpr={val_bpr:.4f} ppl={val_ppl:.3f}{acc_s} "
                    f"(best={state.best_val_bpr:.4f})"
                )
                wandb_run.log(
                    {
                        "step": step,
                        "val_loss": val_loss,
                        "val_bpr": val_bpr,
                        "val_ppl": val_ppl,
                        "val_accuracy": vm["accuracy"],
                        "best_val_bpr": state.best_val_bpr,
                    }
                )
                if write_history:
                    history.append({
                        "type": "eval", "step": step, "val_loss": val_loss,
                        "val_bpr": val_bpr, "val_ppl": val_ppl,
                        "val_accuracy": vm["accuracy"], "tokens": cum_tokens,
                        "wall_sec": time.time() - t_start,
                    })

        # Periodic checkpoint
        save_now = (
            cfg.checkpointing.save_every > 0
            and step > 0
            and step % cfg.checkpointing.save_every == 0
        ) or (step in set(cfg.checkpointing.save_steps))
        if save_now:
            _save(
                cfg, step, model, optimizer, state, rank, num_iterations,
                n_params=n_params, flops_per_token=flops_per_token,
                total_residues=total_residues, world_size=world_size,
                train_time_sec=time.time() - t_start,
            )
            if write_history:
                _write_history(history_path, history)

    # ---- Final eval + checkpoint -------------------------------------------
    if val_loader is not None and cfg.eval.eval_every > 0:
        vm = evaluate_metrics(model, val_loader, eval_batches, cfg.eval.compute_accuracy)
        val_loss = vm["loss"]
        val_bpr = val_loss / math.log(2)
        val_ppl = math.exp(val_loss)
        state.last_val_loss = val_loss
        state.last_val_bpr = val_bpr
        if val_bpr < state.best_val_bpr:
            state.best_val_bpr = val_bpr
        if master:
            acc_s = f" acc={vm['accuracy']:.4f}" if vm["accuracy"] is not None else ""
            print0(
                f"  [final eval] val_loss={val_loss:.4f} val_bpr={val_bpr:.4f} "
                f"ppl={val_ppl:.3f}{acc_s} (best={state.best_val_bpr:.4f})"
            )
            wandb_run.log(
                {
                    "step": num_iterations,
                    "val_loss": val_loss,
                    "val_bpr": val_bpr,
                    "val_ppl": val_ppl,
                    "val_accuracy": vm["accuracy"],
                    "best_val_bpr": state.best_val_bpr,
                }
            )
            if write_history:
                history.append({
                    "type": "eval", "step": num_iterations, "val_loss": val_loss,
                    "val_bpr": val_bpr, "val_ppl": val_ppl,
                    "val_accuracy": vm["accuracy"], "tokens": cum_tokens,
                    "wall_sec": time.time() - t_start,
                })

    _save(
        cfg, num_iterations, model, optimizer, state, rank, num_iterations,
        n_params=n_params, flops_per_token=flops_per_token,
        total_residues=total_residues, world_size=world_size,
        train_time_sec=time.time() - t_start,
    )
    if write_history:
        _write_history(history_path, history)

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
    *,
    n_params: Optional[int] = None,
    flops_per_token: Optional[float] = None,
    total_residues: Optional[int] = None,
    world_size: Optional[int] = None,
    train_time_sec: Optional[float] = None,
) -> None:
    """Save model + optimizer + meta in the format ``checkpoint.save_checkpoint`` expects.

    The metadata is written so the checkpoint is **self-describing**: it carries
    the full resolved config plus the trained-artifact facts (real parameter
    count, FLOPs/token, residues seen, wall-clock, world size) that downstream
    tooling — the model-card generator in particular — reports without having to
    re-instantiate the model.
    """
    try:
        from nanoprot import __version__ as _nanoprot_version
    except Exception:  # pragma: no cover - version import is best-effort
        _nanoprot_version = "unknown"

    total_flops = (
        float(flops_per_token) * float(total_residues)
        if (flops_per_token is not None and total_residues is not None)
        else None
    )
    meta = {
        "step": step,
        "smooth_loss": state.smooth_loss,
        "last_val_loss": state.last_val_loss,
        "last_val_bpr": state.last_val_bpr,
        "best_val_bpr": state.best_val_bpr,
        "num_iterations": num_iterations,
        # Trained-artifact facts (new in v0.5; absent in older checkpoints).
        "n_params": n_params,
        "flops_per_token": flops_per_token,
        "total_flops": total_flops,
        "total_residues": total_residues,
        "world_size": world_size,
        "train_time_sec": train_time_sec,
        # Config: keep model_config/training_config for back-compat with the
        # v0.2 loader, AND embed the full resolved config so the checkpoint is
        # fully reproducible from its own metadata.
        "model_config": cfg.model.model_dump(),
        "training_config": cfg.training.model_dump(),
        "config": cfg.model_dump(),
        "name": cfg.name,
        "version": _nanoprot_version,
    }
    save_checkpoint(
        cfg.checkpointing.output_dir,
        step,
        model.state_dict(),
        optimizer.state_dict(),
        meta,
        rank=rank,
    )
