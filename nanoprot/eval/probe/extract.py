"""
Residual-stream activation extraction for cross-architecture probing.

All three nanoprot architectures (gpt2, esm2, mamba) share one trunk layout: the
blocks live at ``model.transformer.h``, each block takes the residual stream as its
first positional argument and returns the updated residual stream, and the top-level
forward is ``embed -> for block in transformer.h -> ln_f -> lm_head``. We exploit that
to sample the residual stream at ``n_layer + 1`` points via PyTorch forward hooks —
non-invasively, so this runs on the *released* checkpoints with no model-code change:

    point 0        = input to block 0     (the post-embedding residual)
    point k (1..n) = output of block k-1  (resid-post of layer k-1)

The hook point is fixed here, once, so the cross-architecture layer comparison is
apples-to-apples (docs/probing_harness_plan.md §5). gpt2 rescales the residual stream
between blocks (``resid_lambdas`` / ``x0_lambdas``) and applies a backout *after* the
trunk; sampling at block boundaries is the clean definition shared by all three archs
(esm2 and mamba are plain pre-LN ``x + sublayer(norm(x))`` with no inter-block term).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List

import torch


def get_blocks(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Return the ordered trunk blocks of any nanoprot model (``model.transformer.h``)."""
    transformer = getattr(model, "transformer", None)
    blocks = getattr(transformer, "h", None) if transformer is not None else None
    if blocks is None:
        raise ValueError(
            f"{type(model).__name__} has no `transformer.h` block list; extraction "
            "assumes the shared nanoprot trunk layout (gpt2 / esm2 / mamba)."
        )
    return blocks


def n_probe_points(model: torch.nn.Module) -> int:
    """Number of residual-stream sampling points for ``model`` = ``n_layer + 1``."""
    return len(get_blocks(model)) + 1


def relative_depths(n_points: int) -> List[float]:
    """Map probe-point indices ``0 .. n_points-1`` to relative depth in ``[0, 1]``.

    Lets layer-wise curves be compared across architectures and scales that differ in
    depth (point 0 -> 0.0 = embedding, last point -> 1.0 = final block output).
    """
    last = max(n_points - 1, 1)
    return [i / last for i in range(n_points)]


@contextmanager
def _capture(model: torch.nn.Module) -> Iterator[dict]:
    """Register forward hooks that fill ``store[k]`` with the residual stream at each
    probe point during a forward pass; cleanly removed on exit."""
    blocks = get_blocks(model)
    store: dict = {}
    handles = []

    def _pre_hook(_module, args):
        x = args[0]
        store[0] = (x[0] if isinstance(x, tuple) else x).detach()

    def _make_out_hook(layer_idx: int):
        def _out_hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            store[layer_idx + 1] = tensor.detach()
        return _out_hook

    handles.append(blocks[0].register_forward_pre_hook(_pre_hook))
    for layer_idx, block in enumerate(blocks):
        handles.append(block.register_forward_hook(_make_out_hook(layer_idx)))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def extract_layers(model: torch.nn.Module, idx: torch.Tensor) -> List[torch.Tensor]:
    """Run one forward pass and return the residual stream at every probe point.

    Parameters
    ----------
    model : the loaded model (any arch); used in eval mode, weights untouched.
    idx   : ``(B, T)`` int64 token ids on the model's device.

    Returns
    -------
    list of ``n_layer + 1`` tensors, each ``(B, T, d_model)``, detached and on the
    model's device, ordered from embedding (point 0) to final block output (point n).
    """
    was_training = model.training
    model.eval()
    with _capture(model) as store:
        model(idx)
    if was_training:
        model.train()
    expected = n_probe_points(model)
    if len(store) != expected:
        raise RuntimeError(
            f"captured {len(store)} probe points, expected {expected} — did the "
            "forward run every block exactly once?"
        )
    return [store[k] for k in range(expected)]


def token_identity_features(idx: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """One-hot amino-acid identity ``(B, T, vocab_size)`` — the trivial probe baseline.

    A linear probe on this measures what is decodable from the residue label alone
    (no context, no learned representation); learned models must beat it to count.
    """
    return torch.nn.functional.one_hot(idx.long(), num_classes=vocab_size).float()
