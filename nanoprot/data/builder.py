"""
Data-loader and tokenizer dispatch.

Given a :class:`NanoprotConfig`, :func:`build_data_loader` returns an
iterator of ``(idx, targets)`` pairs that the training loop can consume
without caring whether it's an AR or MLM job. The branch on
``training.objective`` is the only place the two pipelines diverge.

The tokenizer is also dispatched here (``cfg.tokenizer.name``) so that an
``arch: esm2`` run with ``objective: mlm`` picks up the ESM-2 tokenizer
automatically, and an ``arch: gpt2`` / ``objective: ar`` run keeps the
UniRef50 BPE tokenizer.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import torch

from nanoprot.config import NanoprotConfig


__all__ = ["build_tokenizer", "build_data_loader"]


def build_tokenizer(cfg: NanoprotConfig):
    """Return a tokenizer object matching ``cfg.tokenizer.name``.

    All returned tokenizers expose the same callable surface — at minimum
    ``encode(text, prepend=..., num_threads=...)``, ``decode(ids)``,
    ``get_vocab_size()``, ``get_bos_token_id()``, and (for MLM)
    ``get_mask_token_id()``.
    """
    name = cfg.tokenizer.name
    if name == "bpe":
        from nanoprot.tokenizers.bpe import get_protein_tokenizer
        return get_protein_tokenizer()
    if name == "esm2":
        from nanoprot.tokenizers.esm2 import get_esm2_tokenizer
        return get_esm2_tokenizer()
    raise ValueError(f"unknown tokenizer {name!r}")


def build_data_loader(
    cfg: NanoprotConfig,
    *,
    split: str,
    device: str,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Build an ``(idx, targets)`` iterator for the given split.

    Parameters
    ----------
    cfg : NanoprotConfig
        Validated configuration.
    split : {"train", "val"}
        Which dataset partition to read.
    device : str
        Target device for the tensors (``"cuda"`` / ``"cpu"`` / ``"mps"``).

    Returns
    -------
    iterator
        Infinite iterator yielding ``(idx, targets)`` int64 tensors of shape
        ``(device_batch_size, max_seq_len)`` on ``device``. The semantics of
        ``targets`` depend on the objective:

        - ``ar``  : next-token prediction targets (shifted by 1 from inputs);
        - ``mlm`` : original token ids at masked positions, ``-100`` elsewhere.
    """
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    tokenizer = build_tokenizer(cfg)

    if cfg.training.objective == "ar":
        from nanoprot.data.dataloader import (
            tokenizing_distributed_data_loader_bos_bestfit,
        )
        return tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer,
            cfg.training.device_batch_size,
            cfg.model.max_seq_len,
            split=split,
            device=device,
        )

    if cfg.training.objective == "mlm":
        from nanoprot.data.mlm import mlm_distributed_data_loader_bos_bestfit
        return mlm_distributed_data_loader_bos_bestfit(
            tokenizer,
            cfg.training.device_batch_size,
            cfg.model.max_seq_len,
            split=split,
            device=device,
            mlm_probability=cfg.training.mlm_probability,
            seed=cfg.training.seed,
        )

    raise ValueError(
        f"unknown training objective {cfg.training.objective!r}; "
        f"expected 'ar' or 'mlm'"
    )
