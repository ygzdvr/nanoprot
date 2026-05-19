"""
MLM data loader for ESM-2-style training.

Wraps the autoregressive BOS-aligned packing loader and applies BERT-style
masking on-the-fly. For each batch the AR loader yields, we:

1. take the input row (already packed, BOS-prefixed, of shape (B, T));
2. choose a per-residue mask with probability ``mlm_probability``, excluding
   special tokens (BOS / PAD / EOS / MASK itself);
3. of the masked positions, replace 80% with ``<mask>``, 10% with a uniformly
   random token from the vocabulary, and leave 10% unchanged
   (the standard BERT recipe);
4. emit targets that hold the *original* token ids at masked positions and
   the ``-100`` ignore index everywhere else (so cross-entropy averages over
   the masked positions only).

The function signature matches
:func:`nanoprot.data.dataloader.tokenizing_distributed_data_loader_bos_bestfit`
so the training loop's data dispatch is a simple branch on
:attr:`TrainingConfig.objective <nanoprot.config.TrainingConfig.objective>`.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Set, Tuple

import torch

from nanoprot.data.dataloader import tokenizing_distributed_data_loader_bos_bestfit


def _collect_special_token_ids(tokenizer) -> Set[int]:
    """Return the set of token ids that must never be masked.

    Probes the tokenizer for the common special-token accessors that both
    the BPE and ESM-2 tokenizers expose. Anything missing is silently
    skipped — we err on the side of fewer protections rather than crashing
    on tokenizers that don't implement a given accessor.
    """
    special: Set[int] = set()
    for getter in (
        "get_bos_token_id",
        "get_pad_token_id",
        "get_eos_token_id",
        "get_mask_token_id",
    ):
        fn = getattr(tokenizer, getter, None)
        if fn is None:
            continue
        try:
            tok_id = fn()
        except Exception:
            continue
        if isinstance(tok_id, int) and tok_id >= 0:
            special.add(tok_id)
    return special


def _mlm_mask(
    inputs: torch.Tensor,
    *,
    mask_token_id: int,
    vocab_size: int,
    mlm_probability: float,
    special_ids: Iterable[int],
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply BERT 15%/80/10/10 masking.

    Parameters
    ----------
    inputs : (B, T) int64
        Original tokenized residue sequence (already BOS-aligned + packed).
    mask_token_id : int
        Token id to use for the 80%-replaced positions.
    vocab_size : int
        For sampling random tokens on the 10%-randomised positions.
    mlm_probability : float
        Per-residue probability of being chosen for masking.
    special_ids : iterable of int
        Token ids that must never be masked (BOS / PAD / EOS / MASK itself).
    generator : torch.Generator, optional
        For deterministic masking in tests.

    Returns
    -------
    masked_inputs : (B, T) int64
        The same shape as ``inputs`` with 80%-of-selected replaced by
        ``mask_token_id`` and 10%-of-selected replaced by random tokens.
    targets : (B, T) int64
        Original ids at the selected positions, ``-100`` elsewhere.
    """
    device = inputs.device
    B, T = inputs.shape

    # Per-residue selection probabilities. Start uniform, then zero out
    # any position whose token is in the special set.
    prob = torch.full((B, T), float(mlm_probability), device=device)
    if special_ids:
        # Single fused check: union all special-id positions.
        special_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for sid in special_ids:
            special_mask |= inputs == sid
        prob = prob.masked_fill(special_mask, 0.0)

    selected = torch.bernoulli(prob, generator=generator).to(torch.bool)

    targets = torch.full_like(inputs, fill_value=-100)
    targets[selected] = inputs[selected]

    masked = inputs.clone()
    # Of the selected, 80% become <mask>
    p80 = torch.full_like(prob, 0.8)
    do_mask = torch.bernoulli(p80, generator=generator).to(torch.bool) & selected
    masked[do_mask] = mask_token_id
    # Of the remaining selected, 50% (=10% overall) become random tokens
    not_yet = selected & ~do_mask
    p50 = torch.full_like(prob, 0.5)
    do_random = torch.bernoulli(p50, generator=generator).to(torch.bool) & not_yet
    if do_random.any():
        rand_tokens = torch.randint(
            0, vocab_size, (B, T), device=device, generator=generator
        )
        masked[do_random] = rand_tokens[do_random]
    # The remaining ~10% stay unchanged in `masked`, with their targets set.

    return masked, targets


def mlm_distributed_data_loader_bos_bestfit(
    tokenizer,
    B: int,
    T: int,
    split: str,
    *,
    device: str = "cuda",
    mlm_probability: float = 0.15,
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    buffer_size: int = 1000,
    seed: int | None = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """MLM-targeted data loader, drop-in replacement for the AR loader.

    Yields ``(masked_inputs, targets)`` pairs on ``device``. ``masked_inputs``
    has the same packing invariants as the AR loader (BOS at position 0, no
    padding, ~35% cropping) but with ~15% of residues replaced per the BERT
    masking recipe. ``targets`` holds the original token ids at the masked
    positions and ``-100`` everywhere else.
    """
    mask_token_id = tokenizer.get_mask_token_id()
    vocab_size = tokenizer.get_vocab_size()
    special_ids = _collect_special_token_ids(tokenizer)

    rng: torch.Generator | None
    if seed is not None:
        # Bind RNG to the same device as the eventual tensors. The AR loader
        # places tensors on ``device`` (CUDA / CPU) before we touch them.
        rng = torch.Generator(device=device).manual_seed(int(seed))
    else:
        rng = None

    ar_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer,
        B,
        T,
        split,
        tokenizer_threads=tokenizer_threads,
        tokenizer_batch_size=tokenizer_batch_size,
        device=device,
        buffer_size=buffer_size,
    )

    for inputs, _ar_targets in ar_loader:
        masked, targets = _mlm_mask(
            inputs,
            mask_token_id=mask_token_id,
            vocab_size=vocab_size,
            mlm_probability=mlm_probability,
            special_ids=special_ids,
            generator=rng,
        )
        yield masked, targets
