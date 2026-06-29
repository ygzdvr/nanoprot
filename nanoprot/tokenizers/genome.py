"""
nanoprot.tokenizers.genome — character-level DNA tokenizer for the B-GEN cross-domain experiment
(PRD: brain/plans/B-GEN_genomics_PRD.md). Mirrors :class:`nanoprot.tokenizers.esm2.Esm2Tokenizer`
exactly (same callable surface) so the AR data loader and probe path consume it drop-in.

Vocabulary (10 tokens) — single-nucleotide alphabet {A,C,G,T,N} + ESM-2-style special tokens:

::

  0  <cls>   1  <pad>   2  <eos>   3  <unk>
  4  A       5  C       6  G       7  T       8  N
  9  <mask>

``<cls>`` is the BOS-equivalent (matches the ESM-2 convention the data loader/probe rely on). The
five "real" tokens (A,C,G,T,N) occupy ids 4-8; N is a kept symbol (ambiguous base), not <unk>.
Soft-masked / IUPAC / any other character maps to <unk> (the windows emitted by prepare_genome are
already cleaned to ACGTN, so <unk> should not occur in practice — it is a safety fallback).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

# Token table — DO NOT REORDER (specials first, mirroring the ESM-2 layout).
GENOME_TOKENS: List[str] = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "A", "C", "G", "T", "N",
    "<mask>",
]

ID_CLS = 0
ID_PAD = 1
ID_EOS = 2
ID_UNK = 3
ID_MASK = 9


class GenomeTokenizer:
    """Character-level DNA tokenizer. Interface mirrors
    :class:`nanoprot.tokenizers.esm2.Esm2Tokenizer` (same methods, same semantics)."""

    def __init__(self) -> None:
        self._token_to_id = {tok: i for i, tok in enumerate(GENOME_TOKENS)}
        self._id_to_token = list(GENOME_TOKENS)

    # -- introspection ------------------------------------------------------
    def get_vocab_size(self) -> int:
        return len(GENOME_TOKENS)

    def get_bos_token_id(self) -> int:
        return ID_CLS

    def get_pad_token_id(self) -> int:
        return ID_PAD

    def get_eos_token_id(self) -> int:
        return ID_EOS

    def get_mask_token_id(self) -> int:
        return ID_MASK

    def encode_special(self, name: str) -> int:
        if name not in self._token_to_id:
            raise KeyError(
                f"unknown special token {name!r}; valid: {[t for t in GENOME_TOKENS if t.startswith('<')]}"
            )
        return self._token_to_id[name]

    # -- encode -------------------------------------------------------------
    def _encode_one(self, text: str) -> List[int]:
        unk = ID_UNK
        return [self._token_to_id.get(ch, unk) for ch in text]

    def encode(self, text, prepend: Optional[int] = None, append: Optional[int] = None,
               num_threads: int = 1) -> List[List[int]]:
        """Encode a single sequence or an iterable of sequences (batched). ``prepend``/``append``
        optionally add a token id to every row (the loader prepends ``<cls>``). Returns one
        list of token ids per input sequence. Mirrors the ESM-2 tokenizer API exactly."""
        del num_threads  # parity with the BPE/ESM-2 tokenizers
        texts: Sequence[str] = [text] if isinstance(text, str) else list(text)
        out: List[List[int]] = []
        for s in texts:
            ids = self._encode_one(s)
            if prepend is not None:
                ids = [prepend, *ids]
            if append is not None:
                ids = [*ids, append]
            out.append(ids)
        return out

    # -- decode -------------------------------------------------------------
    def decode(self, ids: Iterable[int]) -> str:
        """Decode token ids to a string; special tokens are emitted verbatim, bases joined."""
        return "".join(self._id_to_token[i] for i in ids)


_GENOME_TOKENIZER_SINGLETON: Optional[GenomeTokenizer] = None


def get_genome_tokenizer() -> GenomeTokenizer:
    """Return a cached :class:`GenomeTokenizer` (no trained state; safe to share)."""
    global _GENOME_TOKENIZER_SINGLETON
    if _GENOME_TOKENIZER_SINGLETON is None:
        _GENOME_TOKENIZER_SINGLETON = GenomeTokenizer()
    return _GENOME_TOKENIZER_SINGLETON
