"""
nanoprot.tokenizers.esm2 — character-level tokenizer matching the ESM-2 alphabet.

The vocabulary (33 tokens) matches the public ``facebook/esm2_*`` tokenizers
exactly, so a model trained here can be compared head-to-head with the
released ESM-2 family without re-tokenisation:

::

  0  <cls>     1  <pad>    2  <eos>     3  <unk>
  4  L         5  A        6  G         7  V
  8  S         9  E       10  R        11  T
 12  I        13  D       14  P        15  K
 16  Q        17  N       18  F        19  Y
 20  M        21  H       22  W        23  C
 24  X        25  B       26  U        27  Z
 28  O        29  .       30  -        31  <null_1>
 32  <mask>

The 20 canonical amino acids occupy ids 4-23. Five rare/ambiguous residues
(X, B, U, Z, O) occupy 24-28. The remaining slots are special / structural
tokens. ``<cls>`` is used as the BOS-equivalent (matches HuggingFace's
``EsmTokenizer.cls_token_id``).

This tokenizer exposes the same callable surface as
:class:`nanoprot.tokenizers.bpe.Tokenizer` so the AR data loader can use
either drop-in.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

# Token table — DO NOT REORDER. Matches the public ESM-2 tokenizer.
ESM2_TOKENS: List[str] = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P",
    "K", "Q", "N", "F", "Y", "M", "H", "W", "C",
    "X", "B", "U", "Z", "O",
    ".", "-", "<null_1>", "<mask>",
]

# Convenience indices
ID_CLS = 0
ID_PAD = 1
ID_EOS = 2
ID_UNK = 3
ID_MASK = 32


class Esm2Tokenizer:
    """Character-level ESM-2 tokenizer.

    The interface mirrors :class:`nanoprot.tokenizers.bpe.Tokenizer` so the
    same data loader can consume both:

    - :meth:`get_vocab_size`     -> 33
    - :meth:`get_bos_token_id`   -> 0  (``<cls>``)
    - :meth:`get_pad_token_id`   -> 1  (``<pad>``)
    - :meth:`get_eos_token_id`   -> 2  (``<eos>``)
    - :meth:`get_mask_token_id`  -> 32 (``<mask>``)
    - :meth:`encode_special`     -> id by special-token name
    - :meth:`encode`             -> list of token-id lists (batched)
    - :meth:`decode`             -> str (single sequence)

    Unknown characters (anything outside the alphabet above) map to ``<unk>``.
    """

    def __init__(self) -> None:
        self._token_to_id = {tok: i for i, tok in enumerate(ESM2_TOKENS)}
        self._id_to_token = list(ESM2_TOKENS)

    # -- introspection ------------------------------------------------------

    def get_vocab_size(self) -> int:
        return len(ESM2_TOKENS)

    def get_bos_token_id(self) -> int:
        return ID_CLS

    def get_pad_token_id(self) -> int:
        return ID_PAD

    def get_eos_token_id(self) -> int:
        return ID_EOS

    def get_mask_token_id(self) -> int:
        return ID_MASK

    def encode_special(self, name: str) -> int:
        """Return the token id of a special token by name (e.g. ``"<cls>"``)."""
        if name not in self._token_to_id:
            raise KeyError(
                f"unknown special token {name!r}; valid: {[t for t in ESM2_TOKENS if t.startswith('<')]}"
            )
        return self._token_to_id[name]

    # -- encode -------------------------------------------------------------

    def _encode_one(self, text: str) -> List[int]:
        unk = ID_UNK
        return [self._token_to_id.get(ch, unk) for ch in text]

    def encode(
        self,
        text,
        prepend: Optional[int] = None,
        append: Optional[int] = None,
        num_threads: int = 1,
    ) -> List[List[int]]:
        """Encode a batch of protein sequences.

        Parameters
        ----------
        text : str | Iterable[str]
            A single sequence or an iterable of sequences. The iterable form
            is what the ``tokenizing_distributed_data_loader_bos_bestfit``
            data loader uses.
        prepend : int, optional
            If set, prepend this token id to every output row (used to
            prepend ``<cls>`` so the dataloader's BOS-aligned packing
            invariant holds).
        append : int, optional
            If set, append this token id to every output row.
        num_threads : int
            Ignored (this character-level tokenizer is already fast); the
            kwarg exists for parity with the BPE tokenizer's API.

        Returns
        -------
        list of list of int
            One list of token ids per input sequence.
        """
        del num_threads  # parity with BPE tokenizer
        if isinstance(text, str):
            texts: Sequence[str] = [text]
        else:
            texts = list(text)
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
        """Decode a list of token ids back to a string. Special tokens are
        emitted verbatim (e.g. ``"<cls>"``); residue tokens are joined with
        no separator."""
        return "".join(self._id_to_token[i] for i in ids)


# -- convenience factory ------------------------------------------------------

_ESM2_TOKENIZER_SINGLETON: Optional[Esm2Tokenizer] = None


def get_esm2_tokenizer() -> Esm2Tokenizer:
    """Return a (cached) :class:`Esm2Tokenizer` instance.

    Mirrors :func:`nanoprot.tokenizers.bpe.get_protein_tokenizer`. The
    character-level ESM-2 tokenizer has no trained state, so the cached
    singleton is safe to share across the training process.
    """
    global _ESM2_TOKENIZER_SINGLETON
    if _ESM2_TOKENIZER_SINGLETON is None:
        _ESM2_TOKENIZER_SINGLETON = Esm2Tokenizer()
    return _ESM2_TOKENIZER_SINGLETON
