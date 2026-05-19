"""
Residue-level tokenizer for protein sequences.

Each amino acid is its own token. No BPE, no learned merges. This is the
canonical tokenization used by most protein bio literature (ESM-2, AlphaFold,
etc.) and gives cleaner γ/β measurements than BPE because there is no BPE
transition zone in the short-lag regime of token-token correlations.

Vocabulary layout:
    0..19 : 20 standard amino acids in alphabetical order
            (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
    20..24: 5 IUPAC ambiguity codes (X, B, U, O, Z)
    25    : <|bos|>
    26    : <|eos|>
Total V = 27.

The class exposes enough of the RustBPETokenizer interface that
`scripts/prot_compute_correlations.py` and `scripts/prot_ngram_entropy.py`
can swap tokenizers by a single CLI flag.
"""

import os
from functools import lru_cache

import torch

# 20 standard amino acids, alphabetical order (stable, documentable index map).
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
# IUPAC ambiguity codes (rare; <1% of residues in UniRef50 combined).
AMBIG_AA = "XBUOZ"
ALL_AA = STANDARD_AA + AMBIG_AA  # 25 characters

BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"


class ResidueTokenizer:
    """
    Minimal residue-level tokenizer. Fast (pure dict lookup), deterministic,
    interchangeable with `RustBPETokenizer` for γ/β measurement scripts.

    Usage:
        tok = ResidueTokenizer()
        ids = tok.encode("MKTAYIAKQR")   # -> list[int]
        tok.get_vocab_size()             # -> 27
        tok.get_bos_token_id()           # -> 25

    Notes:
        - Unknown characters (newlines, '*', numbers, lowercase, etc.) are
          silently dropped during encode. Protein parquet text should contain
          only IUPAC letters, so this is fine for UniRef50/UniRef90/Swiss-Prot/Pfam.
        - `encode(list[str])` returns list[list[int]] matching the
          `RustBPETokenizer` batch interface.
    """

    BOS_ID = 25
    EOS_ID = 26
    VOCAB_SIZE = 27

    def __init__(self):
        self.char_to_id = {c: i for i, c in enumerate(ALL_AA)}  # 25 entries
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.special_tokens = {BOS_TOKEN: self.BOS_ID, EOS_TOKEN: self.EOS_ID}

    # ---- metadata ----------------------------------------------------------
    def get_vocab_size(self) -> int:
        return self.VOCAB_SIZE

    def get_bos_token_id(self) -> int:
        return self.BOS_ID

    def get_special_tokens(self):
        return list(self.special_tokens.keys())

    @lru_cache(maxsize=16)
    def encode_special(self, token_str: str) -> int:
        if token_str not in self.special_tokens:
            raise KeyError(f"Unknown special token: {token_str!r}")
        return self.special_tokens[token_str]

    # ---- encode / decode ---------------------------------------------------
    def _encode_one(self, text: str, prepend=None, append=None) -> list:
        ids = []
        if prepend is not None:
            pid = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(pid)
        c2i = self.char_to_id
        ids.extend(c2i[c] for c in text if c in c2i)
        if append is not None:
            aid = append if isinstance(append, int) else self.encode_special(append)
            ids.append(aid)
        return ids

    def encode(self, text, prepend=None, append=None, num_threads=None):
        # `num_threads` accepted but ignored (this tokenizer is fast enough single-threaded)
        del num_threads
        if isinstance(text, str):
            return self._encode_one(text, prepend, append)
        if isinstance(text, list):
            return [self._encode_one(t, prepend, append) for t in text]
        raise ValueError(f"encode() expects str or list[str], got {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids) -> str:
        out = []
        i2c = self.id_to_char
        for tid in ids:
            if tid in i2c:
                out.append(i2c[tid])
            elif tid == self.BOS_ID:
                out.append(BOS_TOKEN)
            elif tid == self.EOS_ID:
                out.append(EOS_TOKEN)
        return "".join(out)


def get_residue_tokenizer() -> ResidueTokenizer:
    """Drop-in equivalent of `get_protein_tokenizer()` for residue-level work."""
    return ResidueTokenizer()


def get_residue_token_bytes(device: str = "cpu") -> torch.Tensor:
    """
    Return a (V,) int32 tensor where entry i = number of amino-acid residues
    that token i represents. For residue-level tokenization this is trivially
    1 for every real AA token and 0 for special tokens.

    Used by `evaluate_bpb` to compute bits-per-residue (BPR) without any
    additional plumbing.
    """
    V = ResidueTokenizer.VOCAB_SIZE
    tb = torch.ones(V, dtype=torch.int32, device=device)
    tb[ResidueTokenizer.BOS_ID] = 0
    tb[ResidueTokenizer.EOS_ID] = 0
    return tb
