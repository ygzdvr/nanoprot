"""Tests for :mod:`nanoprot.tokenizers.esm2`."""

from __future__ import annotations

from nanoprot.tokenizers.esm2 import (
    ESM2_TOKENS,
    ID_CLS,
    ID_EOS,
    ID_MASK,
    ID_PAD,
    ID_UNK,
    Esm2Tokenizer,
    get_esm2_tokenizer,
)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def test_vocab_size_is_33() -> None:
    assert Esm2Tokenizer().get_vocab_size() == 33
    assert len(ESM2_TOKENS) == 33


def test_special_token_indices_match_public_esm2() -> None:
    t = Esm2Tokenizer()
    assert t.get_bos_token_id() == ID_CLS == 0
    assert t.get_pad_token_id() == ID_PAD == 1
    assert t.get_eos_token_id() == ID_EOS == 2
    assert t.encode_special("<unk>") == ID_UNK == 3
    assert t.get_mask_token_id() == ID_MASK == 32


def test_canonical_amino_acid_ids() -> None:
    # The 20 standard AAs occupy ids 4-23 in the public ESM-2 order.
    t = Esm2Tokenizer()
    expected = {
        "L": 4, "A": 5, "G": 6, "V": 7, "S": 8, "E": 9, "R": 10, "T": 11,
        "I": 12, "D": 13, "P": 14, "K": 15, "Q": 16, "N": 17, "F": 18,
        "Y": 19, "M": 20, "H": 21, "W": 22, "C": 23,
    }
    for aa, idx in expected.items():
        encoded = t.encode(aa)[0]
        assert encoded == [idx], f"expected {aa} -> {idx}, got {encoded}"


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

def test_encode_single_string_returns_list_of_lists() -> None:
    t = Esm2Tokenizer()
    out = t.encode("MVK")
    assert isinstance(out, list) and len(out) == 1
    assert out[0] == [20, 7, 15]  # M=20, V=7, K=15


def test_encode_batch_is_per_sequence() -> None:
    t = Esm2Tokenizer()
    out = t.encode(["MVK", "AGE"])
    assert out == [[20, 7, 15], [5, 6, 9]]


def test_encode_prepend_appends_bos() -> None:
    t = Esm2Tokenizer()
    out = t.encode("MVK", prepend=t.get_bos_token_id())
    assert out[0][0] == ID_CLS
    assert out[0][1:] == [20, 7, 15]


def test_unknown_characters_map_to_unk() -> None:
    t = Esm2Tokenizer()
    out = t.encode("@?$")
    assert out[0] == [ID_UNK, ID_UNK, ID_UNK]


def test_round_trip_residues_only() -> None:
    t = Esm2Tokenizer()
    seq = "MGRVKLHWEAGNDPTYIQCFSXBUZO"
    ids = t.encode(seq)[0]
    decoded = t.decode(ids)
    assert decoded == seq


def test_num_threads_kwarg_accepted_but_ignored() -> None:
    # The BPE tokenizer takes num_threads; ours must accept it for API parity.
    t = Esm2Tokenizer()
    out = t.encode(["MVK"], num_threads=8)
    assert out == [[20, 7, 15]]


# ---------------------------------------------------------------------------
# Factory + caching
# ---------------------------------------------------------------------------

def test_get_esm2_tokenizer_returns_singleton() -> None:
    a = get_esm2_tokenizer()
    b = get_esm2_tokenizer()
    assert a is b
