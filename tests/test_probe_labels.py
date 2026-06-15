"""Probe-label loading + the correctness-critical token<->label alignment."""

from __future__ import annotations

import json
from collections import Counter

import pytest
import torch

from nanoprot.eval.probe.labels import (
    ProbeProtein, assign_splits, default_tokenizer, encode_protein,
    load_probe_dataset, tokenize_and_align,
)
from nanoprot.eval.probe.linear import flatten_residues
from nanoprot.tokenizers.esm2 import ID_CLS


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def test_align_prepends_and_masks_bos() -> None:
    tok = default_tokenizer()
    seq, labels = "MKTA", [0, 1, 2, 0]
    ids, aligned = tokenize_and_align(seq, labels, tok, add_bos=True)
    assert len(ids) == len(aligned) == len(seq) + 1
    assert ids[0] == ID_CLS                       # <cls> prepended (training convention)
    assert aligned[0] == -100                     # ...and masked
    assert ids[1:] == tok.encode(seq)[0]          # residues follow, 1:1
    assert aligned[1:] == labels                  # residue i -> token i+1


def test_align_without_bos_is_one_to_one() -> None:
    tok = default_tokenizer()
    ids, aligned = tokenize_and_align("AGV", [1, 2, 0], tok, add_bos=False)
    assert ids == tok.encode("AGV")[0]
    assert aligned == [1, 2, 0]


def test_align_rejects_non_one_to_one_tokenizer() -> None:
    class _Bpe:                                    # pretends 3 residues -> 5 tokens
        def encode(self, s): return [[1, 2, 3, 4, 5]]
        def get_bos_token_id(self): return 0
    with pytest.raises(ValueError, match="not 1:1"):
        tokenize_and_align("MKT", [0, 1, 2], _Bpe())


def test_align_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        tokenize_and_align("MKT", [0, 1], default_tokenizer())


def test_encode_protein_then_flatten_recovers_labels() -> None:
    """The full chain: BOS masked, unlabelled residue masked, labelled residues land
    at the right token positions after flattening."""
    tok = default_tokenizer()
    p = ProbeProtein("p", "MKTA", [0, 1, -100, 2], "train")   # residue T (idx 2) unlabelled
    ids, aligned = encode_protein(p, tok)
    assert ids.shape == (1, 5) and aligned.shape == (1, 5)
    assert aligned[0].tolist() == [-100, 0, 1, -100, 2]
    # fake per-token features; flatten should keep only the 3 labelled residues
    feats = torch.arange(5 * 3, dtype=torch.float32).reshape(1, 5, 3)
    X, y = flatten_residues(feats, aligned, ignore_index=-100)
    assert y.tolist() == [0, 1, 2]
    assert torch.equal(X[0], feats[0, 1])          # residue 0 ("M") -> token position 1
    assert torch.equal(X[2], feats[0, 4])          # residue 3 ("A") -> token position 4


# ---------------------------------------------------------------------------
# Cached dataset I/O
# ---------------------------------------------------------------------------

def _write_cache(path, proteins, n_classes=3, class_names=("H", "E", "C")):
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(json.dumps({
        "concept": "ss3", "source": "test", "n_classes": n_classes,
        "class_names": list(class_names), "ignore_index": -100,
    }))
    with (path / "data.jsonl").open("w") as fh:
        for p in proteins:
            fh.write(json.dumps(p) + "\n")


def test_load_probe_dataset_roundtrip(tmp_path) -> None:
    _write_cache(tmp_path, [
        {"id": "p1", "sequence": "MKT", "labels": [0, 1, 2], "split": "train"},
        {"id": "p2", "sequence": "AG", "labels": [2, -100], "split": "val"},
    ])
    ds = load_probe_dataset(tmp_path)
    assert ds.n_classes == 3 and ds.class_names == ["H", "E", "C"]
    assert len(ds.proteins) == 2
    assert ds.split("train")[0].id == "p1"
    assert ds.split_sizes() == {"train": 1, "val": 1}


def test_load_rejects_label_length_mismatch(tmp_path) -> None:
    _write_cache(tmp_path, [{"id": "bad", "sequence": "MKT", "labels": [0, 1], "split": "train"}])
    with pytest.raises(ValueError):
        load_probe_dataset(tmp_path)


def test_load_rejects_out_of_range_label(tmp_path) -> None:
    _write_cache(tmp_path, [{"id": "bad", "sequence": "MK", "labels": [0, 9], "split": "train"}])
    with pytest.raises(ValueError):
        load_probe_dataset(tmp_path)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_assign_splits_deterministic_order_independent_and_proportioned() -> None:
    ids = [f"p{i}" for i in range(2000)]
    a = assign_splits(ids, fracs=(0.8, 0.1, 0.1), seed=0)
    b = assign_splits(list(reversed(ids)), fracs=(0.8, 0.1, 0.1), seed=0)
    assert a == b                                  # order-independent (hash-based)
    assert set(a.values()) <= {"train", "val", "test"}
    counts = Counter(a.values())
    assert abs(counts["train"] / 2000 - 0.8) < 0.04
    assert counts["train"] > counts["val"] and counts["train"] > counts["test"]


def test_assign_splits_changes_with_seed() -> None:
    ids = [f"p{i}" for i in range(500)]
    assert assign_splits(ids, seed=0) != assign_splits(ids, seed=1)


def test_assign_splits_rejects_bad_fracs() -> None:
    with pytest.raises(ValueError):
        assign_splits(["a", "b"], fracs=(0.5, 0.4, 0.4))
