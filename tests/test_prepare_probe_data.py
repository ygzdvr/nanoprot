"""NetSurfP-2.0 npz parsing: sequence + Q8->SS3 decode, eval mask, cache round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from nanoprot.eval.probe.labels import load_probe_dataset
from scripts.prepare_probe_data import (
    CH_AA, CH_EVAL_MASK, CH_SEQ_MASK, NSP_AA_ORDER, NSP_Q8_ORDER,
    load_npz, parse_netsurfp, write_cache,
)


def _synthetic(n_feat: int = 65) -> np.ndarray:
    """2 proteins (L=6): 'ACDE' with Q8 GHEC, and 'MKT' with Q8 IBS."""
    data = np.zeros((2, 6, n_feat), dtype=np.float32)
    for sample, (seq, q8) in enumerate([("ACDE", "GHEC"), ("MKT", "IBS")]):
        for pos, (aa, q) in enumerate(zip(seq, q8)):
            data[sample, pos, CH_SEQ_MASK] = 1.0
            data[sample, pos, NSP_AA_ORDER.index(aa)] = 1.0
            data[sample, pos, 57 + NSP_Q8_ORDER.index(q)] = 1.0
    return data


def test_parse_decodes_sequence_and_ss3() -> None:
    parsed = parse_netsurfp(_synthetic())
    assert len(parsed) == 2
    _, seq0, lab0 = parsed[0]
    assert seq0 == "ACDE"
    assert lab0 == [0, 0, 1, 2]          # G,H->helix; E->strand; C->coil
    _, seq1, lab1 = parsed[1]
    assert seq1 == "MKT"
    assert lab1 == [0, 1, 2]             # I->helix; B->strand; S->coil


def test_eval_mask_masks_unscored_positions() -> None:
    data = _synthetic()
    data[0, 0, CH_EVAL_MASK] = 1.0       # only score positions 0 and 2 of protein 0
    data[0, 2, CH_EVAL_MASK] = 1.0
    _, _, lab0 = parse_netsurfp(data, use_eval_mask=True)[0]
    assert lab0 == [0, -100, 1, -100]


def test_unknown_residue_becomes_X() -> None:
    data = _synthetic()
    data[1, 0, CH_AA] = 0.0              # all-zero AA = unknown
    _, seq1, _ = parse_netsurfp(data)[1]
    assert seq1[0] == "X"


def test_parse_rejects_too_few_features() -> None:
    with pytest.raises(ValueError):
        parse_netsurfp(np.zeros((2, 6, 10), dtype=np.float32))


def test_load_npz_finds_3d_array_and_ids(tmp_path) -> None:
    p = tmp_path / "t.npz"
    np.savez(p, ids=np.array(["p0", "p1"]), data=_synthetic())
    data, ids = load_npz(p)
    assert data.shape == (2, 6, 65)
    assert list(ids) == ["p0", "p1"]


def test_write_cache_roundtrips_through_loader(tmp_path) -> None:
    parsed = parse_netsurfp(_synthetic())
    proteins = [{"id": pid, "sequence": seq, "labels": labels, "split": "train"}
                for pid, seq, labels in parsed]
    write_cache(tmp_path, proteins, provenance={"source": "test"})
    ds = load_probe_dataset(tmp_path)
    assert ds.concept == "ss3" and ds.source == "netsurfp" and ds.n_classes == 3
    assert len(ds.proteins) == 2
    assert ds.proteins[0].sequence == "ACDE" and ds.proteins[0].labels == [0, 0, 1, 2]
