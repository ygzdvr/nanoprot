"""NetSurfP-2.0 npz parsing: sequence + Q8->SS3 decode, eval mask, cache round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from nanoprot.eval.probe.labels import load_probe_dataset
from scripts.prepare_probe_data import (
    CH_AA, CH_EVAL_MASK, CH_SEQ_MASK, NSP_AA_ORDER, NSP_Q8_ORDER,
    load_npz, parse_netsurfp, parse_swissprot, sse_to_ss3, write_cache,
)


def test_sse_to_ss3_maps_and_masks_low_plddt() -> None:
    sse = ["a", "a", "b", "c", "b", "a"]
    plddt = [90, 50, 95, 80, 40, 100]
    # a->helix(0), b->strand(1), c->coil(2); residues with pLDDT<70 masked
    assert sse_to_ss3(sse, plddt, plddt_min=70) == [0, -100, 1, 2, -100, 0]


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


def test_parse_netsurfp_ss8_uses_raw_q8() -> None:
    _, _, lab0 = parse_netsurfp(_synthetic(), concept="ss8")[0]
    # protein 0 Q8 = "GHEC"; indices into GHIBESTC: G=0, H=1, E=4, C=7
    assert lab0 == [0, 1, 4, 7]


def test_parse_netsurfp_rsa_is_regression() -> None:
    data = _synthetic()
    data[0, 0, 55] = 0.3      # set RSA (ch 55) on the first two residues of protein 0
    data[0, 1, 55] = 0.8
    _, _, lab0 = parse_netsurfp(data, concept="rsa")[0]
    assert lab0[0] == pytest.approx(0.3) and lab0[1] == pytest.approx(0.8)
    assert all(isinstance(x, float) for x in lab0)        # continuous targets


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
    write_cache(tmp_path, proteins, {"source": "test"}, source="netsurfp")
    ds = load_probe_dataset(tmp_path)
    assert ds.concept == "ss3" and ds.source == "netsurfp" and ds.n_classes == 3
    assert len(ds.proteins) == 2
    assert ds.proteins[0].sequence == "ACDE" and ds.proteins[0].labels == [0, 0, 1, 2]


# ---------------------------------------------------------------------------
# Swiss-Prot source (HELIX/STRAND/TURN features -> SS3)
# ---------------------------------------------------------------------------

_SP_XML = """<?xml version="1.0"?>
<uniprot xmlns="https://uniprot.org/uniprot">
  <entry><accession>P00001</accession>
    <feature type="helix"><location><begin position="2"/><end position="4"/></location></feature>
    <feature type="strand"><location><begin position="6"/><end position="7"/></location></feature>
    <sequence length="8">MKTAYIAK</sequence></entry>
  <entry><accession>P00002</accession>
    <feature type="chain"><location><begin position="1"/><end position="5"/></location></feature>
    <sequence length="5">GVSER</sequence></entry>
  <entry><accession>P00003</accession>
    <feature type="helix"><location><begin position="1"/><end position="3"/></location></feature>
    <sequence length="4">WCLA</sequence></entry>
</uniprot>"""


def _write_sp(path):
    import gzip
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(_SP_XML)


def test_parse_swissprot_decodes_ss3_from_features(tmp_path) -> None:
    p = tmp_path / "sp.xml.gz"
    _write_sp(p)
    parsed = parse_swissprot(p)
    # P00002 has no helix/strand feature -> skipped; P00001, P00003 kept
    assert [acc for acc, _, _ in parsed] == ["P00001", "P00003"]
    _, seq, labels = parsed[0]
    assert seq == "MKTAYIAK"
    # helix 2-4 -> pos1-3 (0); strand 6-7 -> pos5-6 (1); everything else coil (2)
    assert labels == [2, 0, 0, 0, 2, 1, 1, 2]


def test_parse_swissprot_respects_max_proteins(tmp_path) -> None:
    p = tmp_path / "sp.xml.gz"
    _write_sp(p)
    assert len(parse_swissprot(p, max_proteins=1)) == 1
