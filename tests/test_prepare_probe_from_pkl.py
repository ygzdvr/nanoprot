"""Tests for prepare_probe_from_pkl pure helpers (pickle -> nanoprot probe cache)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.prepare_probe_from_pkl import label_key, sample_records  # noqa: E402


def test_label_key_finds_the_residue_array():
    rec = {"accession": "P1", "sequence": "MKT", "active": np.array([0, 1, 0])}
    assert label_key(rec) == "active"


def test_label_key_skips_pfam_metadata():
    rec = {"accession": "P1", "sequence": "MKT", "disorder": np.array([0, 0, 1]),
           "pfam_ids": ["PF1"], "pfam_names": ["x"]}
    assert label_key(rec) == "disorder"


def test_sample_prefers_positive_containing_proteins():
    recs = [{"accession": f"P{i}", "sequence": "M", "active": np.array([0])} for i in range(5)]
    recs += [{"accession": f"Q{i}", "sequence": "M", "active": np.array([1])} for i in range(3)]
    out = sample_records(recs, "active", max_proteins=3)
    assert len(out) == 3
    assert all(r["accession"].startswith("Q") for r in out)   # positives chosen first


def test_sample_pads_with_negatives_when_few_positives():
    recs = [{"accession": "Q0", "sequence": "M", "active": np.array([1])}]
    recs += [{"accession": f"P{i}", "sequence": "M", "active": np.array([0])} for i in range(5)]
    out = sample_records(recs, "active", max_proteins=3)
    assert len(out) == 3
    assert out[0]["accession"] == "Q0"                        # the lone positive, first
