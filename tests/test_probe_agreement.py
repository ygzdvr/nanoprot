"""Label-level agreement between two SS3 sources (the §6 sanity check)."""

from __future__ import annotations

from nanoprot.eval.probe.labels import ProbeDataset, ProbeProtein
from scripts.probe_agreement import label_agreement


def _ds(source, proteins):
    return ProbeDataset(concept="ss3", source=source, n_classes=3,
                        class_names=["helix", "strand", "coil"], proteins=proteins,
                        ignore_index=-100)


def test_agreement_counts_only_shared_sequence_matched_proteins() -> None:
    a = _ds("A", [ProbeProtein("p1", "MKTA", [0, 1, 2, 0], "train"),
                  ProbeProtein("p2", "GV", [0, 1], "train")])          # not in B
    b = _ds("B", [ProbeProtein("p1", "MKTA", [0, 1, 2, 2], "train"),   # pos3 differs (0 vs 2)
                  ProbeProtein("p3", "WW", [0, 0], "train")])          # not in A
    r = label_agreement(a, b)
    assert r["n_proteins"] == 1            # only p1 is shared + sequence-identical
    assert r["n_residues"] == 4
    assert r["agreement"] == 0.75          # 3 of 4 residues agree
    assert r["confusion"][0][0] == 1 and r["confusion"][0][2] == 1   # helix kept / helix->coil


def test_agreement_skips_sequence_mismatch() -> None:
    a = _ds("A", [ProbeProtein("p1", "MKTA", [0, 1, 2, 0], "train")])
    b = _ds("B", [ProbeProtein("p1", "MKTV", [0, 1, 2, 0], "train")])  # last residue differs
    assert label_agreement(a, b)["n_proteins"] == 0


def test_agreement_skips_ignored_positions() -> None:
    a = _ds("A", [ProbeProtein("p1", "MK", [0, -100], "train")])
    b = _ds("B", [ProbeProtein("p1", "MK", [0, 1], "train")])
    r = label_agreement(a, b)
    assert r["n_residues"] == 1 and r["agreement"] == 1.0   # the -100 position is dropped
