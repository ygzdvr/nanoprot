"""Unit tests for the B-GEN genome per-position label builders (option-1 concepts).

Verifies splice-transition detection, windowed-GC binning, and the strand+phase-aware
codon-frame math on toy inputs (no hg38 needed), so the labels are regression-protected.
"""
import numpy as np

from scripts.prepare_genome_probes import (
    IGNORE_INDEX, labels_exon, labels_splice, labels_gc, labels_frame,
)


def test_exon_mask():
    lab = labels_exon(10, [(2, 5)])
    assert list(map(int, lab)) == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]


def test_splice_internal_donor_acceptor():
    # exons [2,6) and [12,16) -> intron [6,12): internal donor@6, acceptor@12 (+gene-edge sites @2/@16)
    lab = labels_splice(20, [(2, 6), (12, 16)], tol=1)
    assert lab[6] == 1 and lab[5] == 1 and lab[7] == 1, "donor at exon->intron"
    assert lab[12] == 2 and lab[11] == 2 and lab[13] == 2, "acceptor at intron->exon"
    assert set(np.where(lab == 1)[0].tolist()) >= {5, 6, 7}
    assert set(np.where(lab == 2)[0].tolist()) >= {11, 12, 13}


def test_gc_low_high():
    lab = labels_gc("A" * 10 + "GC" * 5, win=2, lo=0.42, hi=0.52)
    assert lab[0] == 0 and lab[2] == 0, "A-run is low-GC"
    assert lab[-1] == 2 and lab[-3] == 2, "GC-run is high-GC"


def test_frame_plus_strand_phase():
    assert list(map(int, labels_frame(9, [(0, 9, "+", 0)]))) == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    # phase 1: pos0 is codon-2 (one base before the first complete codon)
    assert list(map(int, labels_frame(9, [(0, 9, "+", 1)]))) == [2, 0, 1, 2, 0, 1, 2, 0, 1]


def test_frame_minus_strand():
    # - strand: codon-0 sits at the genomic 3' end (the transcript 5')
    assert list(map(int, labels_frame(9, [(0, 9, "-", 0)]))) == [2, 1, 0, 2, 1, 0, 2, 1, 0]


def test_frame_ignore_outside_cds():
    lab = labels_frame(12, [(3, 9, "+", 0)])
    assert lab[0] == IGNORE_INDEX and lab[11] == IGNORE_INDEX
    assert list(map(int, lab[3:9])) == [0, 1, 2, 0, 1, 2]
