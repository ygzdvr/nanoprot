"""Tests for homology-aware cluster splits (nanoprot/eval/probe/cluster.py).

The pure split logic is tested with a synthetic clustering (hermetic, fast). The real
mmseqs2 path is a single gated slow test, skipped when the binary is absent.
"""

import pytest

from nanoprot.eval.probe.cluster import (
    assign_splits_by_cluster,
    assign_splits_clustered,
    cluster_sequences,
    find_mmseqs,
    select_test_disjoint_train,
    split_counts_by_cluster,
)
from nanoprot.eval.probe.labels import assign_splits


def test_singleton_clusters_match_legacy_hash_split():
    # When every protein is its own cluster (rep == id), the cluster split must reduce
    # EXACTLY to the legacy per-protein hash split — same hash convention.
    ids = [f"P{i:05d}" for i in range(500)]
    singleton = {pid: pid for pid in ids}
    clustered = assign_splits_by_cluster(singleton, fracs=(0.8, 0.1, 0.1), seed=0)
    legacy = assign_splits(ids, fracs=(0.8, 0.1, 0.1), seed=0)
    assert clustered == legacy


def test_all_members_of_a_cluster_share_a_split():
    member_to_rep = {"a1": "a1", "a2": "a1", "a3": "a1",
                     "b1": "b1", "b2": "b1", "b3": "b1"}
    splits = assign_splits_by_cluster(member_to_rep, seed=3)
    assert splits["a1"] == splits["a2"] == splits["a3"]
    assert splits["b1"] == splits["b2"] == splits["b3"]


def test_split_counts_reports_no_straddle():
    member_to_rep = {"a1": "a1", "a2": "a1", "b1": "b1"}
    splits = assign_splits_by_cluster(member_to_rep, seed=1)
    diag = split_counts_by_cluster(member_to_rep, splits)
    assert diag["straddling_clusters"] == 0
    assert diag["n_clusters"] == 2
    assert sum(diag["proteins"].values()) == 3


def test_split_counts_flags_a_straddle():
    member_to_rep = {"a1": "a1", "a2": "a1"}
    bad = {"a1": "train", "a2": "test"}          # one cluster, two splits -> leakage
    diag = split_counts_by_cluster(member_to_rep, bad)
    assert diag["straddling_clusters"] == 1


def test_fracs_must_sum_to_one():
    with pytest.raises(ValueError):
        assign_splits_by_cluster({"a": "a"}, fracs=(0.5, 0.4, 0.4))


def test_assign_by_cluster_is_deterministic():
    m = {f"x{i}": f"r{i % 7}" for i in range(40)}
    assert assign_splits_by_cluster(m, seed=2) == assign_splits_by_cluster(m, seed=2)


# --- gated: real mmseqs2 clustering (slow, needs the binary) ------------------

@pytest.mark.slow
@pytest.mark.skipif(find_mmseqs() is None, reason="mmseqs2 binary not installed")
def test_cluster_sequences_groups_identical_sequences():
    seq_a = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKAL"
             "PDAQFEVVHSLAKWKR")
    seq_b = ("GGGGGGGGGGPPPPPPPPPPWWWWWWWWWWCCCCCCCCCCDDDDDDDDDDEEEEEEEEEE"
             "FFFFFFFFFFHHHHHHHHHH")
    ids = ["dup1", "dup2", "other"]
    seqs = [seq_a, seq_a, seq_b]
    m2r = cluster_sequences(ids, seqs, min_seq_id=0.3)
    assert m2r["dup1"] == m2r["dup2"]            # identical -> same cluster
    assert m2r["other"] != m2r["dup1"]           # divergent -> different cluster
    splits = assign_splits_clustered(ids, seqs, seed=0)
    assert splits["dup1"] == splits["dup2"]      # homologs never straddle


@pytest.mark.slow
@pytest.mark.skipif(find_mmseqs() is None, reason="mmseqs2 binary not installed")
def test_select_test_disjoint_train_drops_homologs():
    seq_a = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKAL"
             "PDAQFEVVHSLAKWKR")
    seq_far = ("GGGGGGGGGGPPPPPPPPPPWWWWWWWWWWCCCCCCCCCCDDDDDDDDDDEEEEEEEEEE"
               "FFFFFFFFFFHHHHHHHHHH")
    keep = select_test_disjoint_train([seq_a, seq_far], [seq_a], min_seq_id=0.3)
    assert 0 not in keep         # train[0] is the test sequence -> dropped
    assert 1 in keep             # train[1] is unrelated -> kept
