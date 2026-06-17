"""
Homology-aware (cluster-based) train/val/test splits for probe data.

Random per-protein splits leak homologs across train and test: a near-duplicate of a
test protein sitting in train inflates probe scores, and it is the single most common
reviewer objection in protein representation work. We instead cluster sequences at a
fixed identity threshold (mmseqs2 ``easy-cluster``, 30% by default) and assign whole
*clusters* to splits, so no two near-homologs straddle the train/test boundary.

The split assignment itself reuses the deterministic hash convention from
``labels.assign_splits`` (sha1 of ``seed:rep_id``), applied to each cluster's
representative — so the result is reproducible and order-independent, just at cluster
granularity instead of protein granularity.

``assign_splits_by_cluster`` is pure (it takes a precomputed ``member -> rep`` map) and
is unit-tested with a synthetic clustering. ``cluster_sequences`` shells out to mmseqs2
and is exercised by a gated slow test and the real cache builds.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

# Default location of the mmseqs2 static binary installed under the repo cache
# (.cache/tools/mmseqs/bin/mmseqs). cluster.py lives at nanoprot/eval/probe/, so the
# repo root is three parents up from the package dir.
_DEFAULT_MMSEQS = Path(__file__).resolve().parents[3] / ".cache" / "tools" / "mmseqs" / "bin" / "mmseqs"


def _hash_unit(key: str) -> float:
    """Deterministic ``[0, 1)`` hash — identical convention to ``labels.assign_splits``."""
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)
    return (h % 1_000_000) / 1_000_000.0


def assign_splits_by_cluster(member_to_rep: Dict[str, str],
                             fracs: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                             seed: int = 0) -> Dict[str, str]:
    """Assign each protein to train/val/test **by its cluster**, deterministically.

    Every member of a cluster receives the same split (we hash the cluster
    representative id, not the member id), so homologs never straddle the split
    boundary. Pure and testable — the clustering itself is computed elsewhere.
    """
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"fracs must sum to 1, got {fracs}")
    tr, va = fracs[0], fracs[0] + fracs[1]
    out: Dict[str, str] = {}
    for member, rep in member_to_rep.items():
        r = _hash_unit(f"{seed}:{rep}")
        out[member] = "train" if r < tr else ("val" if r < va else "test")
    return out


def find_mmseqs(mmseqs_bin: Optional[str] = None) -> Optional[str]:
    """Locate the mmseqs binary (explicit arg > ``$MMSEQS_BIN`` > repo cache > PATH)."""
    for cand in (mmseqs_bin, os.environ.get("MMSEQS_BIN"), str(_DEFAULT_MMSEQS)):
        if cand and Path(cand).exists():
            return cand
    return shutil.which("mmseqs")


def cluster_sequences(ids: Sequence[str], seqs: Sequence[str], *,
                      min_seq_id: float = 0.3, coverage: float = 0.8, cov_mode: int = 0,
                      mmseqs_bin: Optional[str] = None, tmp_dir: Optional[str] = None,
                      threads: int = 4) -> Dict[str, str]:
    """Cluster sequences with mmseqs2 ``easy-cluster``; return ``member_id -> rep_id``.

    ``min_seq_id`` is the sequence-identity threshold (0.3 = 30%, the conventional
    homology-reduction cutoff for secondary-structure / PLM probing). Raises
    ``FileNotFoundError`` if mmseqs is unavailable, so callers can decide whether to fall
    back to per-protein splits.
    """
    if len(ids) != len(seqs):
        raise ValueError("ids and seqs must be the same length")
    binp = find_mmseqs(mmseqs_bin)
    if binp is None:
        raise FileNotFoundError(
            "mmseqs2 not found — set $MMSEQS_BIN or install the static binary to "
            ".cache/tools/mmseqs (see runs/ or the repo README).")
    owns_tmp = tmp_dir is None
    tmp = Path(tmp_dir or tempfile.mkdtemp(prefix="mmseqs_clu_"))
    try:
        fasta = tmp / "in.fasta"
        with open(fasta, "w") as fh:
            for pid, seq in zip(ids, seqs):
                fh.write(f">{pid}\n{seq}\n")
        res = tmp / "clu"
        cmd = [binp, "easy-cluster", str(fasta), str(res), str(tmp / "work"),
               "--min-seq-id", str(min_seq_id), "-c", str(coverage),
               "--cov-mode", str(cov_mode), "--threads", str(threads), "-v", "1"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        member_to_rep: Dict[str, str] = {}
        with open(f"{res}_cluster.tsv") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rep, member = line.rstrip("\n").split("\t")
                member_to_rep[member] = rep
        for pid in ids:                       # any id mmseqs dropped maps to itself
            member_to_rep.setdefault(pid, pid)
        return member_to_rep
    finally:
        if owns_tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def assign_splits_clustered(ids: Sequence[str], seqs: Sequence[str],
                            fracs: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                            seed: int = 0, **cluster_kw) -> Dict[str, str]:
    """Cluster sequences, then assign whole clusters to train/val/test (convenience)."""
    member_to_rep = cluster_sequences(ids, seqs, **cluster_kw)
    return assign_splits_by_cluster(member_to_rep, fracs=fracs, seed=seed)


def split_counts_by_cluster(member_to_rep: Dict[str, str], splits: Dict[str, str]) -> dict:
    """Diagnostics for a clustered split: protein counts, cluster counts, and a check
    that no cluster's members straddle two splits (the property that prevents leakage)."""
    rep_split: Dict[str, str] = {}
    straddling = set()
    for member, rep in member_to_rep.items():
        s = splits[member]
        if rep in rep_split and rep_split[rep] != s:
            straddling.add(rep)
        rep_split[rep] = s
    prot_counts: Dict[str, int] = {}
    for s in splits.values():
        prot_counts[s] = prot_counts.get(s, 0) + 1
    clu_counts: Dict[str, int] = {}
    for s in rep_split.values():
        clu_counts[s] = clu_counts.get(s, 0) + 1
    return {"proteins": prot_counts, "clusters": clu_counts,
            "n_clusters": len(rep_split), "straddling_clusters": len(straddling)}
