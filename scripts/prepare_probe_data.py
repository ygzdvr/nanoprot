#!/usr/bin/env python
"""
Build a cached SS3 probe dataset (the writer for nanoprot.eval.probe.labels).

Phase 1 implements ``--source netsurfp``: the published NetSurfP-2.0 benchmark
(Klausen et al. 2019), which gives a fixed Train + CB513-test split directly
comparable to ESM-2 / prior PLM secondary-structure numbers. (``swissprot`` and
``dssp`` sources — the other two legs of the §6 triangulation — come next.)

NetSurfP-2.0 .npz layout (confirmed from the DTU dataset page):
  3-D array (n_samples, seq_position, features); features:
    0-19  : amino-acid one-hot, order ``ACDEFGHIKLMNPQRSTVWY`` (all-zero = unknown)
    50    : sequence mask (1 = real residue)
    52    : evaluation mask (CB513 only — which residues to score)
    57-64 : Q8 secondary structure one-hot, order ``GHIBESTC``
We decode the sequence + Q8, map Q8->SS3 (helix{G,H,I}/strand{E,B}/coil{S,T,C}),
honour the CB513 eval mask, and write meta.json + data.jsonl + provenance.json.

Files (download on a LOGIN node; compute nodes have no internet):
  https://services.healthtech.dtu.dk/services/NetSurfP-2.0/training_data/Train_HHblits.npz
  https://services.healthtech.dtu.dk/services/NetSurfP-2.0/training_data/CB513_HHblits.npz

Usage:
  python -m scripts.prepare_probe_data --source netsurfp \
      --netsurfp-dir $NANOPROT_BASE_DIR/probes/raw [--download] \
      --out $NANOPROT_BASE_DIR/probes/ss3_netsurfp --val-frac 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nanoprot.eval.probe.labels import assign_splits

# -- NetSurfP-2.0 feature layout (confirmed against the DTU dataset page) --------
NSP_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"   # channels 0-19
NSP_Q8_ORDER = "GHIBESTC"              # channels 57-64
CH_AA = slice(0, 20)
CH_SEQ_MASK = 50
CH_EVAL_MASK = 52
CH_Q8 = slice(57, 65)
MIN_FEATURES = 65

# Standard Q8 -> SS3: helix {G,H,I}, strand {E,B}, coil {S,T,C}.
Q8_TO_SS3 = {"G": 0, "H": 0, "I": 0, "E": 1, "B": 1, "S": 2, "T": 2, "C": 2}
SS3_NAMES = ["helix", "strand", "coil"]

_DTU_BASE = "https://services.healthtech.dtu.dk/services/NetSurfP-2.0/training_data"


# ---------------------------------------------------------------------------
# Parsing (pure, testable)
# ---------------------------------------------------------------------------

def parse_netsurfp(data: np.ndarray, ids: Optional[np.ndarray] = None, *,
                   use_eval_mask: bool = False,
                   ignore_index: int = -100) -> List[Tuple[str, str, List[int]]]:
    """Decode a NetSurfP-2.0 feature array into ``(id, sequence, ss3_labels)`` per protein.

    ``use_eval_mask`` (CB513) restricts labelled positions to the evaluation mask;
    unknown residues / unscored positions get ``ignore_index``. ``len(labels) ==
    len(sequence)`` always.
    """
    if data.ndim != 3 or data.shape[2] < MIN_FEATURES:
        raise ValueError(f"expected (N, L, >={MIN_FEATURES}) array, got {data.shape}")
    out: List[Tuple[str, str, List[int]]] = []
    for i in range(data.shape[0]):
        sample = data[i]
        seq_mask = sample[:, CH_SEQ_MASK] > 0.5
        if not seq_mask.any():
            continue
        rows = sample[seq_mask]
        aa = rows[:, CH_AA]
        q8 = rows[:, CH_Q8]
        aa_known = aa.sum(axis=1) > 0.5
        aa_idx = aa.argmax(axis=1)
        seq = "".join(NSP_AA_ORDER[j] if k else "X" for j, k in zip(aa_idx, aa_known))
        q8_known = q8.sum(axis=1) > 0.5
        q8_idx = q8.argmax(axis=1)
        labels = [Q8_TO_SS3[NSP_Q8_ORDER[j]] if k else ignore_index
                  for j, k in zip(q8_idx, q8_known)]
        if use_eval_mask:
            ev = rows[:, CH_EVAL_MASK] > 0.5
            labels = [lab if e else ignore_index for lab, e in zip(labels, ev)]
        pid = str(ids[i]) if ids is not None else f"netsurfp_{i}"
        out.append((pid, seq, labels))
    return out


def load_npz(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a NetSurfP npz, returning ``(data_3d, ids_or_None)`` robustly (the key
    names are not fixed: we take the single 3-D array and any 1-D string array)."""
    npz = np.load(path, allow_pickle=True)
    data = None
    ids = None
    for key in npz.files:
        arr = npz[key]
        if arr.ndim == 3 and data is None:
            data = arr
        elif arr.ndim == 1 and ids is None:
            ids = arr
    if data is None:
        raise ValueError(f"no 3-D feature array found in {path} (keys: {npz.files})")
    return data, ids


# ---------------------------------------------------------------------------
# Cache writing
# ---------------------------------------------------------------------------

def write_cache(out_dir: Path, proteins: List[dict], provenance: dict,
                ignore_index: int = -100) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps({
        "concept": "ss3", "source": "netsurfp", "n_classes": 3,
        "class_names": SS3_NAMES, "ignore_index": ignore_index,
        "provenance": provenance,
    }, indent=2))
    with (out_dir / "data.jsonl").open("w", encoding="utf-8") as fh:
        for p in proteins:
            fh.write(json.dumps(p) + "\n")
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))


def _download(name: str, dest: Path) -> None:
    import urllib.request
    url = f"{_DTU_BASE}/{name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (login-node, trusted host)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="netsurfp", choices=["netsurfp", "swissprot", "dssp"])
    ap.add_argument("--netsurfp-dir", type=Path, default=None,
                    help="Directory holding Train_HHblits.npz + CB513_HHblits.npz.")
    ap.add_argument("--train-file", default="Train_HHblits.npz")
    ap.add_argument("--test-file", default="CB513_HHblits.npz")
    ap.add_argument("--download", action="store_true",
                    help="Fetch missing npz from DTU (LOGIN NODE — compute nodes have no internet).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.1, help="Val fraction carved from Train.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.source != "netsurfp":
        raise SystemExit(f"--source {args.source} not implemented yet (Phase 1 = netsurfp).")
    if args.netsurfp_dir is None:
        raise SystemExit("--netsurfp-dir is required for --source netsurfp.")

    train_path = args.netsurfp_dir / args.train_file
    test_path = args.netsurfp_dir / args.test_file
    for name, path in ((args.train_file, train_path), (args.test_file, test_path)):
        if not path.exists():
            if args.download:
                _download(name, path)
            else:
                raise SystemExit(
                    f"missing {path}. Download on a login node:\n"
                    f"  curl -o {path} {_DTU_BASE}/{name}\n"
                    f"or re-run with --download.")

    # Train -> train/val (hashed), CB513 -> test.
    train_data, train_ids = load_npz(train_path)
    test_data, test_ids = load_npz(test_path)
    train_parsed = parse_netsurfp(train_data, train_ids, use_eval_mask=False)
    test_parsed = parse_netsurfp(test_data, test_ids, use_eval_mask=True)  # CB513 eval mask

    tv = assign_splits([pid for pid, _, _ in train_parsed],
                       fracs=(1.0 - args.val_frac, args.val_frac, 0.0), seed=args.seed)
    proteins: List[dict] = []
    for pid, seq, labels in train_parsed:
        proteins.append({"id": pid, "sequence": seq, "labels": labels, "split": tv[pid]})
    for pid, seq, labels in test_parsed:
        proteins.append({"id": pid, "sequence": seq, "labels": labels, "split": "test"})

    # sanity: sequences should look like real proteins
    bad = [p["id"] for p in proteins[:50] if set(p["sequence"]) - set(NSP_AA_ORDER + "X")]
    if bad:
        raise SystemExit(f"decoded non-AA characters in {bad} — check NSP_AA_ORDER.")

    counts: dict = {}
    n_labelled = 0
    for p in proteins:
        counts[p["split"]] = counts.get(p["split"], 0) + 1
        n_labelled += sum(1 for lab in p["labels"] if lab != -100)
    provenance = {
        "source": "NetSurfP-2.0 (Klausen et al. 2019)", "base_url": _DTU_BASE,
        "train_file": args.train_file, "test_file": args.test_file,
        "aa_order": NSP_AA_ORDER, "q8_order": NSP_Q8_ORDER, "q8_to_ss3": Q8_TO_SS3,
        "val_frac": args.val_frac, "seed": args.seed,
        "split_counts": counts, "n_labelled_residues": n_labelled,
    }
    write_cache(args.out, proteins, provenance)
    print(f"  wrote {len(proteins)} proteins to {args.out}  splits={counts}  "
          f"labelled_residues={n_labelled:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
