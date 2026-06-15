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
                source: str, ignore_index: int = -100) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps({
        "concept": "ss3", "source": source, "n_classes": 3,
        "class_names": SS3_NAMES, "ignore_index": ignore_index,
        "provenance": provenance,
    }, indent=2))
    with (out_dir / "data.jsonl").open("w", encoding="utf-8") as fh:
        for p in proteins:
            fh.write(json.dumps(p) + "\n")
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))


def _download_url(url: str, dest: Path) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (login-node, trusted host)


def _download(name: str, dest: Path) -> None:
    _download_url(f"{_DTU_BASE}/{name}", dest)


# ---------------------------------------------------------------------------
# Swiss-Prot source (experimental-structure SS3 from HELIX/STRAND/TURN features)
# ---------------------------------------------------------------------------

_SP_NS = "{http://uniprot.org/uniprot}"
_SP_URL = ("https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
           "knowledgebase/complete/uniprot_sprot.xml.gz")
_SP_SS = {"helix": 0, "strand": 1, "turn": 2}   # turn folds into coil (2)


def parse_swissprot(path, max_proteins=None) -> List[Tuple[str, str, List[int]]]:
    """Stream Swiss-Prot XML(.gz), yielding ``(accession, sequence, ss3_labels)`` for
    proteins with >=1 helix/strand feature.

    Convention (the standard SS3-from-Swiss-Prot): helix->0, strand->1, and every other
    residue->2 (coil), which assumes an annotated protein is structurally characterised
    (the §6 caveat — Swiss-Prot SS annotations come from experimental structures and are
    biased toward well-studied proteins). Memory-bounded via incremental parsing.
    """
    import gzip
    import xml.etree.ElementTree as ET
    opener = gzip.open if str(path).endswith(".gz") else open
    out: List[Tuple[str, str, List[int]]] = []
    with opener(path, "rb") as fh:
        context = ET.iterparse(fh, events=("start", "end"))
        _, root = next(context)
        for event, elem in context:
            if event != "end" or elem.tag != _SP_NS + "entry":
                continue
            acc = elem.findtext(_SP_NS + "accession")
            seq_el = elem.find(_SP_NS + "sequence")
            seq = "".join((seq_el.text or "").split()) if seq_el is not None else ""
            spans, has_helix_strand = [], False
            for feat in elem.findall(_SP_NS + "feature"):
                ftype = feat.get("type")
                if ftype not in _SP_SS:
                    continue
                loc = feat.find(_SP_NS + "location")
                b_el = loc.find(_SP_NS + "begin") if loc is not None else None
                e_el = loc.find(_SP_NS + "end") if loc is not None else None
                if b_el is None or e_el is None or not b_el.get("position") or not e_el.get("position"):
                    continue
                spans.append((int(b_el.get("position")), int(e_el.get("position")), _SP_SS[ftype]))
                has_helix_strand = has_helix_strand or ftype in ("helix", "strand")
            if acc and seq and has_helix_strand:
                labels = [2] * len(seq)                      # default coil
                for b, e, ss in spans:
                    for i in range(b - 1, min(e, len(seq))):  # 1-indexed inclusive -> 0-indexed
                        labels[i] = ss
                out.append((acc, seq, labels))
            root.clear()                                      # free processed entries
            if max_proteins is not None and len(out) >= max_proteins:
                break
    return out


def build_swissprot(args) -> Tuple[List[dict], dict]:
    path = args.swissprot_file
    if path is None:
        raise SystemExit("--swissprot-file is required for --source swissprot (uniprot_sprot.xml.gz).")
    if not path.exists():
        if args.download:
            _download_url(_SP_URL, path)
        else:
            raise SystemExit(f"missing {path}. Download on a login node:\n"
                             f"  curl -o {path} {_SP_URL}\nor re-run with --download.")
    parsed = parse_swissprot(path, max_proteins=args.max_proteins)
    if not parsed:
        raise SystemExit("no Swiss-Prot proteins with helix/strand features parsed.")
    # No canonical split -> hashed 3-way (train / val / test).
    splits = assign_splits([acc for acc, _, _ in parsed],
                           fracs=(1 - 2 * args.val_frac, args.val_frac, args.val_frac),
                           seed=args.seed)
    proteins = [{"id": acc, "sequence": seq, "labels": labels, "split": splits[acc]}
                for acc, seq, labels in parsed]
    prov = {"source": "Swiss-Prot (UniProtKB reviewed) HELIX/STRAND/TURN features",
            "url": _SP_URL, "convention": "helix->0, strand->1, else->2 (coil)",
            "max_proteins": args.max_proteins, "val_frac": args.val_frac, "seed": args.seed}
    return proteins, prov


# ---------------------------------------------------------------------------
# NetSurfP source (extracted so main() can dispatch by --source)
# ---------------------------------------------------------------------------

def build_netsurfp(args) -> Tuple[List[dict], dict]:
    if args.netsurfp_dir is None:
        raise SystemExit("--netsurfp-dir is required for --source netsurfp.")
    train_path = args.netsurfp_dir / args.train_file
    test_path = args.netsurfp_dir / args.test_file
    for name, path in ((args.train_file, train_path), (args.test_file, test_path)):
        if not path.exists():
            if args.download:
                _download(name, path)
            else:
                raise SystemExit(f"missing {path}. Download on a login node:\n"
                                 f"  curl -o {path} {_DTU_BASE}/{name}\nor re-run with --download.")
    train_data, train_ids = load_npz(train_path)
    test_data, test_ids = load_npz(test_path)
    train_parsed = parse_netsurfp(train_data, train_ids, use_eval_mask=False)
    test_parsed = parse_netsurfp(test_data, test_ids, use_eval_mask=True)  # CB513 eval mask
    tv = assign_splits([pid for pid, _, _ in train_parsed],
                       fracs=(1.0 - args.val_frac, args.val_frac, 0.0), seed=args.seed)
    proteins = [{"id": pid, "sequence": seq, "labels": labels, "split": tv[pid]}
                for pid, seq, labels in train_parsed]
    proteins += [{"id": pid, "sequence": seq, "labels": labels, "split": "test"}
                 for pid, seq, labels in test_parsed]
    bad = [p["id"] for p in proteins[:50] if set(p["sequence"]) - set(NSP_AA_ORDER + "X")]
    if bad:
        raise SystemExit(f"decoded non-AA characters in {bad} — check NSP_AA_ORDER.")
    prov = {"source": "NetSurfP-2.0 (Klausen et al. 2019)", "base_url": _DTU_BASE,
            "train_file": args.train_file, "test_file": args.test_file,
            "aa_order": NSP_AA_ORDER, "q8_order": NSP_Q8_ORDER, "q8_to_ss3": Q8_TO_SS3,
            "val_frac": args.val_frac, "seed": args.seed}
    return proteins, prov


# ---------------------------------------------------------------------------
# DSSP-from-AlphaFold source (predicted-structure SS3 via biotite, pLDDT-filtered)
# ---------------------------------------------------------------------------

_AF_BASE = "https://alphafold.ebi.ac.uk/files"
_AF_MODEL = "model_v6"                        # current AlphaFold DB model version
_SSE_TO_SS3 = {"a": 0, "b": 1, "c": 2}        # biotite annotate_sse: alpha/beta/coil


def sse_to_ss3(sse, plddt, plddt_min: float, ignore_index: int = -100) -> List[int]:
    """Map biotite SSE codes (a/b/c) -> SS3 (helix/strand/coil), masking low-pLDDT residues."""
    return [(_SSE_TO_SS3.get(str(s), 2) if float(p) >= plddt_min else ignore_index)
            for s, p in zip(sse, plddt)]


def _structure_to_ss3(pdb_path, plddt_min: float):
    """Load an AlphaFold PDB and return ``(sequence, ss3_labels)`` (pLDDT-filtered), or
    ``None`` if the structure is unusable. SS is biotite's P-SEA 3-state annotation."""
    from biotite.sequence import ProteinSequence
    from biotite.structure import annotate_sse, filter_amino_acids, get_residues
    from biotite.structure.io.pdb import PDBFile
    arr = PDBFile.read(str(pdb_path)).get_structure(model=1, extra_fields=["b_factor"])
    arr = arr[filter_amino_acids(arr)]
    if arr.array_length() == 0:
        return None
    sse = annotate_sse(arr)
    ca = arr[arr.atom_name == "CA"]
    _, res_names = get_residues(arr)
    if not (len(sse) == ca.array_length() == len(res_names)):
        return None                            # mismatched residue accounting -> skip
    seq = []
    for r in res_names:
        try:
            seq.append(ProteinSequence.convert_letter_3to1(r))
        except Exception:
            seq.append("X")
    return "".join(seq), sse_to_ss3(sse, ca.b_factor, plddt_min)


def _dssp_accessions(args) -> List[str]:
    if args.from_swissprot is not None:
        import json as _json
        ids = []
        with (args.from_swissprot / "data.jsonl").open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    ids.append(_json.loads(line)["id"])
        return ids
    if args.accessions_file is not None:
        return [ln.strip() for ln in args.accessions_file.read_text().splitlines() if ln.strip()]
    raise SystemExit("--source dssp needs --from-swissprot DIR or --accessions-file FILE.")


def build_dssp(args) -> Tuple[List[dict], dict]:
    accs = _dssp_accessions(args)
    af_dir = args.af_dir or (args.out.parent / "af_pdb")
    af_dir.mkdir(parents=True, exist_ok=True)
    parsed, n_missing, n_bad = [], 0, 0
    for acc in accs:
        if len(parsed) >= args.max_proteins:
            break
        pdb_path = af_dir / f"AF-{acc}-F1-{_AF_MODEL}.pdb"
        if not pdb_path.exists():
            try:
                _download_url(f"{_AF_BASE}/AF-{acc}-F1-{_AF_MODEL}.pdb", pdb_path)
            except Exception:
                n_missing += 1
                continue
        try:
            res = _structure_to_ss3(pdb_path, args.plddt_min)
        except Exception:
            n_bad += 1
            continue
        if res is None or not any(lab != -100 for lab in res[1]):
            n_bad += 1
            continue
        parsed.append((acc, res[0], res[1]))
    if not parsed:
        raise SystemExit(f"no usable AlphaFold structures ({n_missing} missing, {n_bad} unusable).")
    splits = assign_splits([a for a, _, _ in parsed],
                           fracs=(1 - 2 * args.val_frac, args.val_frac, args.val_frac), seed=args.seed)
    proteins = [{"id": a, "sequence": s, "labels": l, "split": splits[a]} for a, s, l in parsed]
    prov = {"source": "DSSP-from-AlphaFold (biotite P-SEA 3-state on predicted structure)",
            "af_base": _AF_BASE, "af_model": _AF_MODEL, "plddt_min": args.plddt_min,
            "n_missing": n_missing, "n_unusable": n_bad,
            "max_proteins": args.max_proteins, "val_frac": args.val_frac, "seed": args.seed}
    return proteins, prov


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="netsurfp", choices=["netsurfp", "swissprot", "dssp"])
    # netsurfp
    ap.add_argument("--netsurfp-dir", type=Path, default=None,
                    help="Directory holding Train_HHblits.npz + CB513_HHblits.npz.")
    ap.add_argument("--train-file", default="Train_HHblits.npz")
    ap.add_argument("--test-file", default="CB513_HHblits.npz")
    # swissprot
    ap.add_argument("--swissprot-file", type=Path, default=None,
                    help="uniprot_sprot.xml.gz (for --source swissprot).")
    ap.add_argument("--max-proteins", type=int, default=6000,
                    help="Cap for the swissprot/dssp parse (proteins with SS features).")
    # dssp (DSSP-from-AlphaFold)
    ap.add_argument("--from-swissprot", type=Path, default=None,
                    help="A swissprot cache dir; its accessions are reused (gives protein "
                         "overlap for the label-agreement analysis).")
    ap.add_argument("--accessions-file", type=Path, default=None,
                    help="Text file of UniProt accessions (one per line) for --source dssp.")
    ap.add_argument("--af-dir", type=Path, default=None, help="Where to cache AlphaFold PDBs.")
    ap.add_argument("--plddt-min", type=float, default=70.0,
                    help="Mask residues with AlphaFold pLDDT below this (default 70).")
    # common
    ap.add_argument("--download", action="store_true",
                    help="Fetch missing inputs (LOGIN NODE — compute nodes have no internet).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.source == "netsurfp":
        proteins, prov = build_netsurfp(args)
    elif args.source == "swissprot":
        proteins, prov = build_swissprot(args)
    else:
        proteins, prov = build_dssp(args)

    counts: dict = {}
    n_labelled = 0
    for p in proteins:
        counts[p["split"]] = counts.get(p["split"], 0) + 1
        n_labelled += sum(1 for lab in p["labels"] if lab != -100)
    prov = {**prov, "split_counts": counts, "n_labelled_residues": n_labelled}
    write_cache(args.out, proteins, prov, source=args.source)
    print(f"  wrote {len(proteins)} proteins ({args.source}) to {args.out}  "
          f"splits={counts}  labelled_residues={n_labelled:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
