"""
Probe-label loading + token alignment for the cross-architecture probing harness.

This is the data interface shared by ``prepare_probe_data.py`` (writer) and
``run_probes.py`` (reader). A cached probe dataset is a directory with:

    meta.json   {"concept","source","n_classes","class_names","ignore_index",...}
    data.jsonl  one JSON per protein: {"id","sequence","labels","split"}

``labels`` is a per-residue integer label list, ``len(labels) == len(sequence)``,
each value in ``[0, n_classes)`` or ``ignore_index`` for unlabelled residues.

**Token alignment (the correctness-critical part).** The esm2 alphabet is
character-level and 1:1 with residues. The *training* loader tokenises every document
as ``encode(seq, prepend=<cls>)`` with no append (see
``nanoprot/data/dataloader.py``), i.e. the model always saw ``[<cls>, r0, r1, ...]``.
So the probe matches that exactly: prepend ``<cls>`` (mask it with ``ignore_index``),
then residue *i* lands at token position *i + 1*. A hard check rejects any tokenizer
that is not 1:1 (e.g. a BPE tokenizer), which would silently break the alignment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

DEFAULT_IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Cached dataset
# ---------------------------------------------------------------------------

@dataclass
class ProbeProtein:
    id: str
    sequence: str
    labels: List[int]
    split: str


@dataclass
class ProbeDataset:
    concept: str
    source: str
    n_classes: int
    class_names: List[str]
    proteins: List[ProbeProtein]
    ignore_index: int = DEFAULT_IGNORE_INDEX
    meta: dict = field(default_factory=dict)

    def split(self, name: str) -> List[ProbeProtein]:
        """Proteins in a given split (``"train"`` / ``"val"`` / ``"test"``)."""
        return [p for p in self.proteins if p.split == name]

    def split_sizes(self) -> Dict[str, int]:
        sizes: Dict[str, int] = {}
        for p in self.proteins:
            sizes[p.split] = sizes.get(p.split, 0) + 1
        return sizes


def load_probe_dataset(path: Union[str, Path]) -> ProbeDataset:
    """Load a cached probe dataset directory, validating shapes + label ranges."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    n_classes = int(meta["n_classes"])
    ignore = int(meta.get("ignore_index", DEFAULT_IGNORE_INDEX))

    proteins: List[ProbeProtein] = []
    with (path / "data.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            p = ProbeProtein(d["id"], d["sequence"], list(d["labels"]), d["split"])
            if len(p.sequence) != len(p.labels):
                raise ValueError(
                    f"protein {p.id}: {len(p.sequence)} residues vs {len(p.labels)} labels"
                )
            for lab in p.labels:
                if lab != ignore and not (0 <= lab < n_classes):
                    raise ValueError(
                        f"protein {p.id}: label {lab} outside [0,{n_classes}) and != ignore({ignore})"
                    )
            proteins.append(p)
    return ProbeDataset(
        concept=meta["concept"], source=meta["source"], n_classes=n_classes,
        class_names=list(meta["class_names"]), ignore_index=ignore,
        proteins=proteins, meta=meta,
    )


# ---------------------------------------------------------------------------
# Token alignment
# ---------------------------------------------------------------------------

def tokenize_and_align(sequence: str, labels: Sequence[int], tokenizer, *,
                       add_bos: bool = True,
                       ignore_index: int = DEFAULT_IGNORE_INDEX) -> Tuple[List[int], List[int]]:
    """Tokenise one protein and return ``(token_ids, aligned_labels)`` of equal length.

    With ``add_bos=True`` (the training convention) a ``<cls>`` token is prepended and
    its label set to ``ignore_index``; residue *i* then sits at token position *i + 1*.
    Raises if the tokenizer is not 1:1 with residues (which would break the alignment).
    """
    if len(sequence) != len(labels):
        raise ValueError(f"{len(sequence)} residues vs {len(labels)} labels")
    res_ids = tokenizer.encode(sequence)[0]
    if len(res_ids) != len(sequence):
        raise ValueError(
            f"tokenizer is not 1:1 ({len(res_ids)} tokens for {len(sequence)} residues) — "
            "per-residue probing requires the character-level esm2 alphabet."
        )
    aligned = list(labels)
    if add_bos:
        return [tokenizer.get_bos_token_id(), *res_ids], [ignore_index, *aligned]
    return list(res_ids), aligned


def encode_protein(protein: ProbeProtein, tokenizer, *, add_bos: bool = True,
                   ignore_index: int = DEFAULT_IGNORE_INDEX):
    """``tokenize_and_align`` for one :class:`ProbeProtein`, returned as ``(1, T)`` LongTensors
    ready for ``extract_layers(model, ids)`` and ``flatten_residues(feats, labels)``."""
    import torch
    ids, aligned = tokenize_and_align(
        protein.sequence, protein.labels, tokenizer,
        add_bos=add_bos, ignore_index=ignore_index,
    )
    return (torch.tensor([ids], dtype=torch.long),
            torch.tensor([aligned], dtype=torch.long))


def default_tokenizer():
    """The shared 33-token esm2 alphabet tokenizer (all archs use it)."""
    from nanoprot.tokenizers.esm2 import get_esm2_tokenizer
    return get_esm2_tokenizer()


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def assign_splits(ids: Sequence[str], fracs: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                  seed: int = 0) -> Dict[str, str]:
    """Deterministic protein-level train/val/test assignment.

    Splits by hashing each protein id (not by shuffling), so the assignment is stable
    regardless of input order and reproducible across runs. For sources without a
    canonical split (Swiss-Prot, DSSP); the published benchmark keeps its own split.
    """
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"fracs must sum to 1, got {fracs}")
    tr, va = fracs[0], fracs[0] + fracs[1]
    out: Dict[str, str] = {}
    for pid in ids:
        h = int(hashlib.sha1(f"{seed}:{pid}".encode("utf-8")).hexdigest(), 16)
        r = (h % 1_000_000) / 1_000_000.0
        out[pid] = "train" if r < tr else ("val" if r < va else "test")
    return out
