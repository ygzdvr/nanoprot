#!/usr/bin/env python
"""
Cross-checkpoint probe driver for the trajectory-emergence experiment (Pillar 2).

For ONE trajectory model dir (``release_traj/nanoprot-{arch}-{scale}-s{seed}``) and ONE
probe dataset, probe EVERY saved checkpoint and record how learned-minus-baseline
decodability of a biological concept rises over training. Two design points from the PI
review are baked in here:

  * FIXED BASELINE. The random-init baseline is built and its probe features gathered
    ONCE per (arch,scale,seed) and reused for every checkpoint, so the Delta(t) curve is
    not contaminated by a freshly-instantiated (noisy) baseline at each step. Delta(t) =
    probe(checkpoint_t) - probe(fixed baseline).
  * PER-LAYER x PER-CHECKPOINT output, so all three layer views are derivable downstream:
    (i) best-layer-by-validation at each checkpoint, (ii) a fixed final-best layer tracked
    over time, (iii) the full layer x checkpoint heatmap.

Emits a per-(model,task) JSON with the full grid + appends a best-by-val row per
checkpoint to a flat CSV. GPU forward passes only — no training, no gradient updates to
the model.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from nanoprot.training.checkpoint import load_pretrained  # noqa: E402
from nanoprot.eval.probe.extract import relative_depths  # noqa: E402
from nanoprot.eval.probe.labels import load_probe_dataset  # noqa: E402
from nanoprot.eval.probe.run import build_random_init, gather_features  # noqa: E402
from nanoprot.eval.probe.linear import (  # noqa: E402
    evaluate_probe, evaluate_regression, fit_linear_probe, fit_linear_regression,
)

NAME_RE = re.compile(r"nanoprot-(?P<arch>gpt2|esm2|mamba)-(?P<scale>XS|S|M|L)-s(?P<seed>\d+)")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def list_checkpoint_steps(traj_dir: Path) -> List[int]:
    """Sorted saved step numbers from ``model_{step:06d}.pt`` files."""
    steps = []
    for p in glob.glob(str(Path(traj_dir) / "model_*.pt")):
        try:
            steps.append(int(Path(p).stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(steps)


def best_layer_by_val(per_layer: List[dict], metric: str) -> int:
    """Index of the layer with the highest VALIDATION metric (selection on val)."""
    return max(per_layer, key=lambda r: r["val"][metric])["layer"]


def delta_per_layer(learned: List[dict], baseline: List[dict], metric: str) -> List[float]:
    """learned_test - baseline_test at each layer (baseline is the FIXED reference)."""
    base = {r["layer"]: r["test"][metric] for r in baseline}
    return [r["test"][metric] - base.get(r["layer"], float("nan")) for r in learned]


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def _features(model, dataset, tokenizer, *, max_residues, device) -> Dict[str, tuple]:
    pool = dataset.level == "sequence"     # fold/family/fitness: mean-pool residue reps (A-BREADTH)
    return {sp: gather_features(model, dataset.split(sp), tokenizer, task=dataset.task,
                                ignore_index=dataset.ignore_index, max_residues=max_residues,
                                device=device, pool=pool)
            for sp in ("train", "val", "test")}


def _per_layer_sweep(feats: Dict[str, tuple], n_classes: int, task: str, fit_kw: dict) -> List[dict]:
    """Fit a probe per layer on train; evaluate on val + test. Returns EVERY layer (the
    caller selects views) — unlike run._sweep which keeps only the best."""
    Xtr, ytr = feats["train"]
    Xva, yva = feats["val"]
    Xte, yte = feats["test"]
    out: List[dict] = []
    for k in range(len(Xtr)):
        if task == "regression":
            probe = fit_linear_regression(Xtr[k], ytr, **fit_kw)
            val, test = evaluate_regression(probe, Xva[k], yva), evaluate_regression(probe, Xte[k], yte)
        else:
            probe = fit_linear_probe(Xtr[k], ytr, n_classes, **fit_kw)
            val = evaluate_probe(probe, Xva[k], yva, n_classes)
            test = evaluate_probe(probe, Xte[k], yte, n_classes)
        out.append({"layer": k, "val": val, "test": test})
    return out


def run_trajectory(traj_dir: Path, probe_data: Path, *, steps: Optional[List[int]] = None,
                   metric: Optional[str] = None, max_residues: Optional[int] = None,
                   device: Optional[str] = None, fit_kw: Optional[dict] = None) -> dict:
    """Probe every (or a subset of) checkpoint of one trajectory model vs a FIXED baseline."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fit_kw = fit_kw or {}
    dataset = load_probe_dataset(probe_data)
    task, n_classes = dataset.task, dataset.n_classes
    metric = metric or ("r2" if task == "regression" else "macro_f1")

    all_steps = list_checkpoint_steps(traj_dir)
    if not all_steps:
        raise SystemExit(f"no model_*.pt checkpoints in {traj_dir}")
    steps = [s for s in all_steps if (steps is None or s in steps)]

    # FIXED baseline: build the random-init model + gather its features ONCE, reuse for all
    # checkpoints. cfg comes from the last checkpoint (config is embedded in every meta).
    _, cfg, _, tok = load_pretrained(traj_dir, step=all_steps[-1], device=device, return_tokenizer=True)
    baseline_model = build_random_init(cfg.model, device=device)
    base_feats = _features(baseline_model, dataset, tok, max_residues=max_residues, device=device)
    baseline_per_layer = _per_layer_sweep(base_feats, n_classes, task, fit_kw)
    del baseline_model, base_feats
    rd = relative_depths(len(baseline_per_layer))

    checkpoints: List[dict] = []
    for step in steps:
        model, _, meta, tok = load_pretrained(traj_dir, step=step, device=device, return_tokenizer=True)
        feats = _features(model, dataset, tok, max_residues=max_residues, device=device)
        learned = _per_layer_sweep(feats, n_classes, task, fit_kw)
        del model, feats
        deltas = delta_per_layer(learned, baseline_per_layer, metric)
        bl = best_layer_by_val(learned, metric)
        checkpoints.append({
            "step": step,
            "train_residues": meta.get("total_residues"),
            "train_flops": meta.get("total_flops"),
            "best_layer_by_val": bl,
            "best_delta": deltas[bl],
            "best_learned": learned[bl]["test"][metric],
            "per_layer": [
                {"layer": l["layer"], "rel_depth": rd[l["layer"]],
                 "learned_test": l["test"][metric],
                 "baseline_test": baseline_per_layer[l["layer"]]["test"][metric],
                 "delta": deltas[l["layer"]],
                 "per_class_f1": l["test"].get("per_class_f1")}
                for l in learned
            ],
        })

    return {
        "concept": dataset.concept, "source": dataset.source, "metric": metric, "task": task,
        "class_names": dataset.class_names,
        "baseline_per_layer": [
            {"layer": b["layer"], "rel_depth": rd[b["layer"]], "test": b["test"][metric]}
            for b in baseline_per_layer
        ],
        "checkpoints": checkpoints,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, required=True,
                    help="release_traj/nanoprot-{arch}-{scale}-s{seed} (saved interval ckpts).")
    ap.add_argument("--probe-data", type=Path, required=True, help="A cached probe dataset dir.")
    ap.add_argument("--out", type=Path, required=True, help="Results CSV (appended).")
    ap.add_argument("--sidecar-dir", type=Path, default=None,
                    help="Where to write the full layer x checkpoint JSON (default: --out's dir).")
    ap.add_argument("--steps", type=int, nargs="+", default=None, help="Subset of steps (default all).")
    ap.add_argument("--metric", default=None, choices=["macro_f1", "accuracy", "r2", "spearman"])
    ap.add_argument("--max-residues", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    m = NAME_RE.search(args.traj_dir.name)
    arch = m["arch"] if m else "?"
    scale = m["scale"] if m else "?"
    seed = int(m["seed"]) if m else -1

    out = run_trajectory(
        args.traj_dir, args.probe_data, steps=args.steps, metric=args.metric,
        max_residues=args.max_residues, device=args.device,
        fit_kw={"epochs": args.epochs, "seed": args.seed, "device": args.device},
    )

    # flat CSV: one best-by-val row per checkpoint (the emergence curve's primary x/y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.out.exists()
    with args.out.open("a", newline="") as fh:
        cols = ["arch", "scale", "seed", "step", "train_residues", "train_flops",
                "concept", "source", "metric", "best_layer", "learned_test", "delta"]
        w = csv.DictWriter(fh, fieldnames=cols)
        if is_new:
            w.writeheader()
        for c in out["checkpoints"]:
            w.writerow({"arch": arch, "scale": scale, "seed": seed, "step": c["step"],
                        "train_residues": c["train_residues"], "train_flops": c["train_flops"],
                        "concept": out["concept"], "source": out["source"], "metric": out["metric"],
                        "best_layer": c["best_layer_by_val"],
                        "learned_test": round(c["best_learned"], 6), "delta": round(c["best_delta"], 6)})

    side_dir = args.sidecar_dir or args.out.parent
    side_dir.mkdir(parents=True, exist_ok=True)
    side = side_dir / f"traj_{arch}-{scale}-s{seed}_{out['source']}_{out['concept']}.json"
    side.write_text(json.dumps({"arch": arch, "scale": scale, "seed": seed, **out}, indent=2))

    n = len(out["checkpoints"])
    last = out["checkpoints"][-1] if n else {}
    print(f"  {arch}-{scale}-s{seed} [{out['source']}/{out['concept']}]: {n} ckpts, "
          f"final Δ={last.get('best_delta', float('nan')):+.4f} @L{last.get('best_layer_by_val','?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
