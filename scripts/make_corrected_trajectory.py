#!/usr/bin/env python
"""Write CORRECTED-FLOP copies of the trajectory CSVs (fixes the mamba scan seq_len bug in the data).

Each cell's train_flops (a per-run total, = fpt_buggy * total_residues) is multiplied by the gate-verified
correction ratio fpt_corrected/fpt_buggy (mamba only; gpt2 unchanged). Since the iso-FLOP axis is
C(t)=train_flops*step/train_residues = fpt*step, scaling train_flops by the ratio yields the corrected
axis. val_loss.csv has no train_flops (its compute axis is rebuilt from the concept files' meta via
load_meta), so it is copied verbatim. Output dirs feed rank_reversal / forecast_protocol / crossover
unchanged, so any change in their results is due ONLY to the corrected FLOPs.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import statistics as st
from collections import defaultdict
from pathlib import Path

from scripts.reversal_corrected_flops import corrected_fpt_lookup


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, default=Path(".cache/nanoprot/release"))
    ap.add_argument("--dirs", nargs="+", default=[".cache/nanoprot/trajectory_results",
                                                  ".cache/nanoprot/trajectory_results_genome"])
    args = ap.parse_args()

    lut, gate_err = corrected_fpt_lookup(args.release)
    print(f"  correction lookup gate: max |recon-logged|/logged = {gate_err:.4%} "
          f"({'PASS' if gate_err < 0.02 else 'FAIL — abort'})")
    if gate_err >= 0.02:
        return 1
    # ratio[(arch,scale)] = fpt_corr/fpt_buggy, seed-averaged (identical across seeds by construction)
    ratio = {}
    byas = defaultdict(list)
    for (a, s, seed), (fb, fc) in lut.items():
        byas[(a, s)].append(fc / fb)
    for k, vs in byas.items():
        ratio[k] = st.mean(vs)
    print("  correction ratio (train_flops *= ratio):")
    for k in sorted(ratio):
        print(f"    {k[0]:5s} {k[1]}: {ratio[k]:.4f}")

    for d in args.dirs:
        src = Path(d)
        if not src.exists():
            continue
        dst = Path(str(src) + "_corrected")
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        n_files, n_scaled = 0, 0
        for fn in sorted(src.glob("*.csv")):
            rows = list(csv.DictReader(open(fn)))
            if fn.name == "val_loss.csv" or not rows or "train_flops" not in rows[0]:
                shutil.copy(fn, dst / fn.name); n_files += 1; continue
            for r in rows:
                k = (r["arch"], r["scale"])
                if k in ratio and ratio[k] != 1.0:
                    r["train_flops"] = f"{float(r['train_flops']) * ratio[k]:.6e}"; n_scaled += 1
            with (dst / fn.name).open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
            n_files += 1
        # copy any non-csv (json trajectory dumps) so downstream loaders that glob them still work
        for fn in src.glob("*.json"):
            shutil.copy(fn, dst / fn.name)
        print(f"  {src.name} -> {dst.name}: {n_files} csv, {n_scaled} mamba rows rescaled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
