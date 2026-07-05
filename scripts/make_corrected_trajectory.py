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


def cfpt_from_release(release_dir: Path):
    """cfpt[(arch,scale)] = corrected flops/token, averaged over seeds (deterministic per arch/scale).
    Returns (cfpt, gate_err). We OVERWRITE train_flops := cfpt*train_residues downstream (not a ratio-
    multiply), so the result is correct whether a row logged BUGGY flops (old seeds) or already-CORRECTED
    flops (new seeds trained with the fixed code) -- a ratio-multiply would double-correct the new seeds."""
    lut, gate_err = corrected_fpt_lookup(release_dir)
    byas = defaultdict(list)
    for (a, s, seed), (fb, fc) in lut.items():
        byas[(a, s)].append(fc)
    return {k: st.mean(vs) for k, vs in byas.items()}, gate_err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, default=Path(".cache/nanoprot/release"),
                    help="protein release dir (source of protein cfpt; ctx 512)")
    ap.add_argument("--genome-release", type=Path, default=Path(".cache/nanoprot/genome_traj"),
                    help="genome training dir (source of genome cfpt; ctx 1024 -> gpt2 fpt differs ~9%)")
    ap.add_argument("--dirs", nargs="+", default=[".cache/nanoprot/trajectory_results",
                                                  ".cache/nanoprot/trajectory_results_genome"])
    args = ap.parse_args()

    # Genome gpt2's per-token cost is ~9% higher than protein's (attention over ctx 1024 vs 512), so the
    # genome axis MUST use genome configs. Build a separate cfpt per domain and route each results dir to
    # the matching one (dir name containing "genome" -> genome cfpt, else protein cfpt).
    cfpt_prot, gate_prot = cfpt_from_release(args.release)
    cfpt_gen, gate_gen = cfpt_from_release(args.genome_release) if args.genome_release.exists() else ({}, 0.0)
    print(f"  correction gate: protein={gate_prot:.4%}  genome={gate_gen:.4%} "
          f"({'PASS' if max(gate_prot, gate_gen) < 0.02 else 'FAIL — abort'})")
    if max(gate_prot, gate_gen) >= 0.02:
        return 1
    print("  corrected flops/token (train_flops := cfpt * train_residues; OVERWRITE, not multiply):")
    for lbl, cf in (("protein", cfpt_prot), ("genome", cfpt_gen)):
        for k in sorted(cf):
            print(f"    {lbl:7s} {k[0]:5s} {k[1]}: {cf[k]:.4e}")

    for d in args.dirs:
        src = Path(d)
        if not src.exists():
            continue
        cfpt = cfpt_gen if "genome" in src.name else cfpt_prot
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
                if k in cfpt:
                    r["train_flops"] = f"{cfpt[k] * float(r['train_residues']):.6e}"; n_scaled += 1
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
