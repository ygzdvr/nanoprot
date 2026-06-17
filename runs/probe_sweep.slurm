#!/bin/bash
# =============================================================================
# nanoprot — cross-architecture probe sweep (Pillar 3)
#
# For each of the 36 release checkpoints, probe every cached source under
# $NANOPROT_BASE_DIR/probes/*/ (ss3/ss8/rsa across netsurfp/swissprot/dssp), appending
# one results row per (checkpoint, source) to probe_results/<source>.csv plus a
# per-layer JSON sidecar. Forward passes + linear probes only — NO training.
#
# Prerequisites:
#   1) the release checkpoints under $NANOPROT_BASE_DIR/release/nanoprot-*-s*
#   2) at least one cached probe dataset:  python -m scripts.prepare_probe_data ...
#
# Submit:  mkdir -p logs && sbatch runs/probe_sweep.slurm
# Then:    python -m scripts.plot_probes --results $NANOPROT_BASE_DIR/probe_results/ss3_netsurfp.csv \
#              --sidecar-dir $NANOPROT_BASE_DIR/probe_results --out docs/figures/probes
# =============================================================================
#SBATCH --job-name=nanoprot-probe
#SBATCH --output=logs/%A_%a.%x.out
#SBATCH --error=logs/%A_%a.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=ailab
#SBATCH --array=0-35%8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hd0216@princeton.edu

set -euo pipefail

NANOPROT_REPO="${NANOPROT_REPO:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/nanoprot}"
NANOPROT_VENV="${NANOPROT_VENV:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/.venv}"

module purge
module load gcc/11
module load cudatoolkit/12.8
module load proxy/default

export NANOPROT_BASE_DIR="${NANOPROT_BASE_DIR:-$NANOPROT_REPO/.cache/nanoprot}"
export HF_HOME="$NANOPROT_REPO/.cache/huggingface"
export XDG_CACHE_HOME="$NANOPROT_REPO/.cache"
export TORCH_HOME="$NANOPROT_REPO/.cache/torch"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1          # single-threaded torch (avoids CPU-watchdog kills)
export MKL_NUM_THREADS=1
export NANOPROT_DISABLE_FA3=1     # compute nodes have no internet; skip FA3 fetch -> SDPA

[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
source "$NANOPROT_VENV/bin/activate"
cd "$NANOPROT_REPO"

MAX_RESIDUES="${MAX_RESIDUES:-200000}"   # cap residues per split (bounds memory/time)

mapfile -t CELLS < <(ls -d "$NANOPROT_BASE_DIR"/release/nanoprot-*-s* 2>/dev/null | sort)
N_CELLS=${#CELLS[@]}
if [ "${SLURM_ARRAY_TASK_ID}" -ge "$N_CELLS" ]; then
    echo "Array task ${SLURM_ARRAY_TASK_ID} >= ${N_CELLS} checkpoints; nothing to do."
    exit 0
fi
CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"
echo "  probe task ${SLURM_ARRAY_TASK_ID}/${N_CELLS}: $(basename "$CELL")"

shopt -s nullglob
SOURCES=()
for d in "$NANOPROT_BASE_DIR"/probes/*/; do
    [ -f "${d}meta.json" ] && SOURCES+=("${d%/}")   # any cache dir (ss3/ss8/rsa/...); skips raw/
done
if [ ${#SOURCES[@]} -eq 0 ]; then
    echo "ERROR: no cached probe datasets at $NANOPROT_BASE_DIR/probes/*/ (with meta.json) — run prepare_probe_data first." >&2
    exit 1
fi

for SRC in "${SOURCES[@]}"; do
    [ -d "$SRC" ] || continue
    NAME="$(basename "$SRC")"   # e.g. ss3_netsurfp
    echo "  -> source $NAME"
    python -m scripts.run_probes \
        --checkpoint "$CELL" \
        --probe-data "$SRC" \
        --out "$NANOPROT_BASE_DIR/probe_results/${NAME}.csv" \
        --max-residues "$MAX_RESIDUES" \
        --device cuda
done
echo "[done] $(basename "$CELL")"
