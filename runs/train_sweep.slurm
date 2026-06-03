#!/bin/bash
# =============================================================================
# nanoprot — data-budget sweep training array (for L(N,D) scaling laws)
#
# Trains the configs from `python -m scripts.gen_sweep_configs` (default: 3 archs
# x S scale x 6 token budgets = 18 runs), holding model size fixed and varying D,
# so the data-exponent beta is identifiable. Same contract as train_release.slurm:
# resumable / skip-completed, offline wandb.
#
# Prerequisites: python -m scripts.gen_sweep_configs ; UniRef50 parquet staged.
# Submit:  mkdir -p logs && sbatch runs/train_sweep.slurm
#          (internet-less compute nodes: add --export=ALL,NANOPROT_DISABLE_FA3=1)
# =============================================================================
#SBATCH --job-name=nanoprot-sweep
#SBATCH --output=logs/%A_%a.%x.out
#SBATCH --error=logs/%A_%a.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --partition=ailab
#SBATCH --array=0-17%8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hd0216@princeton.edu

set -euo pipefail

NANOPROT_REPO="${NANOPROT_REPO:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/nanoprot}"
NANOPROT_VENV="${NANOPROT_VENV:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/.venv}"
NPROC="${NPROC:-4}"

module purge
module load gcc/11
module load cudatoolkit/12.8
module load proxy/default

export NANOPROT_BASE_DIR="${NANOPROT_BASE_DIR:-$NANOPROT_REPO/.cache/nanoprot}"
export HF_HOME="$NANOPROT_REPO/.cache/huggingface"
export XDG_CACHE_HOME="$NANOPROT_REPO/.cache"
export TORCH_HOME="$NANOPROT_REPO/.cache/torch"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
source "$NANOPROT_VENV/bin/activate"
cd "$NANOPROT_REPO"

DATA_DIR_EXPECTED="$NANOPROT_BASE_DIR/uniref50_parquet"
if ! ls "$DATA_DIR_EXPECTED"/*.parquet >/dev/null 2>&1; then
    echo "ERROR: no *.parquet shards at $DATA_DIR_EXPECTED — build the corpus first." >&2
    exit 1
fi

mapfile -t CONFIGS < <(ls -1 "$NANOPROT_REPO"/configs/sweep/nanoprot-sweep-*.yaml | sort)
N_CONFIGS=${#CONFIGS[@]}
if [ "${SLURM_ARRAY_TASK_ID}" -ge "$N_CONFIGS" ]; then
    echo "Array task ${SLURM_ARRAY_TASK_ID} >= ${N_CONFIGS} configs; nothing to do."
    exit 0
fi
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
NAME="$(basename "$CFG" .yaml)"
OUTDIR="$NANOPROT_BASE_DIR/sweep/$NAME"

echo "  sweep task ${SLURM_ARRAY_TASK_ID}/${N_CONFIGS}: $NAME  ($NPROC GPUs)"

if [ -d "$OUTDIR" ]; then
    last_meta="$(ls -1 "$OUTDIR"/meta_*.json 2>/dev/null | sort | tail -1 || true)"
    if [ -n "${last_meta}" ]; then
        complete="$(python -c "import json; m=json.load(open('${last_meta}')); print(1 if m.get('step')==m.get('num_iterations') else 0)")"
        if [ "${complete}" = "1" ]; then echo "[skip] ${NAME} complete."; exit 0; fi
    fi
fi

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node="$NPROC" \
    -m scripts.train -- --config "$CFG"
echo "[done] ${NAME}"
