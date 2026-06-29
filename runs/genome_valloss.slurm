#!/bin/bash
# =============================================================================
# nanoprot — B-GEN post-hoc per-checkpoint val_loss (the "loss does NOT reverse" half of the genome
# rank-reversal). Loads each genome checkpoint -> fixed val batches (last hg38 shard) -> mean loss,
# appending one row per (model, checkpoint) to trajectory_results_genome/val_loss.csv. rank_reversal.py
# reads this (optionally) to add the val_bpr loss curve, so we can show capability REVERSES while loss
# does NOT (the Paper B anchor, replicated cross-domain in genomics).
#
# Submit AFTER genome_train.slurm completes (parallel to genome_probe.slurm):
#   sbatch --dependency=afterany:<genome_train_jobid> runs/genome_valloss.slurm
# Then rank_reversal + forecast_protocol on trajectory_results_genome (see genome_probe.slurm header).
# =============================================================================
#SBATCH --job-name=nanoprot-genvalloss
#SBATCH --output=logs/%A_%a.%x.out
#SBATCH --error=logs/%A_%a.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=ailab
#SBATCH --array=0-11%6
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NANOPROT_DISABLE_FA3=1

[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
source "$NANOPROT_VENV/bin/activate"
cd "$NANOPROT_REPO"

# All 12 genome models (gpt2/mamba x S/M x 3 seeds). Override MODEL_FILTER to subset.
MODEL_FILTER="${MODEL_FILTER:-genome-.*-s[0-9]}"
mapfile -t MODELS < <(ls -d "$NANOPROT_BASE_DIR"/genome_traj/genome-*-s* 2>/dev/null | sort | grep -E "$MODEL_FILTER")
N=${#MODELS[@]}
if [ "$N" -eq 0 ]; then
    echo "ERROR: no genome models under $NANOPROT_BASE_DIR/genome_traj (run genome_train.slurm first)." >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -ge "$N" ]; then
    echo "Array task ${SLURM_ARRAY_TASK_ID} >= ${N} models; nothing to do."; exit 0
fi
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "  genome val_loss ${SLURM_ARRAY_TASK_ID}/${N}: $(basename "$MODEL")"

python -m scripts.posthoc_val_loss_eval \
    --traj-dir "$MODEL" \
    --out "$NANOPROT_BASE_DIR/trajectory_results_genome/val_loss.csv" \
    --device cuda
echo "[done] $(basename "$MODEL")"
