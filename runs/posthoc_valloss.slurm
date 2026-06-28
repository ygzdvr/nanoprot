#!/bin/bash
# =============================================================================
# nanoprot — post-hoc per-checkpoint val_loss (P1.5), for the matched-loss axis.
#
# The release_traj runs predate history-logging, so they wrote no val_loss curve.
# This recovers val_loss EXACTLY at each saved checkpoint (load ckpt -> fixed val
# batches -> mean loss), appending one row per (model, checkpoint) to val_loss.csv.
# Forward passes only — NO training.
#
# Default scope = the AR pair at L (gpt2-L, mamba-L x3): esm2 stays OFF the AR
# matched-loss axis (masked-CE != AR bpr), and S/M already have val_loss. Override
# MODEL_FILTER to widen. eval-tokens defaults to the script default (4.2M), matching
# the S/M rows so val_loss is comparable across scales (shared UniRef50 "last" shard).
#
# Submit (after the P1.4 probe array so it never steals GPUs from it):
#   sbatch --dependency=afterany:<P1.4_jobid> runs/posthoc_valloss.slurm
# Then the matched-loss figure (plot_emergence.py) picks up the new L rows.
# =============================================================================
#SBATCH --job-name=nanoprot-valloss
#SBATCH --output=logs/%A_%a.%x.out
#SBATCH --error=logs/%A_%a.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=ailab
#SBATCH --array=0-5%6
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

# AR pair at L only (esm2 off the AR matched-loss axis; S/M already done). Override to widen.
MODEL_FILTER="${MODEL_FILTER:-(-gpt2-L-s[0-9]|-mamba-L-s[0-9])}"
mapfile -t MODELS < <(ls -d "$NANOPROT_BASE_DIR"/release_traj/nanoprot-*-s* 2>/dev/null | sort | grep -E "$MODEL_FILTER")
N=${#MODELS[@]}
if [ "$N" -eq 0 ]; then
    echo "ERROR: no models match MODEL_FILTER='$MODEL_FILTER' under release_traj/." >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -ge "$N" ]; then
    echo "Array task ${SLURM_ARRAY_TASK_ID} >= ${N} models; nothing to do."; exit 0
fi
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "  val_loss ${SLURM_ARRAY_TASK_ID}/${N}: $(basename "$MODEL")"

python -m scripts.posthoc_val_loss_eval \
    --traj-dir "$MODEL" \
    --out "$NANOPROT_BASE_DIR/trajectory_results/val_loss.csv" \
    --device cuda
echo "[done] $(basename "$MODEL")"
