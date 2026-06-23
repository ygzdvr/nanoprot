#!/bin/bash
# =============================================================================
# nanoprot — trajectory-emergence re-train (Pillar 2)
#
# Re-trains a SUBSET of the grid (default gpt2,mamba x {S,M} x 3 seeds = 12 cells)
# while saving ~20 log-spaced (denser-early) intermediate checkpoints per run, so we
# can probe WHEN biological concepts become decodable across training time. Writes to
# release_traj/ (NOT release/) so the public release checkpoints are never clobbered.
#
# These are small (S/M, ~35-150M) models, so this is hours-to-a-day of compute, not the
# multi-day release run. After it completes, probe every checkpoint with
#   sbatch runs/trajectory_probe.slurm
# then analyze with scripts/analyze_emergence.py.
#
# Prerequisites:
#   1) configs generated:
#        python -m scripts.gen_release_configs --intermediate --n-intermediate 20 \
#            --archs gpt2 mamba --scales S M --seeds 0 1 2 \
#            --ckpt-root release_traj --skip-calib --out configs/trajectory --force
#   2) UniRef50 parquet shards at $NANOPROT_BASE_DIR/uniref50_parquet
#
# Submit:  mkdir -p logs && sbatch runs/train_trajectory.slurm
# =============================================================================
#SBATCH --job-name=nanoprot-traj
#SBATCH --output=logs/%A_%a.%x.out
#SBATCH --error=logs/%A_%a.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --partition=ailab
#SBATCH --array=0-11%6
# S/M cells are small; gpu:2 + 6h is generous. --time>=02:00:00 keeps the array on
# gpu-short (not the 25-task gpu-test pool). No resume-from-checkpoint: an incomplete
# cell restarts at step 0. Compute nodes lack internet -> submit with
#   sbatch --export=ALL,NANOPROT_DISABLE_FA3=1 runs/train_trajectory.slurm
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hd0216@princeton.edu

set -euo pipefail

NANOPROT_REPO="${NANOPROT_REPO:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/nanoprot}"
NANOPROT_VENV="${NANOPROT_VENV:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/.venv}"
NPROC="${NPROC:-2}"   # GPUs per task; must match --gres=gpu:N above
CONFIG_DIR="${CONFIG_DIR:-$NANOPROT_REPO/configs/trajectory}"

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
export HF_HUB_OFFLINE=1

[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
source "$NANOPROT_VENV/bin/activate"
cd "$NANOPROT_REPO"

DATA_DIR_EXPECTED="$NANOPROT_BASE_DIR/uniref50_parquet"
if ! ls "$DATA_DIR_EXPECTED"/*.parquet >/dev/null 2>&1; then
    echo "ERROR: no *.parquet shards at $DATA_DIR_EXPECTED — stage UniRef50 first." >&2
    exit 1
fi

mapfile -t CONFIGS < <(ls -1 "$CONFIG_DIR"/nanoprot-*.yaml | sort)
N_CONFIGS=${#CONFIGS[@]}
if [ "$N_CONFIGS" -eq 0 ]; then
    echo "ERROR: no configs in $CONFIG_DIR — run gen_release_configs --intermediate ... --out $CONFIG_DIR" >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -ge "$N_CONFIGS" ]; then
    echo "Array task ${SLURM_ARRAY_TASK_ID} >= ${N_CONFIGS} configs; nothing to do."
    exit 0
fi
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
NAME="$(basename "$CFG" .yaml)"
OUTDIR="$NANOPROT_BASE_DIR/release_traj/$NAME"

echo "============================================================"
echo "  Trajectory task ${SLURM_ARRAY_TASK_ID}/${N_CONFIGS}: $NAME"
echo "  Config : $CFG"
echo "  Outdir : $OUTDIR  (GPUs: $NPROC)"
echo "============================================================"

# Skip if already complete (newest meta has step == num_iterations).
if [ -d "$OUTDIR" ]; then
    last_meta="$(ls -1 "$OUTDIR"/meta_*.json 2>/dev/null | sort | tail -1 || true)"
    if [ -n "${last_meta}" ]; then
        complete="$(python -c "import json; m=json.load(open('${last_meta}')); print(1 if m.get('step')==m.get('num_iterations') else 0)")"
        if [ "${complete}" = "1" ]; then
            echo "[skip] ${NAME} already complete (${last_meta})."
            exit 0
        fi
        echo "[restart] ${NAME} incomplete; retraining from scratch (no resume)."
    fi
fi

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node="$NPROC" \
    -m scripts.train -- --config "$CFG"

echo "[done] ${NAME}"
