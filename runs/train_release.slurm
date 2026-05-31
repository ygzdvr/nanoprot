#!/bin/bash
# =============================================================================
# nanoprot v0.5 — model-release training array
#
# Trains the full release grid: 3 archs x 4 scales x 3 seeds = 36 cells, one
# per array task. Resumable / skip-completed: a cell whose final checkpoint
# exists (meta.step == meta.num_iterations) is skipped, so re-submitting only
# trains what is missing. WANDB runs offline; sync afterwards with
#   wandb sync --sync-all wandb/
#
# Prerequisites:
#   1) configs generated:  python -m scripts.gen_release_configs
#   2) UniRef50 parquet shards present at $NANOPROT_BASE_DIR/uniref50_parquet
#   3) (recommended) run runs/calibrate_throughput.slurm first and adjust
#      --time / device_batch_size to the measured throughput.
#
# Submit:
#   mkdir -p logs
#   sbatch runs/train_release.slurm
# =============================================================================
#SBATCH --job-name=nanoprot-release
#SBATCH --output=logs/%A_%a.%x.out
#SBATCH --error=logs/%A_%a.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --partition=ailab
#SBATCH --array=0-35%8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hd0216@princeton.edu

set -euo pipefail

# ---------------------------------------------------------------------------
# Cluster-specific paths (env-overridable so a public user can adapt).
# ---------------------------------------------------------------------------
NANOPROT_REPO="${NANOPROT_REPO:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/nanoprot}"
NANOPROT_VENV="${NANOPROT_VENV:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/.venv}"
NPROC="${NPROC:-4}"   # GPUs per task; must match --gres=gpu:N above

# ---------------------------------------------------------------------------
# Modules (della) — adapt on other clusters.
# ---------------------------------------------------------------------------
module purge
module load gcc/11
module load cudatoolkit/12.8
module load proxy/default

# ---------------------------------------------------------------------------
# Caches: keep everything under the project tree, not $HOME.
# ---------------------------------------------------------------------------
export NANOPROT_BASE_DIR="${NANOPROT_BASE_DIR:-$NANOPROT_REPO/.cache/nanoprot}"
export HF_HOME="$NANOPROT_REPO/.cache/huggingface"
export HF_DATASETS_CACHE="$NANOPROT_REPO/.cache/huggingface/datasets"
export XDG_CACHE_HOME="$NANOPROT_REPO/.cache"
export TORCH_HOME="$NANOPROT_REPO/.cache/torch"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

# ---------------------------------------------------------------------------
# Environment.
# ---------------------------------------------------------------------------
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
source "$NANOPROT_VENV/bin/activate"
cd "$NANOPROT_REPO"

# ---------------------------------------------------------------------------
# Pick this task's config (stable sorted order matches array indices 0..35).
# ---------------------------------------------------------------------------
mapfile -t CONFIGS < <(ls -1 "$NANOPROT_REPO"/configs/release/nanoprot-*.yaml | sort)
N_CONFIGS=${#CONFIGS[@]}
if [ "$N_CONFIGS" -ne 36 ]; then
    echo "WARNING: expected 36 release configs, found $N_CONFIGS." >&2
    echo "Run: python -m scripts.gen_release_configs" >&2
fi
if [ "${SLURM_ARRAY_TASK_ID}" -ge "$N_CONFIGS" ]; then
    echo "Array task ${SLURM_ARRAY_TASK_ID} >= ${N_CONFIGS} configs; nothing to do."
    exit 0
fi
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
NAME="$(basename "$CFG" .yaml)"
OUTDIR="$NANOPROT_BASE_DIR/release/$NAME"

echo "============================================================"
echo "  Task ${SLURM_ARRAY_TASK_ID}/${N_CONFIGS}: $NAME"
echo "  Config : $CFG"
echo "  Outdir : $OUTDIR"
echo "  GPUs   : $NPROC"
echo "============================================================"

# ---------------------------------------------------------------------------
# Skip if already complete (final checkpoint present).
# A run is complete iff its newest meta has step == num_iterations.
# ---------------------------------------------------------------------------
if [ -d "$OUTDIR" ]; then
    last_meta="$(ls -1 "$OUTDIR"/meta_*.json 2>/dev/null | sort | tail -1 || true)"
    if [ -n "${last_meta}" ]; then
        complete="$(python -c "import json; m=json.load(open('${last_meta}')); print(1 if m.get('step')==m.get('num_iterations') else 0)")"
        if [ "${complete}" = "1" ]; then
            echo "[skip] ${NAME} already complete (${last_meta})."
            exit 0
        fi
        # NB: the training loop has no resume-from-checkpoint; an incomplete cell
        # restarts at step 0 (its partial checkpoints are overwritten).
        echo "[restart] ${NAME} has an incomplete checkpoint; retraining from scratch."
    fi
fi

# ---------------------------------------------------------------------------
# Train.
# ---------------------------------------------------------------------------
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node="$NPROC" \
    -m scripts.train -- --config "$CFG"

echo "[done] ${NAME}"
