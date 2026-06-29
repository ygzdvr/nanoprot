#!/bin/bash
# =============================================================================
# nanoprot — B-GEN GPU smoke test (1 model, ~40 steps) BEFORE the 12-model suite.
# Validates the genome train-path on GPU (FA3/bf16) + a DECREASING bits-per-residue
# (init ~ log2(10) = 3.32 bits). The CPU forward+backward smoke already passed; this confirms the
# GPU/precision path at scale before committing the suite. Runs configs/genome/_smoke/genome-smoke.yaml.
#
# Submit AFTER the data-prep (chained):
#   jid=$(sbatch --parsable runs/prepare_genome.slurm)
#   sbatch --dependency=afterok:$jid runs/genome_smoke.slurm
# Then inspect logs/<jid>.*.out for a sane, decreasing bpr; only then sbatch runs/genome_train.slurm.
# =============================================================================
#SBATCH --job-name=nanoprot-gensmoke
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=ailab
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
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1

[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
source "$NANOPROT_VENV/bin/activate"
cd "$NANOPROT_REPO"

if ! ls "$NANOPROT_BASE_DIR/hg38_parquet"/*.parquet >/dev/null 2>&1; then
    echo "ERROR: no hg38 shards at $NANOPROT_BASE_DIR/hg38_parquet — run prepare_genome.slurm first." >&2
    exit 1
fi

CFG="$NANOPROT_REPO/configs/genome/_smoke/genome-smoke.yaml"
echo "============================================================"
echo "  B-GEN GPU smoke: $CFG  (~40 steps, expect bpr decreasing from ~3.32 bits)"
echo "============================================================"
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=1 -m scripts.train -- --config "$CFG"
echo "[done] genome smoke — inspect the bpr trajectory above before launching genome_train.slurm."
