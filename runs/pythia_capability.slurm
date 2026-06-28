#!/bin/bash
# =============================================================================
# Option A — cross-domain (NLP) capability trajectories from public Pythia checkpoints.
# Computes LAMBADA-openai accuracy + held-out LM loss per (size, step) -> docs/pythia_capability.csv,
# which the forecasting analysis consumes to show the method generalizes beyond proteins.
# Needs internet (proxy) for HF downloads; fp32 (bf16 corrupts confident-checkpoint CE/argmax).
# Submit:  mkdir -p logs && sbatch runs/pythia_capability.slurm
# =============================================================================
#SBATCH --job-name=pythia-cap
#SBATCH --output=logs/%A.%x.out
#SBATCH --error=logs/%A.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --partition=ailab
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hd0216@princeton.edu

set -euo pipefail
NANOPROT_REPO="${NANOPROT_REPO:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/nanoprot}"
NANOPROT_VENV="${NANOPROT_VENV:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/.venv}"

module purge
module load gcc/11
module load cudatoolkit/12.8
module load proxy/default   # internet for HF downloads

export HF_HOME="$NANOPROT_REPO/.cache/huggingface"
export HF_HUB_OFFLINE=0     # allow downloading public Pythia checkpoints
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
source "$NANOPROT_VENV/bin/activate"
cd "$NANOPROT_REPO"

python -m scripts.pythia_capability \
    --sizes 70m 160m 410m \
    --steps 1 2 4 8 16 32 64 128 256 512 1000 2000 4000 8000 16000 32000 64000 143000 \
    --n-examples 500 \
    --out docs/pythia_capability.csv \
    --device cuda
echo "[done] pythia capability sweep"
