#!/bin/bash
# =============================================================================
# nanoprot v0.5 — throughput calibration probe
#
# Runs a short (~120-step) job per architecture at the S scale. Two purposes:
#   1) THROUGHPUT: the loop prints `tok/s` every 10 steps; read the steady-state
#      value (after warmup, ~step 60+) to size the full array's --time.
#   2) INTEGRATION GATE: the calib configs run quick eval passes, so confirm each
#      arch emits a SANE, DECREASING val_bpr (~log2(33)=5.04 bits at init) before
#      launching the 36-cell array. This is the first real execution of the AR
#      archs' (objective=ar, tokenizer=esm2) path — verify it here, once, not 36x.
#
# Wall-clock cannot be reliably estimated a priori — especially at small scale
# where the run is latency- not FLOP-bound.
#
# Prerequisites:
#   1) python -m scripts.gen_release_configs   (writes configs/release/calib/)
#   2) UniRef50 parquet shards at $NANOPROT_BASE_DIR/uniref50_parquet
#
# Submit:
#   mkdir -p logs
#   sbatch runs/calibrate_throughput.slurm
#
# Then read throughput:
#   grep 'tok/s' logs/<jobid>_*.nanoprot-calib.out | tail -5
# =============================================================================
#SBATCH --job-name=nanoprot-calib
#SBATCH --output=logs/%A_%a.%x.out
#SBATCH --error=logs/%A_%a.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --partition=ailab
#SBATCH --array=0-2
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

# Pre-flight: data must be staged (the built-in downloader fetches English
# ClimbMix text, NOT proteins — do not use it).
DATA_DIR_EXPECTED="$NANOPROT_BASE_DIR/uniref50_parquet"
if ! ls "$DATA_DIR_EXPECTED"/*.parquet >/dev/null 2>&1; then
    echo "ERROR: no *.parquet shards at $DATA_DIR_EXPECTED" >&2
    echo "  Build the corpus first: download the FASTA on a login node, then" >&2
    echo "  sbatch runs/prepare_uniref50.slurm  (see that script's header)." >&2
    exit 1
fi

ARCHS=(gpt2 esm2 mamba)
ARCH="${ARCHS[$SLURM_ARRAY_TASK_ID]}"
CFG="$NANOPROT_REPO/configs/release/calib/calib-${ARCH}.yaml"

echo "============================================================"
echo "  Throughput probe: ${ARCH}  (S scale, ~120 steps)"
echo "  Config: $CFG"
echo "  GPUs  : $NPROC"
echo "  -> read steady-state 'tok/s' (residues/sec) from below"
echo "============================================================"

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node="$NPROC" \
    -m scripts.train -- --config "$CFG"

echo "[calib done] ${ARCH}: grep this log for 'tok/s' and use the steady-state value."
