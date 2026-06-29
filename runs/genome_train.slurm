#!/bin/bash
# =============================================================================
# nanoprot — B-GEN genomics arch-reversal re-train (PRD: brain/plans/B-GEN_genomics_PRD.md)
#
# Trains the genomics suite — attention(gpt2) vs SSM(mamba) x {S,M} x 3 seeds = 12 cells —
# saving ~20 log-spaced (denser-early) intermediate checkpoints per run, so the rank-reversal
# analysis can probe WHEN genomic concepts (exon/intron) become decodable across compute. Writes
# to genome_traj/ (the config ckpt_root). Mirrors runs/train_trajectory.slurm; only the data,
# config dir, and output dir change (controlled cross-domain replication of the protein protocol).
#
# Prerequisites:
#   1) configs:   python -m scripts.gen_genome_configs --out configs/genome --force
#   2) hg38 shards at $NANOPROT_BASE_DIR/hg38_parquet  (scripts.prepare_genome; download hg38 on a
#      LOGIN node first — compute nodes have no internet).
#   3) SMOKE-TEST the genome train path ONCE before the array (catches any data-loader/embedding bug
#      at ~minutes, not at suite scale). train.py takes only --config, so use the dedicated short
#      smoke config via its launcher:
#        sbatch runs/genome_smoke.slurm        # runs configs/genome/_smoke/genome-smoke.yaml (~40 steps)
#        # then inspect logs/<jid>.*.out for a SANE, DECREASING bits-per-residue (init ~ log2(10) ≈ 3.32).
#      (The CPU forward+backward smoke already passed; this confirms the GPU FA3/bf16 path at scale.)
#
# Submit:  mkdir -p logs && sbatch runs/genome_train.slurm
# =============================================================================
#SBATCH --job-name=nanoprot-gentrain
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
# --time>=02:00:00 keeps the array on gpu-short (not the 25-task gpu-test pool).
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hd0216@princeton.edu

set -euo pipefail

NANOPROT_REPO="${NANOPROT_REPO:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/nanoprot}"
NANOPROT_VENV="${NANOPROT_VENV:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/.venv}"
NPROC="${NPROC:-2}"   # GPUs per task; must match --gres=gpu:N above
CONFIG_DIR="${CONFIG_DIR:-$NANOPROT_REPO/configs/genome}"

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

DATA_DIR_EXPECTED="$NANOPROT_BASE_DIR/hg38_parquet"
if ! ls "$DATA_DIR_EXPECTED"/*.parquet >/dev/null 2>&1; then
    echo "ERROR: no *.parquet shards at $DATA_DIR_EXPECTED — stage hg38 first (scripts.prepare_genome)." >&2
    exit 1
fi

mapfile -t CONFIGS < <(ls -1 "$CONFIG_DIR"/genome-*.yaml | sort)
N_CONFIGS=${#CONFIGS[@]}
if [ "$N_CONFIGS" -eq 0 ]; then
    echo "ERROR: no configs in $CONFIG_DIR — run scripts.gen_genome_configs --out $CONFIG_DIR" >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -ge "$N_CONFIGS" ]; then
    echo "Array task ${SLURM_ARRAY_TASK_ID} >= ${N_CONFIGS} configs; nothing to do."
    exit 0
fi
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
NAME="$(basename "$CFG" .yaml)"
OUTDIR="$NANOPROT_BASE_DIR/genome_traj/$NAME"

echo "============================================================"
echo "  B-GEN task ${SLURM_ARRAY_TASK_ID}/${N_CONFIGS}: $NAME"
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
