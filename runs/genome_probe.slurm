#!/bin/bash
# =============================================================================
# nanoprot — B-GEN genomics trajectory probing (PRD: brain/plans/B-GEN_genomics_PRD.md)
#
# For each genomics trajectory model under $NANOPROT_BASE_DIR/genome_traj/genome-*-s* (one per array
# task), probe EVERY saved checkpoint against the model's FIXED random-init baseline, for the genomic
# concept(s) (default exon/intron). Appends a best-by-val row per (model, task, checkpoint) to
# trajectory_results_genome/<task>.csv + a layer x ckpt JSON sidecar. Mirrors runs/trajectory_probe.slurm;
# only the model dir, task caches, and results dir change. Forward passes only — NO training.
#
# Prereq: runs/genome_train.slurm produced genome_traj/.../model_*.pt, and the genome probe cache
# exists (scripts.prepare_genome_probes --out $NANOPROT_BASE_DIR/probes/genome_exon).
#
# Submit:  mkdir -p logs && sbatch runs/genome_probe.slurm
# Then:    python -m scripts.rank_reversal  --results-dir $NANOPROT_BASE_DIR/trajectory_results_genome
#          python -m scripts.forecast_protocol --results-dir $NANOPROT_BASE_DIR/trajectory_results_genome \
#              --concepts exon     # genome has the single 'exon' concept (forecast_protocol defaults to the protein 5)
# =============================================================================
#SBATCH --job-name=nanoprot-genprobe
#SBATCH --output=logs/%A_%a.%x.out
#SBATCH --error=logs/%A_%a.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=ailab
#SBATCH --array=0-11%8
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

MAX_RESIDUES="${MAX_RESIDUES:-100000}"
TASKS="${TASKS:-genome_exon}"   # genomic concept caches (per-position exon/intron; add splice/promoter later)
RESULTS_DIR="$NANOPROT_BASE_DIR/trajectory_results_genome"

# Optional MODEL_FILTER (grep -E pattern); results CSVs are append-only (no dedup) — see
# trajectory_probe.slurm. Use it to probe only a new wave without duplicating rows.
if [ -n "${MODEL_FILTER:-}" ]; then
    mapfile -t MODELS < <(ls -d "$NANOPROT_BASE_DIR"/genome_traj/genome-*-s* 2>/dev/null | sort | grep -E "$MODEL_FILTER")
else
    mapfile -t MODELS < <(ls -d "$NANOPROT_BASE_DIR"/genome_traj/genome-*-s* 2>/dev/null | sort)
fi
N=${#MODELS[@]}
if [ "$N" -eq 0 ]; then
    echo "ERROR: no trajectory models at $NANOPROT_BASE_DIR/genome_traj/ — run genome_train.slurm first." >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -ge "$N" ]; then
    echo "Array task ${SLURM_ARRAY_TASK_ID} >= ${N} models; nothing to do."; exit 0
fi
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "  genome-probe ${SLURM_ARRAY_TASK_ID}/${N}: $(basename "$MODEL")"

for task in $TASKS; do
    SRC="$NANOPROT_BASE_DIR/probes/${task}"
    [ -d "$SRC" ] || { echo "  -> skip $task (no cache at $SRC)"; continue; }
    echo "  -> $task"
    python -m scripts.run_trajectory_probes \
        --traj-dir "$MODEL" \
        --probe-data "$SRC" \
        --out "$RESULTS_DIR/${task}.csv" \
        --sidecar-dir "$RESULTS_DIR" \
        --max-residues "$MAX_RESIDUES" \
        --device cuda
done
echo "[done] $(basename "$MODEL")"
