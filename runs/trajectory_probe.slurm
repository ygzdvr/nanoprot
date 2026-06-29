#!/bin/bash
# =============================================================================
# nanoprot — trajectory-emergence probing (Pillar 2)
#
# For each trajectory model under $NANOPROT_BASE_DIR/release_traj/nanoprot-*-s* (one per
# array task), probe EVERY saved checkpoint against the model's FIXED random-init baseline,
# for the core local-structure tasks (ss3/ss8/rsa on NetSurfP). Appends a best-by-val row
# per (model, task, checkpoint) to trajectory_results/<task>.csv + a full layer x ckpt JSON
# sidecar. The driver builds the baseline ONCE per model and reuses it across checkpoints.
#
# Prereq: runs/train_trajectory.slurm has produced release_traj/.../model_*.pt, and the
# probe caches exist (scripts.prepare_probe_data). Forward passes only — NO training.
#
# Submit:  mkdir -p logs && sbatch runs/trajectory_probe.slurm
# Then:    python -m scripts.analyze_emergence --results-dir $NANOPROT_BASE_DIR/trajectory_results
# =============================================================================
#SBATCH --job-name=nanoprot-trajprobe
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
# Core local-structure tasks probed at every checkpoint (the first emergence wave).
# A-BREADTH (Nature breadth lever): add sequence-level concepts here — they auto-pool (run_trajectory_probes
# detects level="sequence" and mean-pools residue reps). e.g.  TASKS="pfam_swissprot" sbatch runs/trajectory_probe.slurm
TASKS="${TASKS:-ss3_netsurfp ss8_netsurfp rsa_netsurfp}"

# Optional MODEL_FILTER (a grep -E pattern) restricts WHICH trajectory models this run probes.
# The results CSVs are append-only with NO dedup, so re-probing an already-done model duplicates
# its rows. Use MODEL_FILTER to probe only a new wave, e.g.:
#   MODEL_FILTER='(-L-s[0-9]|-esm2-M-s[0-9])'  -> only the L + esm2 models (skip done S/M).
if [ -n "${MODEL_FILTER:-}" ]; then
    mapfile -t MODELS < <(ls -d "$NANOPROT_BASE_DIR"/release_traj/nanoprot-*-s* 2>/dev/null | sort | grep -E "$MODEL_FILTER")
else
    mapfile -t MODELS < <(ls -d "$NANOPROT_BASE_DIR"/release_traj/nanoprot-*-s* 2>/dev/null | sort)
fi
N=${#MODELS[@]}
if [ "$N" -eq 0 ]; then
    echo "ERROR: no trajectory models at $NANOPROT_BASE_DIR/release_traj/ — run train_trajectory.slurm first." >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -ge "$N" ]; then
    echo "Array task ${SLURM_ARRAY_TASK_ID} >= ${N} models; nothing to do."; exit 0
fi
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "  trajectory-probe ${SLURM_ARRAY_TASK_ID}/${N}: $(basename "$MODEL")"

for task in $TASKS; do
    SRC="$NANOPROT_BASE_DIR/probes/${task}"
    [ -d "$SRC" ] || { echo "  -> skip $task (no cache at $SRC)"; continue; }
    echo "  -> $task"
    python -m scripts.run_trajectory_probes \
        --traj-dir "$MODEL" \
        --probe-data "$SRC" \
        --out "$NANOPROT_BASE_DIR/trajectory_results/${task}.csv" \
        --sidecar-dir "$NANOPROT_BASE_DIR/trajectory_results" \
        --max-residues "$MAX_RESIDUES" \
        --device cuda
done
echo "[done] $(basename "$MODEL")"
