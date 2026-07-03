#!/bin/bash
# =============================================================================
# nanoprot — synthetic selectivity-law sweep (Paper B spotlight centerpiece).
# Trains minimal attention & selective-SSM models on a battery of per-position labelling tasks spanning
# the order/periodicity axis (all at a matched Bayes ceiling), logging task-F1 over training so the
# EARLY iso-FLOP margin d_early = F1(attn) - F1(ssm) can be regressed on the axis. Tiny models (~0.3M
# params); each cell ~1-3 min. Array = TASKS x ARCHS x SEEDS; each cell writes its OWN CSV (no append
# race), merged by scripts/synth_regress.py.
#
# Prereq: python -m scripts.synth_selectivity calibrate   (writes docs/synth_task_spec.json)
# Submit:  mkdir -p logs && sbatch --partition=ailab runs/synth_selectivity.slurm
#          (a100:  sbatch --qos=gpu-short --account=ehazan runs/synth_selectivity.slurm)
# Then:    python -m scripts.synth_regress   (merges cells/ -> the dose-response figure)
# =============================================================================
#SBATCH --job-name=nanoprot-synth
#SBATCH --output=logs/%A_%a.%x.out
#SBATCH --error=logs/%A_%a.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --array=0-109%24
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hd0216@princeton.edu

set -euo pipefail

NANOPROT_REPO="${NANOPROT_REPO:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/nanoprot}"
NANOPROT_VENV="${NANOPROT_VENV:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/.venv}"
export NANOPROT_BASE_DIR="${NANOPROT_BASE_DIR:-$NANOPROT_REPO/.cache/nanoprot}"

TASKS=(comp_R2 comp_R6 ratio order_adj motif_R2 motif_R4 period_4 period_8 period_16 aperiodic)
ARCHS=(attn ssm)
SEEDS=(0 1 2 3 4)
STEPS="${STEPS:-2000}"

NT=${#TASKS[@]}; NA=${#ARCHS[@]}; NS=${#SEEDS[@]}
TOTAL=$((NT * NA * NS))
ID=${SLURM_ARRAY_TASK_ID:-0}
if [ "$ID" -ge "$TOTAL" ]; then echo "id $ID >= $TOTAL; nothing to do"; exit 0; fi
# decode id -> (task, arch, seed)
ti=$(( ID / (NA * NS) ))
rem=$(( ID % (NA * NS) ))
ai=$(( rem / NS ))
si=$(( rem % NS ))
TASK=${TASKS[$ti]}; ARCH=${ARCHS[$ai]}; SEED=${SEEDS[$si]}

module purge; module load gcc/11 cudatoolkit/12.8 2>/dev/null || module load gcc/11
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
source "$NANOPROT_VENV/bin/activate"
cd "$NANOPROT_REPO"

OUT="$NANOPROT_BASE_DIR/synth_results/cells/cell_${TASK}_${ARCH}_s${SEED}.csv"
mkdir -p "$(dirname "$OUT")"
echo "[$ID/$TOTAL] task=$TASK arch=$ARCH seed=$SEED steps=$STEPS -> $OUT"
python -m scripts.synth_selectivity run --task "$TASK" --arch "$ARCH" --seed "$SEED" \
    --steps "$STEPS" --out "$OUT"
echo "[done] $TASK/$ARCH/s$SEED"
