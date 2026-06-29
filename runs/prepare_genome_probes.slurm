#!/bin/bash
# =============================================================================
# nanoprot — build the B-GEN GENCODE exon/intron probe cache (per-position labels).
# CPU-only (partition=cpu), mirrors prepare_genome.slurm. Reads the staged hg38 FASTA + GENCODE GTF,
# windows gene bodies, writes meta.json/data.jsonl/provenance.json under $NANOPROT_BASE_DIR/probes/.
# Chromosome-disjoint train/val/test splits (no positional leakage).
#
# WINDOW_LEN=1023 (NOT 1024): the model's training INPUT is [<cls> + 1023 bases] = 1024 tokens
# (the loader builds rows of max_seq_len+1=1025 and the input is row[:-1]); 1023-bp probe windows +
# <cls> = 1024 tokens match the trained positions 0..1023 exactly, with no truncation.
#
# Submit:  mkdir -p logs && sbatch runs/prepare_genome_probes.slurm
# =============================================================================
#SBATCH --job-name=nanoprot-genprobedata
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hd0216@princeton.edu

set -euo pipefail

NANOPROT_REPO="${NANOPROT_REPO:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/nanoprot}"
NANOPROT_VENV="${NANOPROT_VENV:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/.venv}"
export NANOPROT_BASE_DIR="${NANOPROT_BASE_DIR:-$NANOPROT_REPO/.cache/nanoprot}"

FASTA="${FASTA:-$NANOPROT_BASE_DIR/hg38.fa.gz}"
GTF="${GTF:-$NANOPROT_BASE_DIR/gencode.v46.annotation.gtf.gz}"
CONCEPT="${CONCEPT:-exon}"
OUT="${OUT:-$NANOPROT_BASE_DIR/probes/genome_${CONCEPT}}"
WINDOW_LEN="${WINDOW_LEN:-1023}"
MAX_N_FRAC="${MAX_N_FRAC:-0.10}"

module purge
module load gcc/11

[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
source "$NANOPROT_VENV/bin/activate"
cd "$NANOPROT_REPO"

for f in "$FASTA" "$GTF"; do
    [ -f "$f" ] || { echo "ERROR: missing input: $f (download on a login node)." >&2; exit 1; }
done

echo "Building $CONCEPT probe cache -> $OUT  (window_len=$WINDOW_LEN)"
python -m scripts.prepare_genome_probes \
    --fasta "$FASTA" \
    --gtf "$GTF" \
    --out "$OUT" \
    --concept "$CONCEPT" \
    --window-len "$WINDOW_LEN" \
    --max-n-frac "$MAX_N_FRAC"

echo "[done] $CONCEPT probe cache at $OUT — consumed by runs/genome_probe.slurm after training."
