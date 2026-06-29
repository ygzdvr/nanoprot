#!/bin/bash
# =============================================================================
# nanoprot — prepare the GRCh38/hg38 corpus for B-GEN (FASTA -> char-windowed parquet shards)
#
# CPU-only, no GPU (partition=cpu, NOT GPU-gated). Two-step workflow, mirroring
# runs/prepare_uniref50.slurm:
#
#   STEP 1 (once, on a LOGIN node — compute nodes have no internet):
#     wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz \
#          -O "$NANOPROT_BASE_DIR/hg38.fa.gz"                                     # ~984 MB
#     wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.annotation.gtf.gz \
#          -O "$NANOPROT_BASE_DIR/gencode.v46.annotation.gtf.gz"                  # ~51 MB (for probes)
#
#   STEP 2 (this job — windows the genome into shards):
#     mkdir -p logs && sbatch runs/prepare_genome.slurm
#
# Produces $NANOPROT_BASE_DIR/hg38_parquet/shard_*.parquet (last = validation). Then build the probe
# cache + configs (both light; login or a short CPU job):
#     python -m scripts.prepare_genome_probes --fasta "$NANOPROT_BASE_DIR/hg38.fa.gz" \
#         --gtf "$NANOPROT_BASE_DIR/gencode.v46.annotation.gtf.gz" \
#         --out "$NANOPROT_BASE_DIR/probes/genome_exon" --concept exon --window-len 1024
#     python -m scripts.gen_genome_configs --out configs/genome --force
# then (GPU): smoke-test the tokenizer, sbatch runs/genome_train.slurm, sbatch runs/genome_probe.slurm.
#
# For a quick subset to smoke-test the pipeline first: MAX_WINDOWS=200000 NUM_SHARDS=8 sbatch ...
# =============================================================================
#SBATCH --job-name=nanoprot-genprepdata
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hd0216@princeton.edu

set -euo pipefail

NANOPROT_REPO="${NANOPROT_REPO:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/nanoprot}"
NANOPROT_VENV="${NANOPROT_VENV:-/scratch/gpfs/EHAZAN/hd0216/Senior-Thesis/scaling/nanochat/NEW-REPO/nanochat/.venv}"
export NANOPROT_BASE_DIR="${NANOPROT_BASE_DIR:-$NANOPROT_REPO/.cache/nanoprot}"

# Tunables (override at submit: e.g. `MAX_WINDOWS=200000 NUM_SHARDS=8 sbatch ...`).
FASTA="${FASTA:-$NANOPROT_BASE_DIR/hg38.fa.gz}"
OUT="${OUT:-$NANOPROT_BASE_DIR/hg38_parquet}"
NUM_SHARDS="${NUM_SHARDS:-200}"
WINDOW_LEN="${WINDOW_LEN:-1024}"
MAX_N_FRAC="${MAX_N_FRAC:-0.10}"
MAX_WINDOWS="${MAX_WINDOWS:-0}"   # 0 = all (~3M windows for hg38 @ 1024bp non-overlapping)

module purge
module load gcc/11

[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
source "$NANOPROT_VENV/bin/activate"
cd "$NANOPROT_REPO"

if [ ! -f "$FASTA" ]; then
    echo "ERROR: genome FASTA not found: $FASTA — download it on a login node (see header)." >&2
    exit 1
fi

echo "Windowing $FASTA -> $OUT  (window_len=$WINDOW_LEN, num_shards=$NUM_SHARDS, max_windows=$MAX_WINDOWS)"
python -m scripts.prepare_genome \
    --fasta "$FASTA" \
    --out "$OUT" \
    --num-shards "$NUM_SHARDS" \
    --window-len "$WINDOW_LEN" \
    --max-n-frac "$MAX_N_FRAC" \
    --max-windows "$MAX_WINDOWS"

echo "[done] genome shards at $OUT — next: prepare_genome_probes + gen_genome_configs, then the GPU run."
