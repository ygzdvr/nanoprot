#!/bin/bash
# =============================================================================
# nanoprot — prepare the UniRef50 corpus (FASTA -> shuffled parquet shards)
#
# CPU-only, no GPU, no internet needed for THIS step. Two-step workflow:
#
#   STEP 1 (once, on a LOGIN node — compute nodes have no internet):
#     mkdir -p "$NANOPROT_BASE_DIR"
#     wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz \
#          -O "$NANOPROT_BASE_DIR/uniref50.fasta.gz"      # ~14 GB, takes a while
#     # Also grab the release notes so the exact UniRef50 release is recorded in
#     # provenance.json (and cited in the model cards) automatically:
#     wget https://ftp.uniprot.org/pub/databases/uniprot/relnotes.txt \
#          -O "$NANOPROT_BASE_DIR/relnotes.txt"
#
#   STEP 2 (this job — shards the downloaded FASTA):
#     mkdir -p logs
#     sbatch runs/prepare_uniref50.slurm
#
# Produces $NANOPROT_BASE_DIR/uniref50_parquet/shard_*.parquet (last shard =
# validation). After it finishes, runs/calibrate_throughput.slurm will pass its
# data pre-flight. For a quick subset to smoke-test the pipeline first, set
# MAX_SEQUENCES (e.g. 200000) and NUM_SHARDS=8 below.
# =============================================================================
#SBATCH --job-name=nanoprot-prepdata
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

# Tunables (override at submit time: e.g. `MAX_SEQUENCES=200000 NUM_SHARDS=8 sbatch ...`).
FASTA="${FASTA:-$NANOPROT_BASE_DIR/uniref50.fasta.gz}"
OUT="${OUT:-$NANOPROT_BASE_DIR/uniref50_parquet}"
NUM_SHARDS="${NUM_SHARDS:-200}"
MIN_LEN="${MIN_LEN:-20}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"   # 0 = use all sequences

module purge
module load gcc/11

[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
source "$NANOPROT_VENV/bin/activate"
cd "$NANOPROT_REPO"

if [ ! -f "$FASTA" ]; then
    echo "ERROR: FASTA not found: $FASTA" >&2
    echo "  Download it first on a login node (see header of this script):" >&2
    echo "  wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz \\" >&2
    echo "       -O \"$FASTA\"" >&2
    exit 1
fi

echo "Sharding $FASTA -> $OUT  (num_shards=$NUM_SHARDS, max_sequences=$MAX_SEQUENCES)"
python -m scripts.prepare_uniref50 \
    --fasta "$FASTA" \
    --out "$OUT" \
    --num-shards "$NUM_SHARDS" \
    --min-len "$MIN_LEN" \
    --max-sequences "$MAX_SEQUENCES"

echo "[done] data ready at $OUT — now: sbatch runs/calibrate_throughput.slurm"
