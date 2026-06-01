#!/bin/bash
# =============================================================================
# Build + install the fused Mamba selective-scan CUDA kernel (mamba_ssm).
#
# GPU Mamba TRAINING requires this — the pure-PyTorch reference scan in
# nanoprot/models/mamba.py is ~100x slower (validated: 4k vs 430k tok/s on an
# H200) and only there for portability / CPU tests. With the kernel installed,
# nanoprot.models.mamba.HAS_SELECTIVE_SCAN_CUDA becomes True and the model uses
# the fused path on CUDA automatically (reference fallback otherwise).
#
# No prebuilt wheel exists for recent torch, so this compiles from source
# (~10 min). Run on a node with nvcc + a GPU visible (a GPU login node works).
# Validated against torch 2.9.1+cu128 / CUDA 12.8 on della (A100 + H200).
#
#   bash runs/install_mamba_ssm.sh
# =============================================================================
set -euo pipefail

NANOPROT_VENV="${NANOPROT_VENV:-$(cd "$(dirname "$0")/../.." && pwd)/.venv}"

module load gcc/11 cudatoolkit/12.8
source "$NANOPROT_VENV/bin/activate"

# Compile for the GPUs you'll train on. "8.0;9.0" = A100 + H100/H200. The build
# also picks up the build host's GPU arch. Narrow this to speed up the build.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
export MAX_JOBS="${MAX_JOBS:-4}"
export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

uv pip install ninja packaging setuptools wheel
# nanoprot only uses selective_scan_fn (it keeps F.conv1d for the conv), so
# causal-conv1d is optional; installed here for completeness / future use.
uv pip install "causal-conv1d>=1.4" --no-build-isolation
uv pip install "mamba-ssm" --no-build-isolation

python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; \
import nanoprot.models.mamba as m; print('HAS_SELECTIVE_SCAN_CUDA =', m.HAS_SELECTIVE_SCAN_CUDA)"
echo "[done] fused Mamba scan installed — GPU Mamba training is now fast."
