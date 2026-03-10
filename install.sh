#!/usr/bin/env bash
# =============================================================================
# install.sh — Less Is More environment setup
#
# Installs the full Python environment in the correct order.
# Verified on: Python 3.12, CUDA 12.x toolkit, A6000 (sm_86) / A100 (sm_80)
#
# Usage (from repo root):
#   bash install.sh
#
# What this script does:
#   1. Creates conda env "scpipe" with Python 3.12
#   2. Installs PyTorch 2.4.1+cu121 via pip  (works on driver >= 525)
#   3. Installs all other dependencies from requirements.txt
#   4. Builds flash-attn 2.7.2.post1 from source
#
# Notes:
#   - Use pip-installed PyTorch, NOT conda — conda PyTorch causes MKL
#     symbol conflicts (iJIT_NotifyEvent) on many Linux systems.
#   - cu121 wheels bundle their own CUDA 12.1 runtime and work on any
#     driver >= 525, including CUDA 12.4/12.6 toolkit machines.
#   - flash-attn 2.8.1 requires nvcc >= 12.5 (-compress-mode=size flag);
#     2.7.2.post1 works with nvcc 12.x (tested: 12.4).
#   - TORCH_CUDA_ARCH_LIST: set to your GPU.
#     A100 = 8.0, A6000/A40 = 8.6, H100 = 9.0, RTX 3090 = 8.6
# =============================================================================

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
step() { echo -e "\n${GREEN}==>${NC} $1"; }
info() { echo -e "${YELLOW}   $1${NC}"; }

# ── User-configurable ────────────────────────────────────────────────────────
ENV_NAME="scpipe"
# Adjust to your GPU architecture (space-separated if multiple):
#   A100=8.0  A6000/A40/RTX3090=8.6  H100=9.0
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 8.6}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
MAX_JOBS="${MAX_JOBS:-8}"
# ─────────────────────────────────────────────────────────────────────────────

# ── Step 1: Create conda environment ─────────────────────────────────────────
step "Creating conda environment '$ENV_NAME' (Python 3.12)"
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -q "^${ENV_NAME} "; then
    info "Environment '$ENV_NAME' already exists — skipping creation."
    info "To recreate: conda env remove -n $ENV_NAME -y"
else
    conda create -n "$ENV_NAME" python=3.12 -y
fi
conda activate "$ENV_NAME"

# ── Step 2: PyTorch 2.4.1+cu121 (pip, NOT conda) ─────────────────────────────
step "Installing PyTorch 2.4.1+cu121 via pip"
info "Using pip wheels — avoids MKL symbol conflict from conda PyTorch"
pip install \
    torch==2.4.1+cu121 \
    torchvision==0.19.1+cu121 \
    torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Step 3: All other dependencies ───────────────────────────────────────────
step "Installing dependencies from requirements.txt"
pip install -r requirements.txt

# ── Step 4: flash-attn from source ───────────────────────────────────────────
step "Building flash-attn 2.7.2.post1 from source"
info "GPU arch: $TORCH_CUDA_ARCH_LIST  |  CUDA_HOME: $CUDA_HOME  |  jobs: $MAX_JOBS"
info "This takes ~5–15 minutes depending on CPU core count."

if [ ! -d "$CUDA_HOME" ]; then
    echo "ERROR: CUDA_HOME='$CUDA_HOME' does not exist."
    echo "       Set CUDA_HOME to your CUDA toolkit path and re-run."
    exit 1
fi

export CUDA_HOME
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS
pip install flash-attn==2.7.2.post1 --no-build-isolation

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Installation complete.${NC}"
echo ""
echo "Activate the environment:  conda activate $ENV_NAME"
echo "Run smoke test (no GPU):   bash smoke_test.sh"
echo "Run smoke test (GPU):      bash smoke_test.sh --tier1"
