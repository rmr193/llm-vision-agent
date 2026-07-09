#!/usr/bin/env bash
# Installs a ROCm-enabled PyTorch + vLLM stack, then the rest of requirements.txt.
#
# Tested against ROCm 6.1 on Ubuntu 22.04 with an AMD Instinct MI-series or
# Radeon RX 7000-series GPU. Adjust ROCM_VERSION below if your host uses a
# different ROCm release.
set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:-6.1}"
PY_BIN="${PY_BIN:-python3}"

echo "== ScreenSage ROCm setup (ROCm ${ROCM_VERSION}) =="

echo "[1/4] Checking for ROCm driver (rocminfo)..."
if ! command -v rocminfo &> /dev/null; then
  echo "!! rocminfo not found. Install the ROCm driver stack first:"
  echo "   https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
  exit 1
fi
rocminfo | grep -A3 "Agent 2" || true

echo "[2/4] Installing ROCm-enabled PyTorch..."
"${PY_BIN}" -m pip install --upgrade pip
"${PY_BIN}" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/rocm${ROCM_VERSION}"

echo "[3/4] Installing vLLM (ROCm build)..."
# vLLM's ROCm wheels track specific torch/ROCm combos; see:
# https://docs.vllm.ai/en/latest/getting_started/amd-installation.html
"${PY_BIN}" -m pip install "vllm[rocm]" || {
  echo "Prebuilt vLLM ROCm wheel not available for this combo; building from source..."
  git clone https://github.com/vllm-project/vllm.git /tmp/vllm-src
  cd /tmp/vllm-src
  PYTORCH_ROCM_ARCH=$(rocminfo | grep -o 'gfx[0-9a-f]*' | head -n1) \
    "${PY_BIN}" -m pip install -e .
  cd -
}

echo "[4/4] Installing remaining Python dependencies..."
"${PY_BIN}" -m pip install -r requirements.txt

echo "== Done. Verify with: python -c 'import torch; print(torch.cuda.is_available(), torch.version.hip)' =="
