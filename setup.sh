#!/usr/bin/env bash
# ==============================================================================
# Portable Runtime Installer for Dell G15 MoE Runner
# Downloads and extracts precompiled CUDA 13.4 llama.cpp binaries into ./runtime
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${SCRIPT_DIR}/runtime"
BUILD_TAG="b11729"

echo "=== [1/3] Preparing Runtime Directory ==="
mkdir -p "$RUNTIME_DIR"

if [[ -f "${RUNTIME_DIR}/llama-server" && -f "${RUNTIME_DIR}/libggml-cuda.so" && -f "${RUNTIME_DIR}/libcublas.so.13" ]]; then
    echo "[✓] Complete CUDA runtime already present in $RUNTIME_DIR"
    exit 0
fi

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

echo "=== [2/3] Downloading llama.cpp with CUDA 13.4 (${BUILD_TAG}) ==="
LLAMA_URL="https://github.com/ggml-org/llama.cpp/releases/download/${BUILD_TAG}/llama-${BUILD_TAG}-bin-ubuntu-cuda-13.4-x64.tar.gz"
CUDART_URL="https://github.com/ggml-org/llama.cpp/releases/download/${BUILD_TAG}/cudart-llama-${BUILD_TAG}-bin-ubuntu-cuda-13.4-x64.tar.gz"

echo "  -> Fetching llama.cpp core binaries..."
curl -L -o "${TEMP_DIR}/llama.tar.gz" "$LLAMA_URL"
tar -xzf "${TEMP_DIR}/llama.tar.gz" -C "$TEMP_DIR"

echo "  -> Fetching CUDA 13.4 runtime libraries..."
curl -L -o "${TEMP_DIR}/cudart.tar.gz" "$CUDART_URL"
tar -xzf "${TEMP_DIR}/cudart.tar.gz" -C "$TEMP_DIR"

echo "=== [3/3] Assembling Standalone Runtime ==="
cp -a "${TEMP_DIR}/llama-${BUILD_TAG}/"* "$RUNTIME_DIR/"
cp -a "${TEMP_DIR}/cudart-llama-${BUILD_TAG}-bin-ubuntu-cuda-13.4-x64/"* "$RUNTIME_DIR/"
chmod +x "${RUNTIME_DIR}"/llama-*

echo "[✓] Standalone portable runtime setup complete in: $RUNTIME_DIR"
