#!/usr/bin/env bash
# ==============================================================================
# Standalone Portable MoE Model Runner
# Tuned for Qwen3.6-35B-A3B on RTX 3050 (6GB VRAM) + 16GB DDR5
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${SCRIPT_DIR}/runtime"

# Source default settings from config.env
if [[ -f "${SCRIPT_DIR}/config.env" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/config.env"
fi

# Ensure runtime exists
if [[ ! -f "${RUNTIME_DIR}/llama-server" || ! -f "${RUNTIME_DIR}/libggml-cuda.so" ]]; then
    echo "[!] Runtime missing. Running portable installer..."
    "${SCRIPT_DIR}/setup.sh"
fi

# Check model
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[!] Target model not found at: $MODEL_PATH"
    echo "    Place your GGUF file in ${SCRIPT_DIR}/models/ or update MODEL_PATH in config.env."
    exit 1
fi

echo "================================================================="
echo "   Dell G15 5530: Standalone MoE AI Runner (RTX 3050 6GB VRAM)"
echo "================================================================="
echo "  • Model:        $(basename "$MODEL_PATH")"
if [[ -n "${MMPROJ_PATH:-}" && -f "$MMPROJ_PATH" ]]; then
    echo "  • Vision:       $(basename "$MMPROJ_PATH")"
fi
echo "  • VRAM Offload: Non-MoE Attention + Head on RTX 3050 (CUDA)"
echo "  • MoE Offload:  $N_CPU_MOE Expert Layers on CPU DDR5 RAM"
echo "  • Hot Cache:    $MOE_CACHE_MIB MiB VRAM Expert Cache"
echo "  • P-Cores:      $THREADS Physical P-Cores Dedicated"
echo "  • KV Cache:     Quantized $CACHE_TYPE_K / $CACHE_TYPE_V (FlashAttention: $FLASH_ATTN)"
echo "  • Web UI:       http://${HOST}:${PORT}"
echo "================================================================="

# Automatically elevate system performance profile if hpc-perf is available
if command -v hpc-perf &>/dev/null; then
    echo "[*] Activating HPC AI Performance Profile (low latency, 95W RAPL, fans)..."
    sudo -n /usr/local/bin/hpc-perf ai 2>/dev/null || true
fi

export LD_LIBRARY_PATH="${RUNTIME_DIR}:${LD_LIBRARY_PATH:-}"

EXTRA_ARGS=()
if [[ -n "${MMPROJ_PATH:-}" && -f "$MMPROJ_PATH" ]]; then
    EXTRA_ARGS+=(--mmproj "$MMPROJ_PATH")
    if [[ "${NO_MMPROJ_OFFLOAD:-false}" == "true" ]]; then
        EXTRA_ARGS+=(--no-mmproj-offload)
    fi
fi

if [[ -n "${MOE_CACHE_MIB:-}" && "$MOE_CACHE_MIB" != "0" ]]; then
    EXTRA_ARGS+=(--moe-cache-mib "$MOE_CACHE_MIB")
fi

echo "[*] Launching llama-server..."
exec "${RUNTIME_DIR}/llama-server" \
    -m "$MODEL_PATH" \
    -ngl "$N_GPU_LAYERS" \
    --n-cpu-moe "$N_CPU_MOE" \
    -c "$CONTEXT_SIZE" \
    -t "$THREADS" \
    -b "$BATCH_SIZE" \
    -ub "$UBATCH_SIZE" \
    -fa "$FLASH_ATTN" \
    -ctk "$CACHE_TYPE_K" \
    -ctv "$CACHE_TYPE_V" \
    --host "$HOST" \
    --port "$PORT" \
    "${EXTRA_ARGS[@]}" \
    "$@"
