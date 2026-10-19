#!/usr/bin/env bash
# ==============================================================================
# Standalone Portable MoE Model Runner
# Tuned for Qwen3.6-35B-A3B on RTX 3050 (6GB VRAM) + 16GB DDR5
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
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
echo "  • VRAM Offload: Non-MoE Attention + Head + 4 MoE Layers on RTX 3050"
echo "  • MoE Offload:  $N_CPU_MOE Expert Layers on CPU DDR5 RAM"
echo "  • Hot Cache:    $MOE_CACHE_MIB MiB VRAM Expert Cache"
echo "  • DMA Speedup:  Host-Registered Pinned DDR5 Direct DMA (GGML_CUDA_REGISTER_HOST=1)"
echo "  • Prefetch:     Async Double-Buffered Stream (GGML_SCHED_PREFETCH_EXPERTS=1)"
echo "  • CPU Threads:  $THREADS P-Cores (Generation) / ${THREADS_BATCH:-$THREADS} Threads (Prefill Batch)"
if [[ -n "${SPEC_TYPE:-}" && "$SPEC_TYPE" != "none" ]]; then
    echo "  • Speculative:  ${SPEC_TYPE} (Draft Engine Active)"
fi
echo "  • KV Cache:     Quantized $CACHE_TYPE_K / $CACHE_TYPE_V (FlashAttention: $FLASH_ATTN, Reuse: ${CACHE_REUSE:-0})"
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
    if [[ -n "${IMAGE_MIN_TOKENS:-}" ]]; then
        EXTRA_ARGS+=(--image-min-tokens "$IMAGE_MIN_TOKENS")
    fi
fi

if [[ -n "${MOE_CACHE_MIB:-}" && "$MOE_CACHE_MIB" != "0" ]]; then
    EXTRA_ARGS+=(--moe-cache-mib "$MOE_CACHE_MIB")
fi

if [[ -n "${THREADS_BATCH:-}" ]]; then
    EXTRA_ARGS+=(-tb "$THREADS_BATCH")
fi

if [[ -n "${CACHE_REUSE:-}" ]]; then
    EXTRA_ARGS+=(--cache-reuse "$CACHE_REUSE")
fi

if [[ -n "${REASONING_FORMAT:-}" ]]; then
    EXTRA_ARGS+=(--reasoning-format "$REASONING_FORMAT")
fi

if [[ "${REASONING_PRESERVE:-false}" == "true" ]]; then
    EXTRA_ARGS+=(--reasoning-preserve)
fi

if [[ -n "${SPEC_TYPE:-}" && "$SPEC_TYPE" != "none" ]]; then
    EXTRA_ARGS+=(--spec-type "$SPEC_TYPE")
    if [[ "$SPEC_TYPE" == "ngram-mod" ]]; then
        EXTRA_ARGS+=(
            --spec-ngram-mod-n-match "${SPEC_NGRAM_MATCH:-24}"
            --spec-ngram-mod-n-min "${SPEC_NGRAM_MIN:-12}"
            --spec-ngram-mod-n-max "${SPEC_NGRAM_MAX:-48}"
        )
    fi
fi

if [[ -n "${TEMPERATURE:-}" ]]; then
    EXTRA_ARGS+=(--temp "$TEMPERATURE")
fi
if [[ -n "${TOP_P:-}" ]]; then
    EXTRA_ARGS+=(--top-p "$TOP_P")
fi
if [[ -n "${TOP_K:-}" ]]; then
    EXTRA_ARGS+=(--top-k "$TOP_K")
fi
if [[ -n "${MIN_P:-}" ]]; then
    EXTRA_ARGS+=(--min-p "$MIN_P")
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
