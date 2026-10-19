# Dell G15 5530: Standalone MoE AI Runner 🚀

[![Hardware](https://img.shields.io/badge/Hardware-Dell%20G15%205530-0076CE.svg)](https://www.dell.com)
[![GPU](https://img.shields.io/badge/GPU-RTX%203050%20(6GB%20VRAM)-76B900.svg)](https://www.nvidia.com)
[![CPU](https://img.shields.io/badge/CPU-Intel%20i5--13450HX%20(6P%2B4E)-0071C5.svg)](https://www.intel.com)
[![Backend](https://img.shields.io/badge/Backend-llama.cpp%20(CUDA%2013.4)-black.svg)](https://github.com/ggml-org/llama.cpp)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A standalone, portable, high-performance runtime and runner designed to execute **massive 35B Mixture-of-Experts (MoE) AI models** (such as `Qwen3.6-35B-A3B`) at full speed on budget **6GB VRAM hardware**.

---

## 1. How It Works: The 6GB VRAM MoE Technique

Running a 35B parameter GGUF model (which weighs ~20 GB) on a 6GB VRAM laptop would normally cause an immediate **Out of Memory (OOM)** crash.

This runner implements the specialized **MoE Offloading Architecture**:
1. **Few Active Parameters (`A3B`)**: Although the model possesses 35 billion parameters in total, it only activates **~3 billion parameters** per token.
2. **`--n-cpu-moe 40`**: Evaluates all 40 MoE expert layers across system DDR5 RAM via the Intel Raptor Lake memory lanes.
3. **`-ngl 99`**: Offloads non-MoE dense layers, self-attention, and output embeddings onto the RTX 3050 CUDA cores.
4. **`--moe-cache-mib 512`**: Allocates a 512 MiB dynamic VRAM cache for frequently hit "hot" experts, eliminating CPU-GPU PCIe ping-pong overhead.
5. **`--no-mmproj-offload`**: Keeps the multimodal vision projector in host memory, preserving ~1 GB of VRAM for attention.
6. **`-fa on` & `-ctk q8_0 -ctv q8_0`**: Quantizes the KV cache to 8-bit and applies FlashAttention, drastically minimizing context VRAM requirements.
7. **`-t 6`**: Pins execution to the 6 high-throughput physical P-cores, avoiding scheduler stalls on slower E-cores.

---

## 2. Quick Start

### Step 1: Install Portable Runtime (One-Time)
```bash
./setup.sh
```
*(Downloads the self-contained CUDA 13.4 prebuilt `llama.cpp` runtime into `./runtime/` without polluting your global system packages).*

### Step 2: Add Models
Place your GGUF model files in `models/` or link them:
```bash
ln -sf /path/to/your/Qwen3.6-35B-A3B.gguf models/model.gguf
ln -sf /path/to/your/mmproj-f16.gguf models/mmproj.gguf
```

### Step 3: Run the Server
```bash
./run.sh
```

Once launched, open your web browser to:
👉 **`http://localhost:8080`**

---

## 3. Configuration (`config.env`)

You can customize runtime parameters in `config.env`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `CONTEXT_SIZE` | `8192` | Context window length in tokens |
| `THREADS` | `6` | Physical performance cores (6 P-cores on i5-13450HX) |
| `N_CPU_MOE` | `36` | Expert layers evaluated on CPU (4 full layers kept on GPU) |
| `MOE_CACHE_MIB`| `1024` | Size of hot-expert cache in fast VRAM (MiB) |
| `GGML_CUDA_REGISTER_HOST` | `1` | DMA host-pinned direct PCIe memory transfers (~20 GB/s) |
| `GGML_SCHED_PREFETCH_EXPERTS` | `1` | Dual-stream asynchronous CUDA expert prefetching |
| `PORT` | `8080` | Local web server and API port |
| `FLASH_ATTN` | `on` | FlashAttention memory optimization |

---

## 4. Live Benchmark Results (Dell G15 5530)

Measured via `llama-bench` on **`Qwen3.6-35B-A3B (Q4_K_M)`** (RTX 3050 6GB Laptop GPU + 16GB DDR5):

| Configuration | Prompt Processing (`pp128`) | Token Generation (`tg32`) | VRAM Used |
| :--- | :--- | :--- | :--- |
| **Baseline** (`-ncmoe 40`, 512M cache, unpinned) | 31.25 t/s | 6.06 t/s | ~2.8 GiB |
| **Optimized** (`-ncmoe 36`, 1024M cache, DMA pinned) | **37.01 t/s (+18%)** | **13.03 t/s (+115% / 2.15x)** | **~4.0 GiB** |

---

## 5. Hardware Profiles Integration

This runner automatically invokes [`hpc-perf`](https://github.com/ullu1221/hpc-system-profiles) upon launch, activating the `AI` profile:
- 95W sustained CPU package power (Intel RAPL).
- Zero-latency C-state sleep clamping (eliminates 1ms wake stutter).
- Dell G15 cooling fans set to performance mode.
