# Intel Arc Pro B70 (Battlemage G31) — LLM Inference Benchmarks

Dual **Intel Arc Pro B70** (Xe2 / Battlemage **G31**, 32 GB GDDR6 each) LLM inference numbers on a
Threadripper workstation, with **wall-power efficiency (tokens/joule)** measured at the plug.
Methodology mirrors [`PMZFX/intel-arc-pro-b70-benchmarks`](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks)
so results are directly comparable.

## Hardware

| | |
|---|---|
| GPUs | 2× Intel **Arc Pro B70** (G31, `8086:e223`), 32 GB each, 256-bit, 608 GB/s, 230 W cap |
| Board | ASRock **TRX50 WS** (PCIE3 = Gen5 x16, PCIE5 = Gen4 x8) |
| CPU | AMD Ryzen Threadripper **9960X** (24c / 48t) |
| RAM | 64 GB DDR5-4800 ECC RDIMM *(currently 1× 32 GB — second stick RMA in progress)* |
| OS | Fedora Server 44, kernel 7.0.10, Mesa 26.0.7 |
| Model SSD | Samsung PM1725a 3.2 TB (U.2) |
| Power meter | Shelly Plug S **Gen3** (local API, true wall draw) |

## Backends

- **Vulkan** — Mesa Anv, general compute (no XMX). `-DGGML_VULKAN=ON`.
- **SYCL** — Intel oneAPI, **uses XMX matrix engines**. Built with **`-DGGML_SYCL_F16=ON`**
  (FP16 XMX path → ~2.4× prompt-processing per [llama.cpp #21517](https://github.com/ggml-org/llama.cpp/issues/21517)).
- *IPEX-LLM is intentionally excluded — Intel archived the repo in Jan 2026.*

## Methodology

- Tool: `llama-cpp` **`llama-bench`**, standard `-p 512 -n 128` (pp512 / tg128).
- GPU configs: **1 GPU** (single device) vs **2 GPU** (tensor split, `-sm layer`).
- Power: Shelly wall draw sampled across each run → **avg W**. Efficiency **`t/J = tg128 t/s ÷ avg W`**.
- Each (model × backend × GPU-config × quant) row is one `llama-bench` invocation; raw JSON in [`raw/`](raw/).
- Results are grouped by **GPU firmware version** (we benchmark, update firmware, re-run).

## Quantization notes (Battlemage / SYCL)

- **Q4_K_M / UD-Q4_K_M** — best tg throughput (53–64 % of mem bandwidth); the comparison standard.
- **Q4_0** — well-optimised SYCL reorder kernels; fast.
- **Q8_0** — recently fixed (66 % bandwidth, now > Q6_K); good quality-ceiling data point.
- i-quants (IQ4_XS…) skipped — compute-heavy to dequant, and 32 GB VRAM makes them unnecessary.

## Models under test

| Model | Type | Quant | Approx size | Target |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | MoE (3B active) | UD-Q4_K_M | 22 GB | single-GPU |
| DeepSeek-R1-Distill-Llama-70B | dense | Q4_K_M | ~42 GB | dual-GPU |
| Llama-3.3-70B-Instruct | dense | Q4_K_M | ~42 GB | dual-GPU |

## Results

| Model | Type | Quant | Size (GiB) | GPUs | pp512 t/s | tg128 t/s | avg W | t/J |
|---|---|---|---|---|---|---|---|---|
| _pending first run_ | | | | | | | | |

_Status: hardware up, toolchain building, models downloading. Results land here after the first run._
</content>
</invoke>
