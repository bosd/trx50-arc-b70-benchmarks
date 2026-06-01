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

## Results — firmware: BIOS 14.10 · GuC 70.65.0 · kernel 7.0.10 · Mesa 26.0.7

`llama-bench -p 512 -n 128`; wall power via Shelly Plug S G3; `t/J = tg128 ÷ avg W`.

| Model | Backend | Type | Quant | Size (GiB) | GPUs | pp512 t/s | tg128 t/s | avg W | t/J |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | vulkan | MoE 3B-act | UD-Q4_K_M | 20.6 | 1 | **1314.7** | 39.7 | 319 | 0.124 |
| Qwen3.6-35B-A3B | vulkan | MoE 3B-act | UD-Q4_K_M | 20.6 | 2 | 1290.8 | 27.9 | 316 | 0.088 |
| Qwen3.6-35B-A3B | sycl   | MoE 3B-act | UD-Q4_K_M | 20.6 | 1 | 973.5 | **69.9** | 307 | **0.228** |
| Qwen3.6-35B-A3B | sycl   | MoE 3B-act | UD-Q4_K_M | 20.6 | 2 | 934.3 | 68.7 | 324 | 0.212 |
| DeepSeek-R1-Distill-70B | vulkan | dense 70B | Q4_K_M | 39.6 | 2 | 229.0 | 5.0 | 327 | 0.015 |
| DeepSeek-R1-Distill-70B | sycl   | dense 70B | Q4_K_M | 39.6 | 2 | 345.9 | **11.7** | 361 | 0.032 |
| Llama-3.3-70B-Instruct | vulkan | dense 70B | Q4_K_M | 39.6 | 2 | 230.2 | 5.0 | 327 | 0.015 |
| Llama-3.3-70B-Instruct | sycl   | dense 70B | Q4_K_M | 39.6 | 2 | 345.3 | **11.7** | 376 | 0.031 |

### Key findings

- **SYCL wins token generation decisively** — **1.8×** (Qwen MoE) to **2.3×** (dense 70B) over Vulkan, at similar/lower power → **~2× tokens-per-joule**. For the production 70B workload: **SYCL 11.7 vs Vulkan 5.0 tg/s**.
- **Prompt processing splits by architecture:** SYCL wins on the **dense 70B** (345 vs 229), Vulkan wins on the **MoE** (1314 vs 973 — its `KHR_coopmat` path is strong on the sparse prompt).
- **Multi-GPU only helps when the model doesn't fit on one card.** Splitting the 20 GB Qwen across two B70s *hurt* tg (Vulkan 39.7→27.9) — `-sm layer` serialises the GPUs. Run fits-on-one models on a **single** card; reserve dual-GPU for the 70Bs.
- **Verdict: SYCL is the production backend** here. Single B70 for ≤~28 GB models; dual-B70 SYCL for 70B (~11–12 tg/s — usable for drafting/helpdesk).
- Device handling that actually engages the split: Vulkan needs `GGML_VK_VISIBLE_DEVICES=1,2 -sm layer` (the `-dev` flag does **not** split); SYCL splits with plain `-sm layer`.
- *IPEX-LLM excluded — Intel archived it Jan 2026; llama.cpp-SYCL is the supported XMX path.*
</content>
</invoke>
