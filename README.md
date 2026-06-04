# Intel Arc Pro B70 + B60 (Battlemage) — LLM Inference Benchmarks

LLM inference numbers for **Intel Arc Pro B70** (Battlemage **G31**, 32 GB) and **B60** (G21, 24 GB)
on a Threadripper workstation, with **tokens-per-joule** efficiency. The original dual-B70 study
(wall power, 70B-class) is below; the newer **single-card B60 model survey** — Gemma 4, MiMo,
Nemotron, Phi-4, Granite-4, Qwen3-Coder, QwQ and more — is summarised in the leaderboard.
Dual-B70 methodology mirrors [`PMZFX/intel-arc-pro-b70-benchmarks`](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks).

## 🏆 B60 single-card leaderboard

Which model to put on the **B60** (24 GB, the efficient/console card)? Best generation backend per
model, ranked by tg/s. **Power = GPU-only (xe energy counter), active generation** — comparable to
the `t/J(GPU)` columns below, **not** the wall-power table. Q4_0 for the MoEs, Q4_K_M for dense.

| # | Model | type | size GiB | tg t/s (backend) | active W | t/J | license | best for |
|---|---|---|---|---|---|---|---|---|
| 1 | Qwen3-4B | 4B dense | 2.5 | **101.8** (SYCL) | 81 | 1.26 | Apache-2.0 | tiny & fast |
| 2 | **Qwen3-Coder-30B-A3B** | MoE 3B-act | 16.2 | **65.2** (Vulkan) | 60 | 1.09 | Apache-2.0 | 🥇 coding (fastest big) |
| 3 | MiMo-7B-RL | 7B dense | 4.4 | 64.3 (SYCL) | 86 | 0.75 | **MIT** | reasoning / watt |
| 4 | **Nemotron-3-Nano-30B-A3B** | hybrid MoE 3B-act | 17.0 | 55.5 (Vulkan) | 65 | 0.85 | NVIDIA-OM | 🥇 general big-brain |
| 5 | Nemotron-Nano-12B-v2 | hybrid 12B | 7.0 | 42.8 (SYCL) | 88 | 0.49 | NVIDIA-OM | fast hybrid |
| 6 | Phi-4-reasoning-plus | 14B dense | 8.4 | 39.2 (SYCL) | 92 | 0.43 | **MIT** | reasoning (MIT) |
| 7 | Qwen3.6-35B-A3B | MoE 3B-act | 20.0 | 37.3 (Vulkan) | 60 | 0.62 | Apache-2.0 | general MoE |
| 8 | Gemma 4 12B | 12B dense | 6.9 | 36.4 (SYCL) | 85 | 0.43 | Gemma | dense daily-driver |
| 9 | Granite-4.0-H-Small | hybrid MoE 9B-act | 17.2 | 34.3 (Vulkan) | 74 | 0.46 | Apache-2.0 | 9B-active rung |
| 10 | QwQ-32B | 32B dense | 18.5 | 17.3 (SYCL) | 92 | 0.19 | Apache-2.0 | dense reasoner (wants B70) |

**Picks:** general → **Nemotron-3-Nano-30B-A3B** · coding → **Qwen3-Coder-30B-A3B** (also the fastest
big model the card holds) · MIT reasoning → **Phi-4-reasoning-plus** / **MiMo-7B-RL**.

### Backend rule (B60): **dense → SYCL, 30B-class MoE → Vulkan**
Every dense model (incl. the 7B/12B/14B and the 12B hybrid) generates fastest on the **SYCL** build;
every 30B-class **MoE** generates fastest on **Vulkan**. The discriminator is MoE-vs-dense, *not*
Mamba — Qwen3-Coder is a pure MoE (no SSM) and still flips to Vulkan. *(This is **B60-specific**: on
the **B70**, SYCL wins across the board — see the main table — so the MoE→Vulkan flip is the smaller
G21's behaviour. SYCL-MoE flags were left at llama-bench defaults.)*

### Past the VRAM wall (don't fit 2 cards)
Giant MoEs benchmarked-by-spec only — need a big-RAM serving node, not more B70s:
**MiMo-V2.5** (~310B, 87–187 GB), **MiniMax / GLM-4.6 / DeepSeek-V3 / Kimi-K2** (456B–1T).
The true **12B-active** rung (GLM-4.5-Air, Nemotron-Super-120B-A12B) needs the **3rd B70 (96 GB)**.

## Study index

| Study | What | File |
|---|---|---|
| B60 vs B70 | single-card head-to-head (gen speed, power, POST) | [`results/b60-vs-b70.md`](results/b60-vs-b70.md) |
| Gemma 4 12B | new `gemma4` arch on the B60 | [`results/gemma4-12b-b60.md`](results/gemma4-12b-b60.md) |
| MiMo | 7B-RL fits; V2.5 (310B) past the wall | [`results/mimo-b60.md`](results/mimo-b60.md) |
| Nemotron ×3 | 30B-A3B = B60 champ; 12B hybrid; 49B on 2 cards | [`results/nemotron-b60.md`](results/nemotron-b60.md) |
| Permissive round | Phi-4 (MIT), Granite-4, Qwen3-Coder, QwQ-32B | [`results/permissive-models-b60.md`](results/permissive-models-b60.md) |
| MoE ladder | speed vs active params (3B → 12B-active) | [`results/moe-ladder-results.md`](results/moe-ladder-results.md) |
| Kimi-Linear | KDA linear attention — flat long-context speed | [`results/kimi-linear-results.md`](results/kimi-linear-results.md) |
| Step-3.7 Flash | big MoE, RAM-bound — needs 96 GB | [`results/step37-flash-findings.md`](results/step37-flash-findings.md) |
| Arch gauntlet | mamba / rwkv / jamba (jamba broken) | [`results/arch-gauntlet-results.md`](results/arch-gauntlet-results.md) |
| Quality-per-joule | efficiency sweep | [`results/quality-per-joule-results.md`](results/quality-per-joule-results.md) |
| Context tax · Embeddings · Spec-decode | misc | [`results/`](results/) |

## Hardware

| | |
|---|---|
| GPUs | **Arc Pro B70** (G31, `8086:e223`, 32 GB) + **Arc Pro B60** (G21, `8086:e211`, 24 GB) — *2nd B70 planned; the B60 study uses the B60 alone* |
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

## Quant sweep — Qwen3.6-35B-A3B (same firmware)

Which quant is fastest/most efficient on the B70? Single B70 where it fits; **Q8_0 needs two cards** (35 GB > one card's ~31 GB usable). Q4_0 from bartowski, the rest from Unsloth. Includes GPU-only power (xe hwmon) for direct comparison with GPU-power benchmarks.

| Backend | Quant | Size (GiB) | GPUs | pp512 | tg128 | wall W | GPU W | t/J(wall) | t/J(GPU) |
|---|---|---|---|---|---|---|---|---|---|
| sycl | **UD-Q4_K_M** | 20.6 | 1 | 972 | **69.9** | 293 | 81 | 0.239 | **0.863** |
| sycl | Q4_0 | 19.4 | 1 | 872 | 62.6 | 324 | 100 | 0.193 | 0.626 |
| sycl | UD-Q6_K | 27.3 | 1 | 706 | 51.3 | 305 | 83 | 0.168 | 0.618 |
| sycl | Q8_0 | 34.4 | 2 | 627 | 46.2 | 289 | 80 | 0.160 | 0.578 |
| vulkan | Q4_0 | 19.4 | 1 | **1377** | 50.9 | 319 | 108 | 0.160 | 0.471 |
| vulkan | UD-Q4_K_M | 20.6 | 1 | 1310 | 39.7 | 318 | 105 | 0.125 | 0.378 |
| vulkan | UD-Q6_K | 27.3 | 1 | 1093 | 34.2 | 321 | 103 | 0.107 | 0.332 |
| vulkan | Q8_0 | 34.4 | 2 | 1132 | 13.9 | 277 | 71 | 0.050 | 0.196 |

### Quant findings

- **Best overall: UD-Q4_K_M on SYCL** — fastest generation (69.9 tg/s), most efficient (0.863 t/J GPU), fits one card. The production pick.
- **Best quant differs by backend:** on **SYCL** Q4_K_M > Q4_0 for tg; on **Vulkan** Q4_0 > Q4_K_M (and Q4_0 gives Vulkan's top pp, 1377). Choose per backend.
- **Higher precision costs speed:** Q6_K / Q8_0 trade ~25–35% tg for quality. Q8_0 (2-GPU SYCL) still does 46 tg/s — viable when quality matters.
- **SYCL dominates multi-GPU:** Q8_0 across two cards — SYCL **46.2** vs Vulkan **13.9** tg/s (3.3×); Vulkan's layer-split serialisation hurts badly (matches the 70B result).

### vs PMZFX — fair, same metric (GPU-only power)

Closest comparable — 35B-A3B MoE, Q4_K_M, SYCL, 1 GPU:

| | tg128 t/s | GPU W | t/J(GPU) |
|---|---|---|---|
| **This box (Qwen3.6)** | **69.9** | 81 | **0.863** |
| [PMZFX](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks) (Qwen3.5) | 54.5 | 92 | 0.59 |

**~28% faster generation and ~46% more efficient** on a like-for-like (GPU-only power) basis. *(Caveat: newer model version + compiler 2026.0 vs 2025.3 + newer llama.cpp commit — not a fully controlled comparison; different CPU/board/BIOS too.)*

## Precision & 1-bit study

Filling the ladder (INT5 / BF16) plus a for-fun 1-bit run. All single-B70, SYCL unless noted.

**Q5_K_M slots into the 35B-A3B ladder — a sweet spot** (nearly Q4_K_M speed at higher quality):

| Quant | Size (GiB) | tg128 | t/J(GPU) |
|---|---|---|---|
| Q4_K_M | 20.6 | 69.9 | 0.863 |
| **Q5_K_M** | 24.1 | **68.3** | 0.742 |
| Q6_K | 27.3 | 51.3 | 0.618 |
| Q8_0 (2-GPU) | 34.4 | 46.2 | 0.578 |

→ Q5_K_M costs **~2 %** tg over Q4_K_M for a precision bump — far closer to Q4 than to Q6.

**BF16 vs Q4_K_M (Qwen3-4B) — the precision ceiling:**

| Backend | Quant | pp512 | tg128 | GPU W | t/J(GPU) |
|---|---|---|---|---|---|
| sycl | BF16 | 1433 | 52.5 | 128 | 0.410 |
| sycl | Q4_K_M | 3280 | **115.7** | 144 | 0.803 |

→ **Q4 is 2.2× faster and ~2× more efficient than BF16** — decode is bandwidth-bound and BF16 is 4× the bytes; BF16 buys you lossless quality at ~half the speed. *(Vulkan edges SYCL on BF16 decode (58.8 vs 52.5), but SYCL crushes the quantized path (115.7 vs 79.9) — its MMVQ/reorder win.)*

**A 70B at ~1.7-bit on a SINGLE B70** — Llama-3.3-70B `UD-IQ1_M` (16 GiB):

| Config | GPUs | tg128 |
|---|---|---|
| Llama-3.3-70B Q4_K_M | 2 | 11.7 |
| **Llama-3.3-70B IQ1_M** | **1** | **11.0** |

→ The 1-bit quant fits a **70B on one card** and runs at ~the speed of Q4 on **two** cards — frees a whole GPU, at a real quality cost. *(FP8 — the other sub-Q4 path — needs vLLM-XPU, which is parked; see below.)*

## Production serving notes (Battlemage SYCL)

These benchmarks use `llama-bench`; for a long-lived **server** (Ollama / `llama-server`) on the B70 the
following runtime flags matter for stability — credit to
[Hal9000AIML/arc-pro-b70-ubuntu-gpu-speedup-bugfixes](https://github.com/Hal9000AIML/arc-pro-b70-ubuntu-gpu-speedup-bugfixes):

- **`GGML_SYCL_DISABLE_OPT=1`** — *required* for MoE on SYCL or slot-init hangs at startup (~5% cost on dense).
- **`-fa 0`** on SYCL + MoE — SYCL flash-attention crashes on MoE (Vulkan FA is fine).
- **`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`** — Level Zero caps allocations at 4 GB; needed for large-context KV.
- **Never set `SYCL_CACHE_PERSISTENT=1`** — kernel-cache corrupts on B70 (SEGV at next boot).
- Build with **`-DGGML_SYCL_HOST_MEM_FALLBACK=ON`** for graceful VRAM-exhaustion fallback.

## vLLM-XPU note

**vLLM-XPU is blocked on this box (Fedora 44, kernel 7.0.10) — parked.** `torch.xpu` device-init fails with
`UR_RESULT_ERROR_UNKNOWN` **consistently** across the `intel/vllm` *and*
[`kyuz0/intel-b70-vllm-toolbox`](https://github.com/kyuz0/intel-b70-ai-toolboxes) containers — every fix tried
(rootless/privileged, `ZES_ENABLE_SYSMAN`, FLAT/COMPOSITE hierarchy, `ONEAPI_DEVICE_SELECTOR=*:gpu` +
`TRITON_INTEL_DEVICE_ARCH=20.2.0` per [crazydart](https://github.com/crazydart/vllm-b70)). Bare-metal
(crazydart wheels) fails differently: Fedora 44 ships no Level Zero *loader*, so L0 enumerates nothing.
Notably `sycl-ls` sees both cards via Level Zero in-container, and **OpenCL + our SYCL-over-OpenCL llama.cpp
work fine** — so it's torch-xpu's specific UR/L0 path. The only common factor is the host **xe driver / kernel
7.0.10**; the working recipes all target **7.0.0**, pointing to a xe regression. **Revisit on a kernel/xe
update** (the firmware-watch catches those). Narrow value anyway: vLLM decode on B70 is ~5 t/s — it only wins
**high-concurrency prefill**. For single-stream, **llama.cpp-SYCL** (this repo) is the backend; it beats vLLM
~4.3× there per PMZFX/Hal9000. When torch-xpu is fixed, **kyuz0's toolbox is the fastest path back in**.
</content>
</invoke>
