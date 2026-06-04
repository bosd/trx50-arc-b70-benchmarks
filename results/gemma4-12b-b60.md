# Gemma 4 12B (Q4_K_M) on the Arc Pro B60 — is the card viable for this model?

**Question:** is it worth keeping the B60 to run Google's new **Gemma 4 12B**?
**Short answer: yes — this is the B60's sweet spot.** A 12B dense model in Q4 is ~7 GB,
fits the B60's 24 GB with ~14 GB to spare for context, runs coherently on the brand-new
`gemma4` arch, and generates at comfortably-interactive speed while sipping ~85 W.

Model: `ggml-org/gemma-4-12B-it-GGUF` → `gemma-4-12B-it-Q4_K_M.gguf` (6.9 GiB).
Card: B60 (G21, 24480 MiB / ~21.5 GiB usable) = card0 / SYCL0 / Vulkan0.
llama.cpp `b50-5aba536` (Jun 1 2026) — has `src/models/gemma4.cpp`, arch supported out of the box.
Power = B60 xe energy counter (`card0/.../energy2_input`), averaged over **active generation only**
(samples >8 W; the whole-run average is misleading because the 7 GB model-load phase idles the GPU near 0 W).

## Numbers (single card, B60)

| Backend | pp512 t/s | tg t/s | active W | peak W | size GiB | t/J |
|---|---|---|---|---|---|---|
| **SYCL** | **1021** | **36.4** | 85 | 94 | 6.9 | 0.42 |
| Vulkan | 525 | 19.4 | 73 | 76 | 6.9 | 0.27 |

- **SYCL is the path: ~36 tg/s and ~2× the prefill of Vulkan.** Same story as every other
  model on these cards — run Gemma 4 on the SYCL build.
- **36 tg/s** is ~3× typical reading speed → fully interactive for a chat/assistant load.
- **1021 tg/s prefill** chews through long prompts (RAG, code files) without a stall.
- **Fit:** 6.9 GiB weights leave **~14.6 GiB** for KV cache → Gemma 4's long context runs
  on this **one card**, no tensor-split, no B70 needed.
- **~85 W active** — the card barely warms up; this is the efficiency point the B60 was bought for.

## Coherence check (new arch — had to verify it isn't garbage like Jamba was)

Prompt: *"Explain in 3 sentences why MoE models are more power-efficient than dense models of the same total size."*
Output (SYCL, B60): correct and coherent — *"MoE models have many parameters, but only a subset
(experts) is activated for each input token. Dense models activate all parameters for every token…"*
✓ `gemma4` arch generates clean text on the B60.

## Verdict — keep the B60 for exactly this

Gemma 4 12B is the **ideal B60 workload**: a single mid-size dense model that fits in 24 GB
with full-context headroom, served from one efficient card. It does **not** want the B70's
extra VRAM (it never gets near 24 GB) and it does **not** want a tensor-split (a 12B in 7 GB
runs best pinned to one card). This is the standalone-assistant role the B60-vs-B70 study
already pointed it at — and Gemma 4 confirms it: the B60 runs the model the TRX50's big
B70s would be wasted on.

Where the B70 still wins is the heavy stuff the B60 *can't* hold — 70B dense, 47 GB MoE,
Step-3.7-class models. For "a smart 12B that's always up," the B60 is the right card and
SYCL is the right backend.
