# Xiaomi MiMo on the TRX50 — what fits, what doesn't (MIT-licensed reasoners)

MiMo (Xiaomi, **MIT licensed**) has split into two very different things, and only one
of them fits this box:

| Model | arch | params | smallest usable GGUF | runs on TRX50? |
|---|---|---|---|---|
| **MiMo-7B-RL** | `mimo` | 7B dense | Q4_K_M 4.4 GiB | ✅ fits one B60 with room to spare |
| **MiMo-V2.5** | `mimo2` | **~310B MoE** (256 exp / 8 active) | IQ1_S 87 GB · Q4 187 GB | ❌ VRAM/RAM wall |

Both archs **are** supported by the box's llama.cpp (`LLM_ARCH_MIMO` + `mimo2` are compiled
in, build `b50-5aba536`) — so V2.5 not running is purely a **memory** problem, not a support gap.

---

## MiMo-7B-RL — the runnable one (and it's good)

The original MiMo reasoning model: a 7B dense that was RL-tuned to punch far above its weight
on math/code. GGUF: `quantflex/MiMo-7B-RL-nomtp-GGUF` (the **`nomtp`** build strips the
multi-token-prediction head that plain llama.cpp can't execute — use this, not the raw 7B GGUF).
Q4_K_M, 36 layers, 32K context. Benchmarked on the **B60** (card0 / SYCL0 / Vulkan0), same
methodology as the other B60 runs (active-generation watts from the xe energy counter).

| Backend | pp512 t/s | tg128 t/s | size GiB | active W | t/J |
|---|---|---|---|---|---|
| **SYCL** | **1654** | **64.3** | 4.4 | 86 | 0.75 |
| Vulkan | 1382 | 35.6 | 4.4 | 75 | 0.48 |

- **64 tg/s on SYCL** — the fastest generator on the B60 so far (it's a 7B, and dense-small is
  what this card loves). ~1.8× Vulkan, same SYCL-wins pattern as everything else.
- **1654 t/s prefill** — snappy even with the long reasoning prompts these models like to emit.
- **4.4 GiB** leaves ~17 GiB free → the full 32K context fits on the single card with margin.
- **Coherence ✓** — trick question *"17 sheep, all but 9 run away, then buys 4× what's left"*:
  it correctly read "all but 9" → 9 remain, then 9 + 4×9 = **45**, with the self-checking
  ("Wait, let me go through it again…") that's the RL reasoner's signature. Clean output, no garbage.

**B60 small-model leaderboard (SYCL, generation):**

| Model | params | tg t/s | active W | t/J |
|---|---|---|---|---|
| MiMo-7B-RL | 7B dense | 64.3 | 86 | 0.75 |
| Gemma 4 12B | 12B dense | 36.4 | 85 | 0.42 |
| Qwen3-4B | 4B dense | 101.8 | 81 | 1.26* |

\*Qwen3-4B from the b60-vs-b70 run. MiMo-7B sits exactly where a 7B should — between the 4B and
the 12B — and is the strongest *reasoning-per-watt* option of the three for math/code work.

**Verdict:** MiMo-7B-RL is another ideal B60 resident — small, MIT, strong reasoning, full
context on one efficient card. Same conclusion as Gemma 4 12B: the kind of model that makes
the B60 worth keeping as a standalone always-up assistant, with the B70s reserved for the
big stuff the B60 can't hold.

---

## MiMo-V2.5 — the "ranks very high on intelligence" flagship — does NOT fit

This is the model topping the leaderboards, and it's a monster: **`mimo2`, ~310B-param MoE**
(48 layers, **256 routed experts, 8 active/token**, 1M context). GGUF sizes (unsloth):

- BF16 **620 GB**, Q8 329 GB, **Q4_K_M 187 GB**, Q2_K_XL 103 GB, **smallest (IQ1_S) 87 GB**.

Against the TRX50's budget:
- **GPU:** 2× B70 = 64 GB (~57 GB Vulkan-usable). Even the **87 GB IQ1_S overflows two cards**,
  and IQ1 is a heavily-lobotomized quant. A *usable* quant (Q4 ≈ 187 GB) needs **6× B70**.
- **RAM offload:** impossible — the box has **30 GB RAM** today. No CPU/RAM fallback path exists
  for a 90–187 GB model.

So V2.5 is a **multi-GPU-server model, not a TRX50 model**. The interesting near-term datapoint:
its smallest IQ-quant (~87 GB) is the first model that would *need the 3rd B70 (96 GB)* just to
load — and even then only at a brutal IQ1. It's the cleanest illustration yet of this box's thesis:
**VRAM is the wall.** The intelligence MiMo-V2.5 is famous for is real, but it lives on the other
side of the wall the 2-card TRX50 sits behind. (If the goal is to actually run V2.5 someday,
the answer isn't more B70s in this box — it's a different, big-RAM serving node.)
