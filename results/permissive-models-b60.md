# Permissive / MIT round on the B60 — Phi-4, Granite-4, Qwen3-Coder, QwQ-32B

Four open-license models, single-card B60, same methodology (active-generation watts from the
xe energy counter; SYCL vs Vulkan). All four loaded and generated coherently; archs `phi3`
(Phi-4), `granitehybrid` (Granite 4), `qwen3moe` (Coder) and `qwen2` (QwQ) are all in
llama.cpp `b50-5aba536`.

| Model | license | arch | quant/size | best tg | t/J |
|---|---|---|---|---|---|
| **Qwen3-Coder-30B-A3B** | Apache-2.0 | qwen3moe (MoE 3B-act) | Q4_0 16.2 GiB | **65.2 (Vk)** | 1.09 |
| Phi-4-reasoning-plus | **MIT** | phi3 (14B dense) | Q4_K_M 8.4 GiB | 39.2 (SYCL) | 0.43 |
| Granite-4.0-H-Small | Apache-2.0 | granitehybrid (32B/9B-act) | Q4_0 17.2 GiB | 34.3 (Vk) | 0.46 |
| QwQ-32B | Apache-2.0 | qwen2 (32B dense) | Q4_K_M 18.5 GiB | 17.3 (SYCL) | 0.19 |

Full per-backend numbers:

| Model | backend | pp512 | tg128 | active W | t/J |
|---|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | sycl | 751 | 44.3 | 50 | 0.886 |
| Qwen3-Coder-30B-A3B | **vulkan** | 989 | **65.2** | 60 | 1.087 |
| Phi-4-reasoning-plus | **sycl** | 1022 | **39.2** | 92 | 0.426 |
| Phi-4-reasoning-plus | vulkan | 738 | 21.3 | 79 | 0.270 |
| Granite-4.0-H-Small | sycl | 486 | 22.4 | 62 | 0.361 |
| Granite-4.0-H-Small | **vulkan** | 608 | **34.3** | 74 | 0.464 |
| QwQ-32B | **sycl** | 472 | **17.3** | 92 | 0.188 |
| QwQ-32B | vulkan | 329 | 9.4 | 79 | 0.119 |

(First clean Phi-4 run was contaminated by 3 concurrent downloads — 17.7 tg/s; re-run idle gives 39.2.
Lesson: never benchmark while the disk is saturated.)

---

## The headline: a robust backend rule falls out — **dense → SYCL, big MoE → Vulkan**

Across the *whole campaign* the B60 now shows a clean split on which backend generates faster:

| dense (SYCL wins) | tg SYCL | tg Vk | | large MoE (Vulkan wins) | tg Vk | tg SYCL |
|---|---|---|---|---|---|---|
| Qwen3-4B | 101.8 | 55.4 | | Qwen3-Coder-30B-A3B | 65.2 | 44.3 |
| MiMo-7B-RL | 64.3 | 35.6 | | Nemotron-3-Nano-30B-A3B | 55.5 | 51.5 |
| Nemotron-12B-v2 (hybrid) | 42.8 | 22.4 | | Qwen3.6-35B-A3B | 37.3 | — |
| Phi-4 14B | 39.2 | 21.3 | | Granite-4.0-H-Small | 34.3 | 22.4 |
| Gemma 4 12B | 36.4 | 19.4 | | | | |
| QwQ-32B | 17.3 | 9.4 | | | | |

**Six dense models all prefer SYCL; four 30B-class MoEs all prefer Vulkan.** The discriminator
is **MoE-vs-dense, not Mamba**: Qwen3-Coder is a *pure* MoE (no SSM) and still flips to Vulkan,
while the Nemotron-12B *hybrid* (Mamba layers, but dense-ish) stays on SYCL. This **corrects the
earlier "SSM favours Vulkan" guess in `nemotron-b60.md`** — it's the sparse expert-gather, not the
state-space layers. Likely cause: the many small expert matmuls in a 30B MoE hit lower per-op
dispatch overhead on Vulkan, whereas SYCL/OpenCL's tuned GEMM wins the big dense matmuls.
**Practical rule for serving on the B60: dense model → SYCL build; 30B-class MoE → Vulkan build.**

## Per-model notes

- **Qwen3-Coder-30B-A3B — the fastest big model on the B60 (65 t/s) and the one you'd actually
  self-host.** Apache-2.0, coding-specialized, fits at 16 GiB with context room, best t/J (1.09) of
  any large model on the card. Wrote a correct `is_palindrome` first try. **This is the B60's
  coding-assistant pick** — purpose-built *and* the quickest 30B-class model measured.
- **Phi-4-reasoning-plus — best MIT reasoner that fits small.** 39 tg/s, **faster than Gemma 4 12B
  despite being a bigger 14B**, MIT-licensed, strong structured reasoning (got 80 km/h cleanly).
  A great always-up reasoning resident next to MiMo-7B-RL.
- **Granite-4.0-H-Small — the ~9B-active rung, finally measured.** IBM's Apache-2.0 hybrid
  Mamba-MoE runs fully on the Arc GPU (34 tg/s Vk). At **9B active** it lands *below* the 3B-active
  MoEs (Nemotron 55, Qwen-Coder 65) and above where a dense 32B would sit — exactly the moe-ladder
  law ("speed tracks active params"). It's the closest this 2-card box can get to the 12B-active
  rung the ladder left open (which still needs the 3rd B70). Coherent, clean.
- **QwQ-32B — fits, but it's the dense-32B tax.** 17.3 tg/s on one B60 (dense → SYCL). The bench
  fits in 18.5 GiB, but QwQ emits *long* reasoning traces and 18.5 GiB leaves little room for a big
  KV cache on the B60 — for real long-reasoning use it wants the **B70's** headroom. Usable, not its
  ceiling. Correct on the LCM problem.

## Updated B60 leaderboard (best backend, single card, generation)

| Model | size GiB | tg t/s | W | t/J | license | best for |
|---|---|---|---|---|---|---|
| Qwen3-4B | 2.5 | 101.8 (S) | 81 | — | Apache | tiny & fast |
| **Qwen3-Coder-30B-A3B** | 16.2 | **65.2 (V)** | 60 | 1.09 | Apache | 🏆 coding (fastest big) |
| MiMo-7B-RL | 4.4 | 64.3 (S) | 86 | 0.75 | MIT | reasoning/watt |
| Nemotron-3-Nano-30B-A3B | 17.0 | 55.5 (V) | 65 | 0.85 | NVIDIA-OM | best general big-brain |
| Nemotron-Nano-12B-v2 | 7.0 | 42.8 (S) | 88 | 0.49 | NVIDIA-OM | fast hybrid |
| Phi-4-reasoning-plus | 8.4 | 39.2 (S) | 92 | 0.43 | MIT | reasoning, MIT |
| Qwen3.6-35B-A3B | 20.0 | 37.3 (V) | 60 | — | Apache | general MoE |
| Gemma 4 12B | 6.9 | 36.4 (S) | 85 | 0.42 | Gemma | dense daily-driver |
| Granite-4.0-H-Small | 17.2 | 34.3 (V) | 74 | 0.46 | Apache | 9B-active hybrid |
| QwQ-32B | 18.5 | 17.3 (S) | 92 | 0.19 | Apache | 32B dense reasoner (wants B70) |

**Bottom line:** for *general* work the B60 champ is still Nemotron-3-Nano-30B-A3B; for *coding*,
**Qwen3-Coder-30B-A3B is both the most useful and the fastest** big model the card can hold; and the
purest **MIT** picks are **Phi-4-reasoning-plus** (reason) and **MiMo-7B-RL** (reason/watt). The
lasting takeaway is the backend rule: **dense → SYCL, 30B MoE → Vulkan.**
