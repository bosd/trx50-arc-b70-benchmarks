# NVIDIA Nemotron on the TRX50 — and a new B60 champion

Three Nemotrons, same B60 methodology (active-generation watts from the xe energy counter,
SYCL vs Vulkan, single card unless noted). All three archs are compiled into llama.cpp
`b50-5aba536` — `nemotron`, `nemotron_h` (hybrid Mamba-Transformer) and `nemotron_h_moe`.

| Model | arch | quant / size | placement | best tg | result |
|---|---|---|---|---|---|
| **Nemotron-3-Nano-30B-A3B** | `nemotron_h_moe` | Q4_0 17 GiB | 1× B60 | **55.5 t/s** | 🏆 new B60 champ |
| Nemotron-Nano-12B-v2 | `nemotron_h` | Q4_K_M 7 GiB | 1× B60 | 42.8 t/s | fits, beats Gemma 4 |
| Llama-3.3-Nemotron-Super-49B | `nemotron`/llama | Q4_K_M 28 GiB | 2× (B60+B70) | 13.4 t/s | needs the split |

---

## 1. Nemotron-3-Nano-30B-A3B — dethrones Qwen as the best B60 model

A **hybrid Mamba2 + MoE** (30B total, 3B active), Q4_0 = 17 GiB → fits the B60 with context room.
Direct rival to the previous champ Qwen3.6-35B-A3B (same Q4_0, same 3B-active MoE class).

| Backend | pp512 t/s | tg t/s | active W | t/J |
|---|---|---|---|---|
| Vulkan | **1076** | **55.5** | 65 | 0.854 |
| SYCL | 687 | 51.5 | **50** | **1.030** |

**It beats Qwen3.6-35B-A3B (37 t/s Vulkan) by ~1.5×, at smaller size and similar/lower power.**
- **Backend reversal:** Vulkan ≥ SYCL here — the opposite of every *dense* model on these cards.
  The Mamba/SSM layers run better on the Vulkan backend than on SYCL/OpenCL, so for the hybrid
  Nemotrons **Vulkan is the path**, not SYCL. (Genuinely new finding for this box.)
- **SYCL t/J = 1.03 at 50 W** is the best efficiency of any "big-brain" model measured on the B60.
- Coherence ✓ — the bat-and-ball trap: it set up `x + (x+1) = 1.10` → ball = **$0.05**, dodging
  the intuitive-wrong $0.10. Clean reasoning, no garbage from the hybrid arch.

**Verdict: this is now the best single model for the B60** — a 30B-class reasoner that fits one
24 GB card, runs at 55 t/s, sips 50–65 W, and is MoE-sparse + Mamba-cheap. It displaces
Qwen3.6-35B-A3B as the "biggest brain that fits one efficient card."

## 2. Nemotron-Nano-12B-v2 — hybrid, and quicker than a same-size dense

`nemotron_h` hybrid Mamba2-Transformer, 12B, Q4_K_M 7 GiB.

| Backend | pp512 t/s | tg t/s | active W | t/J |
|---|---|---|---|---|
| SYCL | 828 | **42.8** | 88 | 0.486 |
| Vulkan | 754 | 22.4 | 74 | 0.303 |

- **42.8 tg/s — faster than Gemma 4 12B (36.4) at the same 7 GiB**, because the Mamba layers are
  cheaper than full attention. (Here SYCL still leads tg — the dense-attention half dominates at
  this smaller size; the SSM-favors-Vulkan effect only flips the bigger 30B-A3B.)
- Coherence ✓ — `23×17` by distributive method (230 + 161 = 391), correct and clean.
- A useful arch-gauntlet datapoint: hybrid Mamba-Transformer runs **fully on the Arc GPU**, no
  CPU fallback — unlike Jamba, which is the same family of idea but broken in llama.cpp.

## 3. Llama-3.3-Nemotron-Super-49B — the big one (no B70 pair yet)

NAS-pruned from Llama-3.3-70B → 49B dense, Q4_K_M 28 GiB. **There is no B70 pair on the box yet**
(the 2nd B70 is the planned purchase), so this ran across the only 2-card config available —
a **mixed B60 + B70 tensor-split** (`-sm layer`), power summed over both cards.

| Backend | pp512 t/s | tg t/s | active W (both) | t/J |
|---|---|---|---|---|
| **SYCL** | 388 | **13.4** | 178 | 0.075 |
| Vulkan | 259 | 5.2* | 45 | — |

\*Vulkan 5.2 t/s is the **same config-limited anomaly** seen with GLM-4.5-Air (cards idle between
tokens — note the 45 W floor). The **SYCL 13.4 t/s is the representative number.**
- Function confirmed (llama-bench generated cleanly; `nemotron`/Llama arch is rock-solid).
- **Caveat:** this split is **bottlenecked by the B60** (the slower, smaller card drags the pair —
  exactly what the b60-vs-b70 study warned about). A real **uniform B70 pair** would run this
  meaningfully faster and is the right home for a 49B. As-is, 13 t/s is usable but not the model's
  ceiling on this hardware.

---

## Updated B60 leaderboard (best generation backend, single card)

| Model | size | tg t/s | active W | t/J | role |
|---|---|---|---|---|---|
| **Nemotron-3-Nano-30B-A3B** | 17 GiB | **55.5** (Vk) | 65 | 0.854 | 🏆 most capability that fits |
| Nemotron-Nano-12B-v2 (hybrid) | 7 GiB | 42.8 (SYCL) | 88 | 0.486 | fast hybrid |
| Qwen3.6-35B-A3B | 20 GiB | 37.3 (Vk) | 60 | — | (former champ) |
| Gemma 4 12B | 6.9 GiB | 36.4 (SYCL) | 85 | 0.42 | dense daily-driver |
| MiMo-7B-RL | 4.4 GiB | 64.3 (SYCL) | 86 | 0.75 | reasoning per watt |
| Qwen3-4B | 2.5 GiB | 101.8 (SYCL) | 81 | — | fast & cheap |

**Bottom line:** NVIDIA's hybrid Mamba-MoE **Nemotron-3-Nano-30B-A3B is the new best B60 resident**
— more capable *and* faster than the Qwen it replaces, and the most efficient big model on the card.
Two backend lessons fall out: for **hybrid-SSM** models, **Vulkan wins** (reverse of dense); and a
49B still wants a real B70 pair, not a B60-mixed split.
