# LFM2.5-8B-A1B on the B60 — the new speed & efficiency king

Liquid AI's **LFM2.5-8B-A1B** (released 2026-06, `lfm2moe` arch — hybrid short-conv + attention
MoE, **8B total / 1B active**). Q4_K_M = 5.3 GiB, single B60, same methodology (xe active-power).
License: "other" (Liquid open license — permissive-ish, *not* OSI/MIT).

| Backend | pp512 t/s | tg t/s | active W | t/J |
|---|---|---|---|---|
| **SYCL** | **3317** | **150.9** | 51 | **2.96** |
| Vulkan | 2867 | 80.8 | 65 | 1.24 |

- **150.9 tg/s — the fastest generation of any model on the B60**, past even Qwen3-4B (101.8).
  That's the 1B-active win: 8B of knowledge, 1B-active compute → it flies.
- **t/J = 2.96** — ~2.4× the next-best efficiency on the card (Qwen3-4B was 1.26), at **51 W**.
- **pp512 = 3317** — by far the highest prefill measured on the B60.
- Coherence ✓ — explained MoE sparse activation correctly (gating invokes only a few experts/token).

## It refines the backend rule

LFM2.5 is a **MoE**, yet **SYCL wins** (150.9 vs 80.8) — the *opposite* of the 30B-class MoEs
(Qwen3-Coder, Nemotron-30B, Granite, Qwen3.6) which all prefer Vulkan. So the rule isn't
"MoE → Vulkan"; it's **large (30B-class) MoE → Vulkan, everything smaller (dense *or* MoE) → SYCL**.
The flip is a property of big sparse models, not of MoE-ness itself. LFM2.5 (8B total) sits firmly
on the small-model SYCL side.

## Where it lands on the MoE ladder

At **1B active** it's below every prior ladder point (all 3B-active or higher), and it behaves
exactly as "speed tracks active params" predicts — the lowest active-param model is the fastest
generator measured. It's the new sparse-end anchor of the ladder.

**Verdict:** for a *small, always-up, ultra-efficient* assistant on the B60, LFM2.5-8B-A1B is now
the top pick on raw speed and tokens/joule — 150 tg/s at 51 W. The only caveat is the non-OSI
license; for a strict MIT requirement, Phi-4 / MiMo remain the picks.
