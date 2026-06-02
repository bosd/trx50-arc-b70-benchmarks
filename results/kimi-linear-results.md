# Kimi-Linear-48B-A3B on 2× Arc B70 — linear attention vs the context tax

**Model:** `moonshotai/Kimi-Linear-48B-A3B-Instruct`, arch `kimi-linear` (in mainline llama.cpp
since ~b7966). 48B total, **~3B active**, hybrid **Kimi Delta Attention (KDA)** + MLA (3:1).
Quant **IQ4_XS, 24.6 GiB**. Runs on the existing build (5aba536) — no fork needed.

## Throughput (MoE-ladder point) — dual B70, SYCL

| Model | total B | active B | size GiB | pp512 | tg128 | GPU W | tg/active-B | tg/total-B |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B (dense) | 4 | 4 | 2.3 | 3286 | 116.2 | 127 | 29.1 | 29.1 |
| Qwen3.6-35B-A3B (MoE) | 35 | 3 | 19.4 | 929 | 54.0 | 91 | 18.0 | 1.5 |
| **Kimi-Linear-48B-A3B (KDA MoE)** | 48 | 3 | 24.6 | 744 | 46.9 | 113 | 15.6 | 1.0 |

Slots in next to Qwen3.6-35B-A3B (both 3B-active): a touch slower (bigger total + KDA/MLA
overhead), but delivers 47 tg/s from a 48B model. KDA **runs on the GPU** (113 W, no CPU fallback).

## The headline: context-tax — KDA vs transformer

Same depth sweep as the transformer baseline (tg64 at increasing prefill depth):

| context depth | **Kimi-Linear (KDA)** | Qwen3-4B (transformer) |
|---|---|---|
| 0 | 46.9 t/s — **100%** | 116.4 t/s — **100%** |
| 2048 | 45.8 — **98%** | 98.0 — 84% |
| 8192 | 42.9 — **91%** | 68.8 — 59% |
| 32768 | 34.2 — **73%** | 31.1 — **27%** |

**At 32k context the transformer keeps 27% of its empty-context speed; KDA keeps 73%** —
~2.7× better retention. The transformer's KV cache grows with context (attention cost
balloons); KDA's recurrent state is **constant-size**, so generation barely slows. This is the
linear-attention long-context promise, demonstrated on real Arc B70 hardware.

## Gotchas
- **Coherence: check via the *server*, not `llama-cli`.** `llama-cli -no-cnv` emitted a `> >`
  artifact that *looked* like garbage; the server (`/completion`, `/v1/chat/completions`) showed
  the model is coherent and correct ("17+26?" → "43"). (Contrast: Jamba is genuinely broken —
  word-salad even via the server.)
- **Don't pass `GGML_SYCL_DISABLE_OPT=1` or `-fa 0`** for kimi-linear on SYCL — it crashed
  `llama-bench` with `sycl::free ... UR_RESULT_ERROR_INVALID_VALUE`. Plain `-ngl 99 -sm layer` works.
- IQ4_XS (24.6 GB) needs **dual-card** (`-sm layer`); single-card SYCL OpenCL ceiling is too low.
