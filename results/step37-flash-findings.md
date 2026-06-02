# Step 3.7 Flash (StepFun) on 2× Arc B70 — a memory-edge case study

**Model:** `stepfun-ai/Step-3.7-Flash` — `step35` arch, **196B params, ~11B active**, 288 experts
(8 used/token), first 3 layers dense, 256k context. Quant: **IQ3_XXS, 72 GB on disk (3.06 bpw)**.
Needs StepFun's `llama.cpp` fork (`step3.7` branch, `step35` not in mainline).

## Verdict: it runs — but it's RAM-bound, not compute-bound

**Step 3.7 Flash does load and generate on the Brain**, but only with **all 288-expert MoE
weights on the CPU** (the ~3 GB of attention/dense on the GPUs). In that config:

| Config | pp t/s | tg t/s | GPU W | Notes |
|---|---|---|---|---|
| all experts on CPU (`-ot exps=CPU`) | **3.1** | **2.3** | ~53 | only reliably-fitting config; GPUs ~idle |
| balanced (experts split GPU/CPU) | OOM | OOM | — | dual-card imbalance overfills one B70 |

It's slow because the entire sparse-MoE compute runs on the Threadripper while the two B70s sit
nearly idle holding only attention.

## Why it won't keep experts on the GPU (the real finding)

72 GB barely fits across **57 GB usable VRAM + ~28 GB RAM (~83 GB)**, and three things conspire:

1. **SYCL is unusable for this.** The Arc cards expose no Level-Zero backend in this runtime, so
   SYCL falls back to **OpenCL**, whose allocation ceiling caps usable VRAM at ~40 GB dual — the
   model OOMs before the RAM is even touched.
2. **Vulkan has the VRAM (58.7 GB) but the dual-card split imbalances.** `-sm layer` piles the
   output/embedding/KV/compute overhead onto the main card on top of its layer share, overfilling
   it (`630 MB buffer allocate failed` while the other card is half-empty). `-ts` rebalancing helps
   the *weights* load, but the larger **pp512 compute buffer** then tips it back over.
3. **RAM is the hard wall.** To drop GPU load enough to fit, you must offload >30 GB of experts to
   CPU — but there's only ~28 GB RAM (one DIMM; the other was DOA). The VRAM-fit and RAM-fit
   windows don't overlap by ~6 GB.

## Actionable: the RAM RMA is the unlock

Restoring the second 32 GB DIMM (→ 64 GB, ~60 GB usable) makes the offload math comfortable:
~30 GB of experts in RAM, ~42 GB on the GPUs with real headroom for compute buffers — at which
point GPU-resident experts become feasible and tg/s should jump well above the 2.3 CPU-only floor.
Until then, **Step 3.7 Flash on this box is a "it runs, slowly, on CPU" curiosity**, not a serving option.

## Repro
- Fork: `github.com/stepfun-ai/llama.cpp` @ `step3.7` (`8f34864`); build Vulkan **and** SYCL with `icx/icpx`.
- Geometry read via `gguf.GGUFReader`: `block_count=45`, `expert_count=288`, `expert_used_count=8`,
  `leading_dense_block_count=3`.
- Working load: `GGML_VK_VISIBLE_DEVICES=1,2 llama-cli -ngl 99 -sm layer -ts 40,60 -ot 'exps=CPU'`.
- `-ot` layer-range regexes must be passed via a script file — they get mangled through ssh/bash quoting.
