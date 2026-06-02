| Model | total B | active B | size GiB | tg128 t/s | GPU W | t/J | tg per active-B | tg per total-B |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B (dense ref) | 4 | 4 | 2.3 | 116.2 | 127 | 0.915 | 29.1 | 29.1 |
| Qwen3.6-35B-A3B | 35 | 3 | 19.4 | 54.0 | 91 | 0.593 | 18.0 | 1.5 |
| Kimi-Linear-48B-A3B | 48 | 3 | 24.6 | 46.9 | 113 | 0.415 | 15.6 | 1.0 |
| GLM-4.5-Air | 110 | 12 | 46.9 | 7.6* | 94 | 0.081 | 0.6* | 0.07* |
| Mixtral-8x7B | 47 | 13 | — | blocked | — | — | — | — |

**Mixtral (13B-active point) blocked — old GGUF format.** Both TheBloke and MaziyarPanahi
Mixtral-8x7B-Instruct GGUFs fail to load on current llama.cpp with
`missing tensor 'blk.0.ffn_down_exps.weight'`: they store experts as the deprecated
*per-expert* tensor layout, while modern llama.cpp only reads the packed `_exps` format.
Most public Mixtral-8x7B GGUFs predate the change, so a fresh re-quant from source would
be needed. Gotcha worth remembering for any pre-2025 MoE GGUF.

***GLM-4.5-Air 7.6 tg/s is NOT a fair 12B-active datapoint — it's config-limited, not the
model's true speed.** A 12B-active MoE should land ~25–35 tg/s (between the 3B-active rungs
and a 13B dense); 7.6 is *slower than the 70B dense*. Verified stable across runs and with
`-fa 1` (no change); GPU sat at ~82–94 W (under-utilized). Cause is the stack of
IQ3_XXS i-quant dequant overhead + glm4moe MoE overhead + the 47 GB weights crowding the
~57 GB dual-card Vulkan budget. **The 12B-active rung stays effectively unmeasured on 2 cards** —
it needs the 3rd B70 (47 GB fits with headroom, and a faster K-quant becomes affordable). This
slowness is itself the strongest single argument for the 3rd card.

**Reading the ladder (3B-active rungs):** among models that fit cleanly, generation speed tracks
**active** params with a real MoE tax — Qwen3.6-35B-A3B and Kimi-Linear-48B-A3B (both 3B active)
run 54 and 47 tg/s, *slower* per-active-billion (18.0, 15.6) than the 4B dense (29.1) due to MoE
overhead (router + all-experts resident + expert gather; Kimi adds KDA/MLA cost). But against an
*equivalent-total* dense model (~13 tg/s for a 35B), they're **~4× faster** (tg-per-total-B 1.5/1.0
is the sparse win). So: MoE buys ~4× the speed of its total size, at a ~40% tax versus an ideal
same-active dense model.
