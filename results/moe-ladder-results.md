| Model | total B | active B | size GiB | tg128 t/s | GPU W | t/J | tg per active-B | tg per total-B |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B (dense ref) | 4 | 4 | 2.3 | 116.2 | 127 | 0.915 | 29.1 | 29.1 |
| Qwen3.6-35B-A3B | 35 | 3 | 19.4 | 54.0 | 91 | 0.593 | 18.0 | 1.5 |
| Mixtral-8x7B | 47 | 13 | — | blocked | — | — | — | — |

**Mixtral (13B-active point) blocked — old GGUF format.** Both TheBloke and MaziyarPanahi
Mixtral-8x7B-Instruct GGUFs fail to load on current llama.cpp with
`missing tensor 'blk.0.ffn_down_exps.weight'`: they store experts as the deprecated
*per-expert* tensor layout, while modern llama.cpp only reads the packed `_exps` format.
Most public Mixtral-8x7B GGUFs predate the change, so a fresh re-quant from source would
be needed. Gotcha worth remembering for any pre-2025 MoE GGUF.

**Reading the ladder:** generation speed tracks **active** params with a real MoE tax —
Qwen3.6-35B-A3B (3B active) runs at 54 tg/s, *slower* per-active-billion (18.0) than the
4B dense (29.1) because of MoE overhead (router + all-experts resident + expert gather).
But against an *equivalent-total* dense model (~13 tg/s for a 35B), the MoE is **~4× faster**
(tg-per-total-B 1.5 is the sparse win). So: MoE buys ~4× the speed of its total size, at a
~40% tax versus an ideal same-active dense model.
