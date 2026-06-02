| Model | N | correct | acc % | wall kJ | GPU kJ | correct/kJ(wall) | J/correct |
|---|---|---|---|---|---|---|---|
| Qwen3-4B-Q4_K_M | 20 | 8 | 40.0 | 40.4 | 15.9 | 0.198 | 5050 |
| Qwen3-4B-BF16 | 20 | 11 | 55.0 | 81.9 | 30.1 | 0.134 | 7445 |
| Falcon-Mamba-7B | 20 | 6 | 30.0 | 99.7 | 30.9 | 0.060 | 16617 |
| RWKV6-World-7B | 20 | 1 | 5.0 | 57.0 | 21.9 | 0.018 | 57000 |
| Jamba-Reasoning-3B | 20 | 0* | 0.0* | 76.0 | 23.8 | 0.000 | NA |
| Qwen3.6-35B-A3B-Q5 | 20 | 15 | 75.0 | 36.7 | 13.2 | 0.409 | 2447 |

***Jamba-Reasoning-3B 0% is a llama.cpp support bug, not the model's capability.**
Follow-up investigation: the model emits incoherent output (`> > >` in raw completion,
word-salad like "Women in Hong is literally, Fat Port Contact..." via the chat template's
`reasoning_content`). Reproduced across **F16 and Q4_K_M**, **CPU (-ngl 0) and SYCL/GPU**,
and **raw `/completion` and templated `/v1/chat/completions --jinja`** — so it is not a
quantization, backend, or prompt-format issue. The hybrid Mamba-Transformer-MoE generates
tokens fast (Study A's throughput is real) but they're garbage on this build (HEAD
`5aba536`, 2026-06-01). Treat Jamba's row as N/A pending a llama.cpp Jamba fix or a
known-good GGUF; the original "parsing artifact" hypothesis was wrong.
