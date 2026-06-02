| Model | Arch | Backend | pp512 t/s | tg128 t/s | wall W | GPU W | GPU-engaged? | t/J(GPU) |
|---|---|---|---|---|---|---|---|---|
| Falcon-Mamba-7B | Mamba-1 SSM | vulkan | 245.3 | 6.7 | 293 | 83 | YES | 0.081 |
| Falcon-Mamba-7B | Mamba-1 SSM | sycl | 275.0 | 22.1 | 393 | 104 | YES | 0.213 |
| RWKV6-World-7B | RWKV RNN | vulkan | 978.3 | 56.5 | 378 | 151 | YES | 0.374 |
| RWKV6-World-7B | RWKV RNN | sycl | 718.2 | 59.4 | 388 | 131 | YES | 0.453 |
| Jamba-Reasoning-3B | hybrid Mamba-MoE | vulkan | 1065.4 | 23.0 | 309 | 66 | partial | 0.348 |
| Jamba-Reasoning-3B | hybrid Mamba-MoE | sycl | 948.3 | 58.4 | 379 | 109 | YES | 0.536 |
| Qwen3-4B | transformer (baseline) | vulkan | 2397.6 | 79.9 | 388 | 136 | YES | 0.588 |
| Qwen3-4B | transformer (baseline) | sycl | 3276.2 | 115.8 | 388 | 162 | YES | 0.715 |
