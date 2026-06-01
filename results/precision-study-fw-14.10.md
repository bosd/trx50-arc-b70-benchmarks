| Model | Backend | Type | Quant | Size (GiB) | GPUs | pp512 t/s | tg128 t/s | wall W | GPU W | t/J(GPU) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | vulkan | MoE 3B-act | Q5_K_M | 24.1 | 1 | 1155.7 | 42.3 | 303 | 96 | 0.441 |
| Qwen3.6-35B-A3B | sycl | MoE 3B-act | Q5_K_M | 24.1 | 1 | 928.6 | 68.3 | 312 | 92 | 0.742 |
| Qwen3-4B | vulkan | dense 4B | BF16 | 7.5 | 1 | 1072.1 | 58.8 | 348 | 118 | 0.498 |
| Qwen3-4B | sycl | dense 4B | BF16 | 7.5 | 1 | 1432.6 | 52.5 | 373 | 128 | 0.410 |
| Qwen3-4B | vulkan | dense 4B | Q4_K_M | 2.3 | 1 | 2389.7 | 79.9 | 381 | 138 | 0.579 |
| Qwen3-4B | sycl | dense 4B | Q4_K_M | 2.3 | 1 | 3279.9 | 115.7 | 383 | 144 | 0.803 |
| Llama-3.3-70B | vulkan | dense 70B | IQ1_M | 16.0 | 1 | 219.1 | 6.6 | 366 | 135 | 0.049 |
| Llama-3.3-70B | sycl | dense 70B | IQ1_M | 16.0 | 1 | 295.7 | 11.0 | 414 | 157 | 0.070 |
