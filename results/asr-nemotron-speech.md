# Speech category — Nemotron 3.5 streaming ASR (RTF + streaming-chunk latency, CPU)

A different benchmark from the rest of this repo: **ASR (speech→text)** is measured in **RTF**
(real-time factor = processing-time ÷ audio-duration; `<1` = faster than real-time) and, for the
streaming model's real use case, **per-chunk latency** at each chunk size.

**CPU only.** torch-xpu (PyTorch on Arc) is blocked on this box (kernel 7.0.10 xe regression — see
README vLLM note), so NeMo can't reach the B60/B70. These are Threadripper 9960X numbers (24 threads),
torch 2.6.0+cpu, NeMo from **`main`** (commit `907edfd` — the stable pip release lacks the model's
`rnnt_bpe_models_prompt` class) in a Python 3.12 uv venv. Audio: 7.43 s LibriSpeech clip, 16 kHz.

## Models

- **`nvidia/nemotron-3.5-asr-streaming-0.6b`** (released 2026-06-04) — multilingual (40 locales),
  Cache-Aware **FastConformer-RNNT** with language-ID prompt conditioning (`EncDecRNNTBPEModelWithPrompt`,
  13 087-token vocab). The target model. Prompt is required: call with
  `override_config=RNNTPromptTranscribeConfig(target_lang="en")` (the bare `target_lang=` kwarg path
  is buggy in this build). It transcribes correctly and emits an auto-detected `<en-US>` language tag.
- **`nvidia/nemotron-speech-streaming-en-0.6b`** — the English sibling it was built from (same arch,
  1024-token vocab, standard `EncDecRNNTBPEModel`). Kept as a lighter-weight reference point.

## Offline RTF (whole-clip `transcribe()`)

| Model | vocab | infer (median/5) | **RTF** | ×realtime |
|---|---|---|---|---|
| nemotron-3.5 (multilingual) | 13 087 | 1.297 s | **0.175** | **5.7×** |
| nemotron-en (English) | 1 024 | 0.616 s | 0.083 | 12.1× |

Both transcribe far faster than real-time on CPU. The multilingual model is ~2× slower than the
English one — the 13× larger vocab (softmax + RNNT joint over 13 k tokens) plus the lang-ID prompt.
Transcript accurate with punctuation + capitalisation either way.

## Streaming-chunk latency — the voice-agent metric (multilingual 3.5, CPU)

Cache-aware streaming simulation (NeMo `speech_to_text_cache_aware_streaming_infer.py`) at the model's
five supported chunk sizes (`att_context_size=[56, R]`, chunk = (R+1)×8×10 ms). "Streaming RTF" =
total streaming compute ÷ audio duration; **`<1` means one CPU thread-pool can sustain a single live
stream in real time.** Per-chunk ≈ compute time to process one chunk step.

| chunk | att_context | streaming time | **streaming RTF** | ~per-chunk compute | sustains 1 live stream on CPU? |
|---|---|---|---|---|---|
| 80 ms | [56, 0] | 14.28 s | **1.92** | ~154 ms | ❌ no (1.9× too slow) |
| 160 ms | [56, 1] | 8.60 s | **1.16** | ~183 ms | ❌ no (just misses) |
| 320 ms | [56, 3] | 4.93 s | **0.66** | ~205 ms | ✅ yes (1.5× headroom) |
| 560 ms | [56, 6] | 3.97 s | **0.53** | ~284 ms | ✅ yes (1.9× headroom) |
| 1120 ms | [56, 13] | 2.87 s | **0.39** | ~410 ms | ✅ yes (2.6× headroom) |

### What this says

- **There's a real CPU latency floor: the 3.5 model needs ≥ 320 ms chunks to keep up with a single
  live stream.** The low-latency **80 ms / 160 ms** voice-agent modes run **slower than real-time on
  CPU** (RTF 1.92 / 1.16) — each chunk's fixed per-step overhead (encoder set-up + 13 k-token RNNT
  joint) dominates when chunks are tiny, so they pile up. Those modes **need a GPU**.
- **The tradeoff is the classic one:** smaller chunk = lower *algorithmic* latency (you get text
  sooner) but *higher* streaming RTF (more per-chunk overhead). 320 ms is the sweet spot on this CPU —
  sub-half-second latency *and* 1.5× compute headroom.
- The English 0.6 B model (2× faster offline) would push the CPU-sustainable floor down toward
  ~160 ms — the multilingual vocab is what costs the low-latency modes.

## Takeaways for the box

- **Speech is CPU-viable here, with a caveat.** Offline/batch transcription and ≥320 ms streaming run
  comfortably real-time on the Threadripper without touching GPU VRAM — good for a voice front-end
  alongside the LLMs. **Sub-320 ms voice-agent latency, though, is GPU-only** on this model.
- **The Arc GPU path needs OpenVINO, not torch.** torch-xpu is blocked, so to reach the B60 for the
  low-latency modes you'd convert to **OpenVINO** (Intel's native Arc stack) or run the **ONNX-int4**
  build (`onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`: encoder/decoder/joint + silero-VAD
  for onnxruntime-genai). GPU only matters for **low-latency** *or* **many concurrent** streams.

*(Harness: `asr-rtf.py` (offline), `asr-stream-sweep.sh` (chunk sweep) + NeMo-main
`speech_to_text_cache_aware_streaming_infer.py`, `EncDecRNNTBPEModelWithPrompt`, CPU.)*
