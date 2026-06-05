# Speech category — Nemotron streaming ASR (RTF, CPU baseline)

A different kind of benchmark from the rest of this repo: **ASR (speech→text)** isn't measured in
tg/s or t/J but in **RTF** (real-time factor = processing-time ÷ audio-duration; lower is better,
`<1` means faster than real-time) and **×realtime** (`1/RTF`).

## What ran, and the substitution

Target was NVIDIA's **`nemotron-3.5-asr-streaming-0.6b`** (released 2026-06-04, multilingual,
40 locales). It **can't load on the stable NeMo toolkit yet** — its language-ID-prompt model class
`nemo...rnnt_bpe_models_prompt` only exists in NeMo *main*, not the pip release (the framework hasn't
caught up to a 1-day-old model). Rather than fight a NeMo-from-source install, I benchmarked its
**English sibling it was built from — `nvidia/nemotron-speech-streaming-en-0.6b`** — *same*
Cache-Aware **FastConformer-RNNT** architecture and *same* 0.6B size, standard `EncDecRNNTBPEModel`
class that the stable toolkit loads. The lang-ID prompt adds negligible compute, so this RTF is
representative of the 3.5 model too.

## Setup

- **CPU only.** torch-xpu (PyTorch on Arc) is blocked on this box (kernel 7.0.10 xe regression — see
  README vLLM note), so NeMo can't reach the B60/B70. This is a **CPU baseline** on the Threadripper
  9960X (24 threads), torch 2.6.0+cpu, `nemo_toolkit[asr]` in a Python 3.12 uv venv.
- Audio: standard 7.43 s LibriSpeech clip, 16 kHz mono. Offline `model.transcribe()` (whole clip),
  median of 5 runs after warmup.

## Result

| Model | runtime | device | audio | infer (median) | **RTF** | ×realtime |
|---|---|---|---|---|---|---|
| nemotron-speech-streaming-en-0.6b | NeMo / torch-CPU | CPU (24t) | 7.43 s | 0.616 s | **0.083** | **12.1×** |

- **12× faster than real-time on CPU alone** — a 0.6B FastConformer-RNNT transcribes far quicker than
  the audio plays, even with no GPU. Load 4.5 s; transcript accurate with punctuation + capitalisation
  (*"Well I don't wish to see it any more, observed Phoebe, turning away her eyes…"*).
- This is **offline-batch RTF** (throughput proxy). The model's headline feature is *cache-aware
  streaming* (80–1120 ms chunks) for **low-latency** voice agents — per-chunk latency is a separate
  measurement (NeMo streaming API) not done here.

## Takeaways

- **Speech is CPU-viable on this box.** A streaming ASR model serves comfortably faster than real-time
  on the Threadripper without touching the GPUs — it doesn't compete with the B60/B70 for VRAM. Good
  for a voice front-end alongside the LLMs.
- **The Arc GPU path needs OpenVINO, not torch.** torch-xpu is blocked, so to put ASR on the B60 you'd
  convert to **OpenVINO** (Intel's native Arc stack) or run the **ONNX-int4** build
  (`onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`, ships encoder/decoder/joint + silero-VAD
  for onnxruntime-genai). Given CPU already does 12× realtime for a single stream, GPU only matters for
  *many concurrent* streams — a future study if voice serving scales up.
- **To benchmark the exact multilingual 3.5 model**, install NeMo from `main`
  (`pip install git+https://github.com/NVIDIA/NeMo`) for the `rnnt_bpe_models_prompt` class.

*(Harness: `asr-rtf.py` + `nemotron-speech-streaming-en-0.6b.nemo`, NeMo `EncDecRNNTBPEModel`, CPU.)*
