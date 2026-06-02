#!/usr/bin/env bash
# Speculative decoding on Arc B70 — when does a tiny draft accelerate a target?
# A small draft proposes K tokens; the target verifies them in one batch; accepted
# tokens are "free". The thesis to test: spec-decode HELPS slow (big dense) targets
# and HURTS fast ones (the draft overhead isn't repaid at low acceptance).
GGUF=/var/lib/models/gguf
SYCL=$HOME/llama.cpp/build-sycl/bin/llama-bench
SPEC=$HOME/llama.cpp/build-sycl/bin/llama-speculative
ONEAPI=/opt/intel/oneapi/setvars.sh
OUT=${1:-spec-decode-results.md}
RAW=raw-spec; mkdir -p "$RAW"
PROMPT="Write a Python class implementing an LRU cache with get and put in O(1), then explain the design choices."

# name|target|target-bench-flags|draft|draft-flags
PAIRS=(
  "Qwen3-0.6B->Qwen3-4B (fast)|$GGUF/Qwen_Qwen3-4B-Q4_K_M.gguf|-dev SYCL0 -sm none|$GGUF/qwen3-0.6b/Qwen_Qwen3-0.6B-Q4_K_M.gguf|-ngld 99 -devd SYCL1"
  "Llama-3.2-1B->Llama-3.3-70B (slow)|$GGUF/Llama-3.3-70B-Instruct-Q4_K_M.gguf|-sm layer|$GGUF/llama3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf|-ngld 0"
)
echo "| draft -> target | baseline tg t/s | spec tg t/s | speedup | accept % |" | tee "$OUT"
echo "|---|---|---|---|---|" | tee -a "$OUT"
for p in "${PAIRS[@]}"; do
  IFS='|' read -r name target tflags draft dflags <<< "$p"
  { [ -f "$target" ] && [ -f "$draft" ]; } || { echo "(skip $name — model missing)" >&2; continue; }
  echo ">>> $name" >&2
  # baseline: target alone (same GPU placement as spec target)
  base=$(bash -c "source $ONEAPI >/dev/null 2>&1; GGML_VK_VISIBLE_DEVICES=1,2 $SYCL -m '$target' -p 64 -n 128 -ngl 99 $tflags -o json" 2>/dev/null \
    | python3 -c "import json,sys;d=json.load(sys.stdin);v=[x['avg_ts'] for x in d if x['n_gen']>0];print(f'{v[0]:.1f}' if v else 'NA')" 2>/dev/null)
  # spec-decode
  dev=$(echo "$tflags" | grep -oE 'SYCL0|layer' | head -1); place="-dev SYCL0"; echo "$tflags" | grep -q layer && place="-sm layer"
  bash -c "source $ONEAPI >/dev/null 2>&1; $SPEC -m '$target' -md '$draft' -p \"$PROMPT\" -n 256 \
     --spec-draft-n-max 16 --spec-draft-n-min 1 -ngl 99 $place $dflags" > "$RAW/${name//[^A-Za-z0-9]/_}.log" 2>&1
  log="$RAW/${name//[^A-Za-z0-9]/_}.log"
  spec=$(grep -oE "decoded[ ]+[0-9]+ tokens in[ ]+[0-9.]+ seconds, speed:[ ]+[0-9.]+ t/s" "$log" | grep -oE "[0-9.]+ t/s" | grep -oE "[0-9.]+" | tail -1)
  acc=$(grep -oE "accept[ ]+=[ ]+[0-9.]+%" "$log" | grep -oE "[0-9.]+" | tail -1)
  sp=$(awk -v s="$spec" -v b="$base" 'BEGIN{if(b+0>0&&s+0>0)printf "%.2fx",s/b; else printf "-"}')
  printf "| %s | %s | %s | %s | %s |\n" "$name" "${base:-NA}" "${spec:-NA}" "$sp" "${acc:-?}" | tee -a "$OUT"
done
echo "DONE — speedup>1x means the draft pays off. Expect the slow 70B target to win, the fast 4B to lose." >&2
