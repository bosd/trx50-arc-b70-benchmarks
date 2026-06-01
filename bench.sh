#!/usr/bin/env bash
# Arc Pro B70 LLM benchmark harness — Vulkan vs SYCL, 1 vs 2 GPU, wall-power efficiency.
# Runs llama-bench (pp512/tg128), samples Shelly wall power, emits a markdown table + raw JSON.
# Usage: ./bench.sh [results-file.md]
#
# Device handling (validated on this box):
#   Vulkan 1-GPU : GGML_VK_VISIBLE_DEVICES=1  + -sm none   (B70 #1 only; 750 Ti is Vulkan0, hidden)
#   Vulkan 2-GPU : GGML_VK_VISIBLE_DEVICES=1,2 + -sm layer  (both B70s; -dev does NOT engage the split)
#   SYCL   1-GPU : -dev SYCL0 -sm none
#   SYCL   2-GPU : -sm layer                                (both B70s; SYCL never sees the 750 Ti)
GGUF=/var/lib/models/gguf
VK=$HOME/llama.cpp/build-vulkan/bin/llama-bench
SYCL=$HOME/llama.cpp/build-sycl/bin/llama-bench
ONEAPI=/opt/intel/oneapi/setvars.sh
POWER=/usr/local/bin/server-power           # prints instantaneous wall watts (Shelly)
OUT=${1:-results.md}
RAW=raw; mkdir -p "$RAW"

# name|gguf|type|quant|backend|gpus
MATRIX=(
 "Qwen3.6-35B-A3B|Qwen3.6-35B-A3B-UD-Q4_K_M.gguf|MoE 3B-act|UD-Q4_K_M|vulkan|1"
 "Qwen3.6-35B-A3B|Qwen3.6-35B-A3B-UD-Q4_K_M.gguf|MoE 3B-act|UD-Q4_K_M|vulkan|2"
 "Qwen3.6-35B-A3B|Qwen3.6-35B-A3B-UD-Q4_K_M.gguf|MoE 3B-act|UD-Q4_K_M|sycl|1"
 "Qwen3.6-35B-A3B|Qwen3.6-35B-A3B-UD-Q4_K_M.gguf|MoE 3B-act|UD-Q4_K_M|sycl|2"
 "DeepSeek-R1-Distill-70B|DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf|dense 70B|Q4_K_M|vulkan|2"
 "DeepSeek-R1-Distill-70B|DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf|dense 70B|Q4_K_M|sycl|2"
 "Llama-3.3-70B-Instruct|Llama-3.3-70B-Instruct-Q4_K_M.gguf|dense 70B|Q4_K_M|vulkan|2"
 "Llama-3.3-70B-Instruct|Llama-3.3-70B-Instruct-Q4_K_M.gguf|dense 70B|Q4_K_M|sycl|2"
)

echo "| Model | Backend | Type | Quant | Size (GiB) | GPUs | pp512 t/s | tg128 t/s | avg W | t/J |" | tee "$OUT"
echo "|---|---|---|---|---|---|---|---|---|---|" | tee -a "$OUT"

for row in "${MATRIX[@]}"; do
  IFS='|' read -r name file type quant backend gpus <<< "$row"
  tag="${name}_${backend}_${gpus}g"
  if [ "$backend" = vulkan ]; then
    bin=$VK; pre=""
    if [ "$gpus" = 1 ]; then envsel="GGML_VK_VISIBLE_DEVICES=1"; extra="-sm none";
    else envsel="GGML_VK_VISIBLE_DEVICES=1,2"; extra="-sm layer"; fi
  else
    bin=$SYCL; pre="source $ONEAPI >/dev/null 2>&1;"; envsel=""
    if [ "$gpus" = 1 ]; then extra="-dev SYCL0 -sm none"; else extra="-sm layer"; fi
  fi
  echo ">>> running $tag" >&2
  pf=$(mktemp); ( while :; do "$POWER" >>"$pf" 2>/dev/null; sleep 1; done ) & sp=$!
  bash -c "$pre $envsel $bin -m $GGUF/$file -p 512 -n 128 -ngl 99 $extra -o json" >"$RAW/$tag.json" 2>"$RAW/$tag.err"
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  avgw=$(awk '{s+=$1;n++} END{if(n)printf "%.0f",s/n; else printf "NA"}' "$pf"); rm -f "$pf"
  read pp tg size < <(python3 - "$RAW/$tag.json" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception: print("NA NA NA"); sys.exit()
pp=tg=0.0; size=0
for x in d:
    size=x.get("model_size",0)
    if x.get("n_prompt",0)>0 and x.get("n_gen",0)==0: pp=x.get("avg_ts",0)
    if x.get("n_gen",0)>0 and x.get("n_prompt",0)==0: tg=x.get("avg_ts",0)
print(f"{pp:.1f} {tg:.1f} {size/1073741824:.1f}")
PY
)
  tj=$(awk -v t="$tg" -v w="$avgw" 'BEGIN{if((w+0)>0 && t!="NA") printf "%.3f", t/w; else printf "NA"}')
  printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
    "$name" "$backend" "$type" "$quant" "$size" "$gpus" "$pp" "$tg" "$avgw" "$tj" | tee -a "$OUT"
done
echo "DONE" >&2
