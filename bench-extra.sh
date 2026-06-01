#!/usr/bin/env bash
# Precision study: Q5_K_M (fills the 35B ladder), Qwen3-4B BF16 vs Q4_K_M (precision ceiling),
# and Llama-3.3-70B at IQ1_M (a 70B at ~1.7-bit on a SINGLE B70). All single-card. Wall + GPU power.
GGUF=/var/lib/models/gguf
VK=$HOME/llama.cpp/build-vulkan/bin/llama-bench
SYCL=$HOME/llama.cpp/build-sycl/bin/llama-bench
ONEAPI=/opt/intel/oneapi/setvars.sh
POWER=/usr/local/bin/server-power
OUT=${1:-extra-results.md}
RAW=raw-extra; mkdir -p "$RAW"
EN=(/sys/class/hwmon/hwmon2/energy2_input /sys/class/hwmon/hwmon4/energy2_input)
gpu_uj(){ local s=0 v; for f in "${EN[@]}"; do v=$(cat "$f" 2>/dev/null); s=$((s+${v:-0})); done; echo "$s"; }

# name|file|type|quant|backend
RUNS=(
 "Qwen3.6-35B-A3B|Qwen_Qwen3.6-35B-A3B-Q5_K_M.gguf|MoE 3B-act|Q5_K_M|vulkan"
 "Qwen3.6-35B-A3B|Qwen_Qwen3.6-35B-A3B-Q5_K_M.gguf|MoE 3B-act|Q5_K_M|sycl"
 "Qwen3-4B|Qwen_Qwen3-4B-bf16.gguf|dense 4B|BF16|vulkan"
 "Qwen3-4B|Qwen_Qwen3-4B-bf16.gguf|dense 4B|BF16|sycl"
 "Qwen3-4B|Qwen_Qwen3-4B-Q4_K_M.gguf|dense 4B|Q4_K_M|vulkan"
 "Qwen3-4B|Qwen_Qwen3-4B-Q4_K_M.gguf|dense 4B|Q4_K_M|sycl"
 "Llama-3.3-70B|Llama-3.3-70B-Instruct-UD-IQ1_M.gguf|dense 70B|IQ1_M|vulkan"
 "Llama-3.3-70B|Llama-3.3-70B-Instruct-UD-IQ1_M.gguf|dense 70B|IQ1_M|sycl"
)
echo "| Model | Backend | Type | Quant | Size (GiB) | GPUs | pp512 t/s | tg128 t/s | wall W | GPU W | t/J(GPU) |" | tee "$OUT"
echo "|---|---|---|---|---|---|---|---|---|---|---|" | tee -a "$OUT"
for row in "${RUNS[@]}"; do
  IFS='|' read -r name file type quant backend <<< "$row"
  [ -f "$GGUF/$file" ] || { echo "(skip $quant $backend — missing)" >&2; continue; }
  tag="${name}_${quant}_${backend}"
  if [ "$backend" = vulkan ]; then bin=$VK; pre=""; envsel="GGML_VK_VISIBLE_DEVICES=1"; extra="-sm none"
  else bin=$SYCL; pre="source $ONEAPI >/dev/null 2>&1;"; envsel=""; extra="-dev SYCL0 -sm none"; fi
  echo ">>> $tag" >&2
  pf=$(mktemp); ( while :; do "$POWER" >>"$pf" 2>/dev/null; sleep 1; done ) & sp=$!
  e0=$(gpu_uj); t0=$(date +%s.%N)
  bash -c "$pre $envsel $bin -m $GGUF/$file -p 512 -n 128 -ngl 99 $extra -o json" >"$RAW/$tag.json" 2>"$RAW/$tag.err"
  t1=$(date +%s.%N); e1=$(gpu_uj)
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  wallw=$(awk '{s+=$1;n++} END{if(n)printf "%.0f",s/n; else printf "NA"}' "$pf"); rm -f "$pf"
  gpuw=$(awk -v a="$e0" -v b="$e1" -v t0="$t0" -v t1="$t1" 'BEGIN{dt=t1-t0; if(dt>0)printf "%.0f",(b-a)/1e6/dt; else printf "NA"}')
  read pp tg size < <(python3 - "$RAW/$tag.json" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception: print("NA NA NA"); sys.exit()
pp=tg=0.0;size=0
for x in d:
    size=x.get("model_size",0)
    if x.get("n_prompt",0)>0 and x.get("n_gen",0)==0: pp=x.get("avg_ts",0)
    if x.get("n_gen",0)>0 and x.get("n_prompt",0)==0: tg=x.get("avg_ts",0)
print(f"{pp:.1f} {tg:.1f} {size/1073741824:.1f}")
PY
)
  tjg=$(awk -v t="$tg" -v w="$gpuw" 'BEGIN{if((w+0)>0&&t!="NA")printf "%.3f",t/w; else printf "NA"}')
  printf "| %s | %s | %s | %s | %s | 1 | %s | %s | %s | %s | %s |\n" "$name" "$backend" "$type" "$quant" "$size" "$pp" "$tg" "$wallw" "$gpuw" "$tjg" | tee -a "$OUT"
done
echo "DONE" >&2
