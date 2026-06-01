#!/usr/bin/env bash
# Extra quant runs appended to the sweep: Q4_0 (1 GPU) + Q8_0 (2 GPU, it's 35 GB > one card), both backends.
# Same metrics as bench-quant.sh (wall + GPU-only power).
GGUF=/var/lib/models/gguf
VK=$HOME/llama.cpp/build-vulkan/bin/llama-bench
SYCL=$HOME/llama.cpp/build-sycl/bin/llama-bench
ONEAPI=/opt/intel/oneapi/setvars.sh
POWER=/usr/local/bin/server-power
OUT=${1:-quant-results.md}
RAW=raw-quant; mkdir -p "$RAW"
EN=(/sys/class/hwmon/hwmon2/energy2_input /sys/class/hwmon/hwmon4/energy2_input)
gpu_uj(){ local s=0 v; for f in "${EN[@]}"; do v=$(cat "$f" 2>/dev/null); s=$((s+${v:-0})); done; echo "$s"; }

# file|quant|gpus|backend
RUNS=(
 "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf|Q4_0|1|vulkan"
 "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf|Q4_0|1|sycl"
 "Qwen3.6-35B-A3B-Q8_0.gguf|Q8_0|2|vulkan"
 "Qwen3.6-35B-A3B-Q8_0.gguf|Q8_0|2|sycl"
)
for row in "${RUNS[@]}"; do
  IFS='|' read -r file quant gpus backend <<< "$row"
  [ -f "$GGUF/$file" ] || { echo "(skip $quant $backend — missing $file)" >&2; continue; }
  tag="${quant}_${backend}_${gpus}g"
  if [ "$backend" = vulkan ]; then bin=$VK; pre=""
    if [ "$gpus" = 1 ]; then envsel="GGML_VK_VISIBLE_DEVICES=1"; extra="-sm none"; else envsel="GGML_VK_VISIBLE_DEVICES=1,2"; extra="-sm layer"; fi
  else bin=$SYCL; pre="source $ONEAPI >/dev/null 2>&1;"; envsel=""
    if [ "$gpus" = 1 ]; then extra="-dev SYCL0 -sm none"; else extra="-sm layer"; fi
  fi
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
  tjw=$(awk -v t="$tg" -v w="$wallw" 'BEGIN{if((w+0)>0&&t!="NA")printf "%.3f",t/w; else printf "NA"}')
  tjg=$(awk -v t="$tg" -v w="$gpuw" 'BEGIN{if((w+0)>0&&t!="NA")printf "%.3f",t/w; else printf "NA"}')
  printf "| Qwen3.6-35B-A3B | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" "$backend" "$quant" "$size" "$gpus" "$pp" "$tg" "$wallw" "$gpuw" "$tjw" "$tjg" | tee -a "$OUT"
done
echo "EXTRA_DONE" >&2
