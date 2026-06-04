#!/usr/bin/env bash
# Gemma 4 12B (Q4_K_M) on the Arc Pro B60 (card0 / SYCL0 / Vulkan0).
# Measures pp512 + tg128 on SYCL and Vulkan; per-card avg watts via xe energy delta.
set +u
GGUF=/var/lib/models/gguf/gemma4-12b/gemma-4-12B-it-Q4_K_M.gguf
SYCL=$HOME/llama.cpp/build-sycl/bin/llama-bench
VK=$HOME/llama.cpp/build-vulkan/bin/llama-bench
ONEAPI=/opt/intel/oneapi/setvars.sh
ENERGY=/sys/class/drm/card0/device/hwmon/hwmon2/energy2_input   # B60, microjoules
RAW=/tmp/gemma4-raw; mkdir -p "$RAW"

run() {  # $1=backend
  local backend=$1 bin pre envsel extra tag
  tag="gemma4-12b-q4_${backend}"
  if [ "$backend" = sycl ]; then
    bin=$SYCL; pre="source $ONEAPI >/dev/null 2>&1;"; envsel=""; extra="-dev SYCL0 -sm none"
  else
    bin=$VK; pre=""; envsel="GGML_VK_VISIBLE_DEVICES=0"; extra="-sm none"
  fi
  echo ">>> $tag" >&2
  local e0 t0 e1 t1
  e0=$(cat "$ENERGY"); t0=$(date +%s.%N)
  bash -c "$pre $envsel $bin -m $GGUF -p 512 -n 128 -ngl 99 $extra -r 3 -o json" >"$RAW/$tag.json" 2>"$RAW/$tag.err"
  t1=$(date +%s.%N); e1=$(cat "$ENERGY")
  awk -v e0="$e0" -v e1="$e1" -v t0="$t0" -v t1="$t1" 'BEGIN{printf "%.1f", ((e1-e0)/1e6)/(t1-t0)}' > "$RAW/$tag.w"
}

run sycl
run vulkan

echo "| Backend | pp512 t/s | tg128 t/s | size GiB | avg W (B60) | t/J |"
echo "|---|---|---|---|---|---|"
for backend in sycl vulkan; do
  tag="gemma4-12b-q4_${backend}"
  w=$(cat "$RAW/$tag.w" 2>/dev/null)
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
  tj=$(awk -v t="$tg" -v w="$w" 'BEGIN{if((w+0)>0)printf "%.3f",t/w; else printf "NA"}')
  printf "| %s | %s | %s | %s | %s | %s |\n" "$backend" "$pp" "$tg" "$size" "$w" "$tj"
done
