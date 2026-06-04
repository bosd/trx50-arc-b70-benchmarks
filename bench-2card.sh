#!/usr/bin/env bash
# usage: bench-2card.sh <gguf> <label>   — tensor-split across B60(card0)+B70(card1)
set +u
GGUF=$1; LABEL=$2
SYCL=$HOME/llama.cpp/build-sycl/bin/llama-bench
VK=$HOME/llama.cpp/build-vulkan/bin/llama-bench
ONEAPI=/opt/intel/oneapi/setvars.sh
E0=/sys/class/drm/card0/device/hwmon/hwmon2/energy2_input  # B60
E1=/sys/class/drm/card1/device/hwmon/hwmon3/energy2_input  # B70
RAW=/tmp/${LABEL}-raw; mkdir -p "$RAW"
sumE(){ awk -v a=$(cat $E0) -v b=$(cat $E1) 'BEGIN{print a+b}'; }

run(){ local backend=$1 bin pre envsel extra tag
  tag="${LABEL}_${backend}"
  if [ "$backend" = sycl ]; then bin=$SYCL; pre="source $ONEAPI >/dev/null 2>&1;"; envsel=""; extra="-sm layer"
  else bin=$VK; pre=""; envsel="GGML_VK_VISIBLE_DEVICES=0,1"; extra="-sm layer"; fi
  bash -c "$pre $envsel $bin -m $GGUF -p 512 -n 128 -ngl 99 $extra -r 3 -o json" >"$RAW/$tag.json" 2>"$RAW/$tag.err"
  ( prev=$(sumE); while :; do sleep 1; cur=$(sumE); awk -v a="$prev" -v b="$cur" 'BEGIN{printf "%.1f\n",(b-a)/1e6}'; prev=$cur; done ) >"$RAW/$tag.w" & samp=$!
  bash -c "$pre $envsel $bin -m $GGUF -p 0 -n 400 -ngl 99 $extra -r 2 -o json" >/dev/null 2>&1
  kill $samp 2>/dev/null
}
run sycl; run vulkan
echo "| Backend | pp512 t/s | tg128 t/s | size GiB | active W (both) | t/J |"
echo "|---|---|---|---|---|---|"
for backend in sycl vulkan; do
  tag="${LABEL}_${backend}"
  w=$(awk '$1>20{s+=$1;n++} END{if(n)printf "%.0f",s/n; else print "NA"}' "$RAW/$tag.w")
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
  tj=$(awk -v t="$tg" -v w="$w" 'BEGIN{if((w+0)>0)printf "%.3f",t/w;else print "NA"}')
  printf "| %s | %s | %s | %s | %s | %s |\n" "$backend" "$pp" "$tg" "$size" "$w" "$tj"
done
