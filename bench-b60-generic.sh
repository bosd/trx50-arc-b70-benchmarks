#!/usr/bin/env bash
# usage: bench-b60-generic.sh <gguf-path> <label>
set +u
GGUF=$1; LABEL=$2
SYCL=$HOME/llama.cpp/build-sycl/bin/llama-bench
VK=$HOME/llama.cpp/build-vulkan/bin/llama-bench
ONEAPI=/opt/intel/oneapi/setvars.sh
ENERGY=/sys/class/drm/card0/device/hwmon/hwmon2/energy2_input
RAW=/tmp/${LABEL}-raw; mkdir -p "$RAW"

run() { # backend
  local backend=$1 bin pre envsel extra tag
  tag="${LABEL}_${backend}"
  if [ "$backend" = sycl ]; then bin=$SYCL; pre="source $ONEAPI >/dev/null 2>&1;"; envsel=""; extra="-dev SYCL0 -sm none"
  else bin=$VK; pre=""; envsel="GGML_VK_VISIBLE_DEVICES=0"; extra="-sm none"; fi
  # speed run (pp512 + tg128, 3 reps)
  bash -c "$pre $envsel $bin -m $GGUF -p 512 -n 128 -ngl 99 $extra -r 3 -o json" >"$RAW/$tag.json" 2>"$RAW/$tag.err"
  # active-power run: pure gen, per-second sampler
  ( prev=$(cat "$ENERGY"); while :; do sleep 1; cur=$(cat "$ENERGY"); awk -v a="$prev" -v b="$cur" 'BEGIN{printf "%.1f\n",(b-a)/1e6}'; prev=$cur; done ) >"$RAW/$tag.w" & samp=$!
  bash -c "$pre $envsel $bin -m $GGUF -p 0 -n 400 -ngl 99 $extra -r 2 -o json" >/dev/null 2>&1
  kill $samp 2>/dev/null
}
run sycl; run vulkan

echo "| Backend | pp512 t/s | tg128 t/s | size GiB | active W | t/J |"
echo "|---|---|---|---|---|---|"
for backend in sycl vulkan; do
  tag="${LABEL}_${backend}"
  w=$(awk '$1>8{s+=$1;n++} END{if(n)printf "%.0f",s/n; else print "NA"}' "$RAW/$tag.w")
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
