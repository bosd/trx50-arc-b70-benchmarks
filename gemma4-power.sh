#!/usr/bin/env bash
set +u
GGUF=/var/lib/models/gguf/gemma4-12b/gemma-4-12B-it-Q4_K_M.gguf
SYCL=$HOME/llama.cpp/build-sycl/bin/llama-bench
VK=$HOME/llama.cpp/build-vulkan/bin/llama-bench
ONEAPI=/opt/intel/oneapi/setvars.sh
ENERGY=/sys/class/drm/card0/device/hwmon/hwmon2/energy2_input

measure() { # backend
  local backend=$1 bin pre envsel extra
  if [ "$backend" = sycl ]; then bin=$SYCL; pre="source $ONEAPI >/dev/null 2>&1;"; envsel=""; extra="-dev SYCL0 -sm none"
  else bin=$VK; pre=""; envsel="GGML_VK_VISIBLE_DEVICES=0"; extra="-sm none"; fi
  # per-second watts sampler -> /tmp/w.$backend
  ( prev=$(cat "$ENERGY"); while :; do sleep 1; cur=$(cat "$ENERGY"); awk -v a="$prev" -v b="$cur" 'BEGIN{printf "%.1f\n",(b-a)/1e6}'; prev=$cur; done ) >/tmp/w.$backend & samp=$!
  # pure generation, long window, 1 rep
  bash -c "$pre $envsel $bin -m $GGUF -p 0 -n 400 -ngl 99 $extra -r 2 -o json" >/tmp/gen.$backend.json 2>/dev/null
  kill $samp 2>/dev/null
  # active watts = mean of samples > 8W (excludes load/idle)
  active=$(awk '$1>8{s+=$1;n++} END{if(n)printf "%.0f",s/n; else printf "NA"}' /tmp/w.$backend)
  peak=$(sort -n /tmp/w.$backend | tail -1)
  tg=$(python3 -c "import json;d=json.load(open('/tmp/gen.$backend.json'));print(round([x['avg_ts'] for x in d if x['n_gen']>0][0],1))")
  tj=$(awk -v t="$tg" -v w="$active" 'BEGIN{if((w+0)>0)printf "%.3f",t/w;else print "NA"}')
  printf "| %s | %s | %s | %s | %s |\n" "$backend" "$tg" "$active" "$peak" "$tj"
}
echo "| Backend | tg(n400) t/s | active W | peak W | t/J |"
echo "|---|---|---|---|---|"
measure sycl
measure vulkan
