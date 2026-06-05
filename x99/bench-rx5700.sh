#!/usr/bin/env bash
# Benchmark a GGUF on the x99 AMD RX 5700 (8GB, RADV/Vulkan). usage: bench-rx5700.sh <gguf> <label>
# Power via amdgpu power1_average (active-gen sampled). Run with the desktop stopped to free VRAM.
set +u
GGUF=$1; LABEL=$2
BIN=$HOME/llama-vulkan/llama-b9536/llama-bench
export LD_LIBRARY_PATH=$HOME/llama-vulkan/llama-b9536
# find amdgpu hwmon with power1_average
PWR=""
for h in /sys/class/drm/card0/device/hwmon/hwmon*; do [ -e "$h/power1_average" ] && PWR="$h/power1_average"; done
RAW=$HOME/llama-vulkan/raw; mkdir -p "$RAW"
# speed (pp512/tg128, 3 reps)
"$BIN" -m "$GGUF" -p 512 -n 128 -ngl 99 -dev Vulkan0 -sm none -r 3 -o json >"$RAW/$LABEL.json" 2>"$RAW/$LABEL.err"
# active-power: pure gen + sampler
( while :; do [ -n "$PWR" ] && awk -v p=$(cat "$PWR") 'BEGIN{printf "%.1f\n",p/1e6}'; sleep 1; done ) >"$RAW/$LABEL.w" & samp=$!
"$BIN" -m "$GGUF" -p 0 -n 400 -ngl 99 -dev Vulkan0 -sm none -r 2 -o json >/dev/null 2>&1
kill $samp 2>/dev/null
w=$(awk '$1>40{s+=$1;n++} END{if(n)printf "%.0f",s/n; else print "NA"}' "$RAW/$LABEL.w")
read pp tg size < <(python3 - "$RAW/$LABEL.json" <<'PY'
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
tj=$(awk -v t="$tg" -v w="$w" 'BEGIN{if((w+0)>0)printf "%.3f",t/w; else print "NA"}')
printf "| %s | %s | %s | %s | %s | %s |\n" "$LABEL" "$pp" "$tg" "$size" "$w" "$tj"
