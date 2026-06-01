#!/usr/bin/env bash
# Quant sweep on Qwen3.6-35B-A3B (single B70): UD-Q4_K_M / UD-Q6_K / Q8_0 × Vulkan + SYCL.
# Captures pp512/tg128 + wall power (Shelly) AND GPU-only power (xe hwmon card energy → PMZFX-comparable).
GGUF=/var/lib/models/gguf
VK=$HOME/llama.cpp/build-vulkan/bin/llama-bench
SYCL=$HOME/llama.cpp/build-sycl/bin/llama-bench
ONEAPI=/opt/intel/oneapi/setvars.sh
POWER=/usr/local/bin/server-power
OUT=${1:-quant-results.md}
RAW=raw-quant; mkdir -p "$RAW"
EN=(/sys/class/hwmon/hwmon2/energy2_input /sys/class/hwmon/hwmon4/energy2_input)   # xe "card" energy, both B70s
gpu_uj(){ local s=0 v; for f in "${EN[@]}"; do v=$(cat "$f" 2>/dev/null); s=$((s+${v:-0})); done; echo "$s"; }

SWEEP=(
 "Qwen3.6-35B-A3B|Qwen3.6-35B-A3B-UD-Q4_K_M.gguf|UD-Q4_K_M"
 "Qwen3.6-35B-A3B|Qwen3.6-35B-A3B-UD-Q6_K.gguf|UD-Q6_K"
 "Qwen3.6-35B-A3B|Qwen3.6-35B-A3B-Q8_0.gguf|Q8_0"
)
echo "| Model | Backend | Quant | Size (GiB) | GPUs | pp512 t/s | tg128 t/s | wall W | GPU W | t/J(wall) | t/J(GPU) |" | tee "$OUT"
echo "|---|---|---|---|---|---|---|---|---|---|---|" | tee -a "$OUT"
for backend in vulkan sycl; do
 for row in "${SWEEP[@]}"; do
  IFS='|' read -r name file quant <<< "$row"
  [ -f "$GGUF/$file" ] || { echo "(skip $quant $backend — gguf missing)" >&2; continue; }
  tag="${quant}_${backend}"
  if [ "$backend" = vulkan ]; then bin=$VK; pre=""; envsel="GGML_VK_VISIBLE_DEVICES=1"; extra="-sm none";
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
pp=tg=0.0; size=0
for x in d:
    size=x.get("model_size",0)
    if x.get("n_prompt",0)>0 and x.get("n_gen",0)==0: pp=x.get("avg_ts",0)
    if x.get("n_gen",0)>0 and x.get("n_prompt",0)==0: tg=x.get("avg_ts",0)
print(f"{pp:.1f} {tg:.1f} {size/1073741824:.1f}")
PY
)
  tjw=$(awk -v t="$tg" -v w="$wallw" 'BEGIN{if((w+0)>0&&t!="NA")printf "%.3f",t/w; else printf "NA"}')
  tjg=$(awk -v t="$tg" -v w="$gpuw" 'BEGIN{if((w+0)>0&&t!="NA")printf "%.3f",t/w; else printf "NA"}')
  printf "| %s | %s | %s | %s | 1 | %s | %s | %s | %s | %s | %s |\n" "$name" "$backend" "$quant" "$size" "$pp" "$tg" "$wallw" "$gpuw" "$tjw" "$tjg" | tee -a "$OUT"
 done
done
echo "DONE" >&2
