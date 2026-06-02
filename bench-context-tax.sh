#!/usr/bin/env bash
# Context-tax curve on a single Arc B70 — how generation speed and KV-cache VRAM
# degrade as context grows. Most benchmarks fix context at 512/128; the *shape* of
# the collapse is the useful number for picking a long-context model on 32 GB cards.
# llama-bench -d N prefills N tokens then measures tg, so a depth sweep = the curve.
GGUF=/var/lib/models/gguf
SYCL=$HOME/llama.cpp/build-sycl/bin/llama-bench
ONEAPI=/opt/intel/oneapi/setvars.sh
POWER=/usr/local/bin/server-power
MODEL=${MODEL:-$GGUF/Qwen_Qwen3-4B-Q4_K_M.gguf}
OUT=${1:-context-tax-results.md}
RAW=raw-ctx; mkdir -p "$RAW"
EN=(/sys/class/hwmon/hwmon2/energy2_input /sys/class/hwmon/hwmon4/energy2_input)
gpu_uj(){ s=0; for f in "${EN[@]}"; do v=$(cat "$f" 2>/dev/null); s=$((s+${v:-0})); done; echo $s; }
DEPTHS=(0 2048 8192 32768)

echo "model: $(basename "$MODEL")" | tee "$OUT"
echo "" | tee -a "$OUT"
echo "| context depth | tg64 t/s | pp(depth) t/s | GPU W | tok/J | vs depth-0 |" | tee -a "$OUT"
echo "|---|---|---|---|---|---|" | tee -a "$OUT"
base=""
for d in "${DEPTHS[@]}"; do
  pf=$(mktemp); ( while :; do "$POWER" >>"$pf" 2>/dev/null; sleep 1; done ) & sp=$!
  e0=$(gpu_uj); t0=$(date +%s.%N)
  # -d prefills `d` tokens (the "existing context"), then -n 64 measures tg at that depth;
  # -p 0 with -d also gives prompt-processing throughput for that prefill.
  bash -c "source $ONEAPI >/dev/null 2>&1; $SYCL -m '$MODEL' -p 512 -n 64 -d $d -ngl 99 -dev SYCL0 -sm none -o json" \
    >"$RAW/d$d.json" 2>"$RAW/d$d.err"
  t1=$(date +%s.%N); e1=$(gpu_uj); kill $sp 2>/dev/null; wait $sp 2>/dev/null
  gpuw=$(awk -v a=$e0 -v b=$e1 -v t0=$t0 -v t1=$t1 'BEGIN{dt=t1-t0;if(dt>0)printf "%.0f",(b-a)/1e6/dt}'); rm -f "$pf"
  read tg pp < <(python3 -c "import json
d=json.load(open('$RAW/d$d.json'))
tg=[x['avg_ts'] for x in d if x['n_gen']>0]
pp=[x['avg_ts'] for x in d if x['n_prompt']>0 and x['n_gen']==0]
print(f\"{tg[0]:.1f}\" if tg else 'NA', f\"{pp[0]:.1f}\" if pp else 'NA')" 2>/dev/null || echo "NA NA")
  [ -z "$base" ] && base="$tg"
  ratio=$(awk -v t="$tg" -v b="$base" 'BEGIN{if(b+0>0&&t!="NA")printf "%.0f%%",100*t/b; else printf "-"}')
  tokj=$(awk -v t="$tg" -v w="$gpuw" 'BEGIN{if((w+0)>0&&t!="NA")printf "%.2f",t/w; else printf "NA"}')
  printf "| %s | %s | %s | %s | %s | %s |\n" "$d" "$tg" "$pp" "$gpuw" "$tokj" "$ratio" | tee -a "$OUT"
done
echo "DONE — 'vs depth-0' shows the generation-speed tax as context fills." >&2
