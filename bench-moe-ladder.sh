#!/usr/bin/env bash
# MoE active-param ladder on a single Arc B70 — does generation speed track ACTIVE
# params, not total? Sparse MoE's promise is "70B knowledge at 13B speed". This
# measures tg/s vs active params and normalizes to tg/s-per-active-billion, with a
# dense model as the all-params-active reference.
GGUF=/var/lib/models/gguf
SYCL=$HOME/llama.cpp/build-sycl/bin/llama-bench
ONEAPI=/opt/intel/oneapi/setvars.sh
POWER=/usr/local/bin/server-power
OUT=${1:-moe-ladder-results.md}
RAW=raw-moe; mkdir -p "$RAW"
EN=(/sys/class/hwmon/hwmon2/energy2_input /sys/class/hwmon/hwmon4/energy2_input)
gpu_uj(){ s=0; for f in "${EN[@]}"; do v=$(cat "$f" 2>/dev/null); s=$((s+${v:-0})); done; echo $s; }

# name|file glob|total B|active B|is_moe(1/0)
MODELS=(
  "Qwen3-4B (dense ref)|Qwen_Qwen3-4B-Q4_K_M.gguf|4|4|0"
  "Qwen3.6-35B-A3B|Qwen_Qwen3.6-35B-A3B-Q4_0.gguf|35|3|1"
  "Mixtral-8x7B|mixtral-8x7b/*Q4_K_M*.gguf|47|13|1"
)
echo "| Model | total B | active B | size GiB | tg128 t/s | GPU W | t/J | tg per active-B | tg per total-B |" | tee "$OUT"
echo "|---|---|---|---|---|---|---|---|---|" | tee -a "$OUT"
for m in "${MODELS[@]}"; do
  IFS='|' read -r name glob tot act moe <<< "$m"
  f=$(ls $GGUF/$glob 2>/dev/null | head -1); [ -n "$f" ] || { echo "(skip $name — missing)" >&2; continue; }
  echo ">>> $name" >&2
  # MoE on SYCL needs GGML_SYCL_DISABLE_OPT=1 and -fa 0 (the quirk from hardware-log).
  if [ "$moe" = 1 ]; then envx="GGML_SYCL_DISABLE_OPT=1"; fa="-fa 0"; else envx=""; fa=""; fi
  pf=$(mktemp); ( while :; do "$POWER" >>"$pf" 2>/dev/null; sleep 1; done ) & sp=$!
  e0=$(gpu_uj); t0=$(date +%s.%N)
  bash -c "source $ONEAPI >/dev/null 2>&1; $envx $SYCL -m '$f' -p 64 -n 128 -ngl 99 -dev SYCL0 -sm none $fa -o json" >"$RAW/${name//[^A-Za-z0-9]/_}.json" 2>"$RAW/${name//[^A-Za-z0-9]/_}.err"
  t1=$(date +%s.%N); e1=$(gpu_uj); kill $sp 2>/dev/null; wait $sp 2>/dev/null
  gpuw=$(awk -v a=$e0 -v b=$e1 -v t0=$t0 -v t1=$t1 'BEGIN{dt=t1-t0;if(dt>0)printf "%.0f",(b-a)/1e6/dt}'); rm -f "$pf"
  read tg size < <(python3 -c "import json
d=json.load(open('$RAW/${name//[^A-Za-z0-9]/_}.json'))
tg=[x['avg_ts'] for x in d if x['n_gen']>0]
sz=d[0]['model_size']/1073741824 if d else 0
print(f\"{tg[0]:.1f}\" if tg else 'NA', f'{sz:.1f}')" 2>/dev/null || echo "NA NA")
  tj=$(awk -v t="$tg" -v w="$gpuw" 'BEGIN{if((w+0)>0&&t!="NA")printf "%.3f",t/w; else printf "NA"}')
  tpa=$(awk -v t="$tg" -v a="$act" 'BEGIN{if(a+0>0&&t!="NA")printf "%.1f",t/a; else printf "NA"}')
  tpt=$(awk -v t="$tg" -v x="$tot" 'BEGIN{if(x+0>0&&t!="NA")printf "%.1f",t/x; else printf "NA"}')
  printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" "$name" "$tot" "$act" "$size" "$tg" "$gpuw" "$tj" "$tpa" "$tpt" | tee -a "$OUT"
done
echo "DONE — 'tg per active-B' should be similar across MoE+dense (speed tracks active params);" >&2
echo "       'tg per total-B' shows MoE's win: high speed despite large total size." >&2
