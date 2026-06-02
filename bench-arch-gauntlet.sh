#!/usr/bin/env bash
# Non-transformer architecture gauntlet on a single Arc B70 (Battlemage).
# Maps which exotic ops actually have SYCL/Vulkan kernels vs. fall back to CPU:
#   falcon-mamba-7b   pure Mamba-1 SSM (attention-free)
#   rwkv6-world-7b    RWKV linear-attention RNN
#   jamba-reason-3b   hybrid Mamba-Transformer-MoE
#   Qwen3-4B          dense transformer BASELINE (same harness, for contrast)
# DIAGNOSTIC: idle 2-card GPU draw is ~49 W. With -ngl 99, a GPU-accelerated arch
# pushes GPU power well above idle; a CPU-fallback arch leaves GPU near-idle AND
# runs slow. The "GPU W" column is therefore the headline result, not just t/s.
GGUF=/var/lib/models/gguf
VK=$HOME/llama.cpp/build-vulkan/bin/llama-bench
SYCL=$HOME/llama.cpp/build-sycl/bin/llama-bench
ONEAPI=/opt/intel/oneapi/setvars.sh
POWER=/usr/local/bin/server-power
OUT=${1:-arch-gauntlet-results.md}
RAW=raw-arch; mkdir -p "$RAW"
EN=(/sys/class/hwmon/hwmon2/energy2_input /sys/class/hwmon/hwmon4/energy2_input)
gpu_uj(){ local s=0 v f; for f in "${EN[@]}"; do v=$(cat "$f" 2>/dev/null); s=$((s+${v:-0})); done; echo "$s"; }
find_gguf(){ ls "$GGUF/$1"/*Q4_K_M*.gguf "$GGUF/$1"*Q4_K_M*.gguf 2>/dev/null | head -1; }

# name|dir-or-file glob|arch family
MODELS=(
  "Falcon-Mamba-7B|falcon-mamba-7b|Mamba-1 SSM"
  "RWKV6-World-7B|rwkv6-world-7b|RWKV RNN"
  "Jamba-Reasoning-3B|jamba-reason-3b|hybrid Mamba-MoE"
  "Qwen3-4B|Qwen_Qwen3-4B|transformer (baseline)"
)

echo "| Model | Arch | Backend | pp512 t/s | tg128 t/s | wall W | GPU W | GPU-engaged? | t/J(GPU) |" | tee "$OUT"
echo "|---|---|---|---|---|---|---|---|---|" | tee -a "$OUT"
for m in "${MODELS[@]}"; do
  IFS='|' read -r name dir arch <<< "$m"
  file=$(find_gguf "$dir")
  [ -n "$file" ] || { echo "(skip $name — gguf missing in $GGUF/$dir)" >&2; continue; }
  for backend in vulkan sycl; do
    if [ "$backend" = vulkan ]; then bin=$VK; pre=""; envsel="GGML_VK_VISIBLE_DEVICES=1"; extra="-sm none"
    else bin=$SYCL; pre="source $ONEAPI >/dev/null 2>&1;"; envsel=""; extra="-dev SYCL0 -sm none"; fi
    [ -x "$bin" ] || { echo "(skip $name $backend — no bin)" >&2; continue; }
    tag="${name}_${backend}"; echo ">>> $tag" >&2
    pf=$(mktemp); ( while :; do "$POWER" >>"$pf" 2>/dev/null; sleep 1; done ) & sp=$!
    e0=$(gpu_uj); t0=$(date +%s.%N)
    bash -c "$pre $envsel $bin -m '$file' -p 512 -n 128 -ngl 99 $extra -o json" >"$RAW/$tag.json" 2>"$RAW/$tag.err"
    t1=$(date +%s.%N); e1=$(gpu_uj)
    kill $sp 2>/dev/null; wait $sp 2>/dev/null
    wallw=$(awk '{s+=$1;n++} END{if(n)printf "%.0f",s/n; else printf "NA"}' "$pf"); rm -f "$pf"
    gpuw=$(awk -v a="$e0" -v b="$e1" -v t0="$t0" -v t1="$t1" 'BEGIN{dt=t1-t0; if(dt>0)printf "%.0f",(b-a)/1e6/dt; else printf "NA"}')
    read pp tg < <(python3 - "$RAW/$tag.json" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception: print("NA NA"); sys.exit()
pp=tg=0.0
for x in d:
    if x.get("n_prompt",0)>0 and x.get("n_gen",0)==0: pp=x.get("avg_ts",0)
    if x.get("n_gen",0)>0 and x.get("n_prompt",0)==0: tg=x.get("avg_ts",0)
print(f"{pp:.1f} {tg:.1f}")
PY
)
    eng=$(awk -v w="$gpuw" 'BEGIN{ if(w=="NA"){print "?"} else if(w+0>70){print "YES"} else if(w+0>55){print "partial"} else {print "NO (CPU fallback)"} }')
    tjg=$(awk -v t="$tg" -v w="$gpuw" 'BEGIN{if((w+0)>0&&t!="NA"&&t+0>0)printf "%.3f",t/w; else printf "NA"}')
    printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" "$name" "$arch" "$backend" "$pp" "$tg" "$wallw" "$gpuw" "$eng" "$tjg" | tee -a "$OUT"
  done
done
echo "DONE — 'GPU-engaged?' NO means the arch's kernels are missing on that backend (the finding)." >&2
