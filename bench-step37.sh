#!/usr/bin/env bash
# Step 3.7 Flash (StepFun) — 198B sparse-MoE vision-language model, ~11B active/token.
# A "can a 198B MoE VLM even run on 2x Arc B70?" experiment. Hybrid VRAM+RAM, MoE
# experts spilled to CPU. Measures pp512/tg128 + wall & GPU power + tokens/Joule,
# same methodology as bench-extra.sh.
#
# ── PREREQUISITES (run once; NOT done automatically — ~76 GB download + fork build) ──
#   1. Custom fork (mainline llama.cpp has no `step35` arch):
#        git clone https://github.com/stepfun-ai/llama.cpp.git ~/llama.cpp-step37
#        cd ~/llama.cpp-step37 && git checkout -b step3.7 origin/step3.7
#        # SYCL build (XMX):
#        source /opt/intel/oneapi/setvars.sh
#        cmake -B build-sycl -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DCMAKE_BUILD_TYPE=Release \
#              -DGGML_SYCL_HOST_MEM_FALLBACK=ON && cmake --build build-sycl -j
#        # Vulkan build (fallback if step35 lacks SYCL kernels):
#        cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-vulkan -j
#   2. Model (smallest quant that approaches fitting on this box):
#        huggingface-cli download stepfun-ai/Step-3.7-Flash-GGUF \
#          --include "*IQ3_XXS*" mmproj-Step-3.7-flash-f16.gguf --local-dir /var/lib/models/gguf/step37
#
# ── FEASIBILITY ── IQ3_XXS = 76 GB vs ~57 GB usable VRAM + ~26 GB usable RAM (~83 GB).
#   Fits only with aggressive MoE expert offload to CPU and a SMALL context. The RAM RMA
#   (back to 64 GB) turns this from knife-edge into comfortable and unlocks Q3_K_M (94 GB).
#   KEY UNKNOWN this run answers: do step35 ops have SYCL/Vulkan kernels, or fall back to CPU?

set -u
STEP=/var/lib/models/gguf/step37
# Sharded GGUF lands in an IQ3_XXS/ subdir; point at shard 00001 (llama.cpp auto-loads the rest).
MODEL=$(ls "$STEP"/IQ3_XXS/*00001*.gguf "$STEP"/IQ3_XXS/*.gguf "$STEP"/*IQ3_XXS*.gguf 2>/dev/null | head -1)
FORK=$HOME/llama.cpp-step37
SYCL_BIN=$FORK/build-sycl/bin/llama-bench
VK_BIN=$FORK/build-vulkan/bin/llama-bench
ONEAPI=/opt/intel/oneapi/setvars.sh
POWER=/usr/local/bin/server-power
OUT=${1:-step37-results.md}
RAW=raw-step37; mkdir -p "$RAW"
EN=(/sys/class/hwmon/hwmon2/energy2_input /sys/class/hwmon/hwmon4/energy2_input)
gpu_uj(){ local s=0 v f; for f in "${EN[@]}"; do v=$(cat "$f" 2>/dev/null); s=$((s+${v:-0})); done; echo "$s"; }

[ -n "$MODEL" ] || { echo "Model not found in $STEP — run the download step in the header." >&2; exit 1; }
[ -x "$SYCL_BIN" ] || [ -x "$VK_BIN" ] || { echo "Build the StepFun fork first (header)." >&2; exit 1; }

# Each row: backend|n_cpu_moe (experts pushed to RAM)|extra flags.
# Sweep n_cpu_moe until it fits VRAM; SYCL MoE needs GGML_SYCL_DISABLE_OPT=1 and -fa 0
# (the same quirk we hit on Qwen3.6-35B-A3B — see hardware-log).
RUNS=(
  "sycl|28|-fa 0"
  "sycl|34|-fa 0"
  "vulkan|28|"
  "vulkan|34|"
)

echo "| Backend | n_cpu_moe | ctx | pp512 t/s | tg128 t/s | wall W | GPU W | t/J(GPU) | notes |" | tee "$OUT"
echo "|---|---|---|---|---|---|---|---|---|" | tee -a "$OUT"
for row in "${RUNS[@]}"; do
  IFS='|' read -r backend ncpu extra <<< "$row"
  if [ "$backend" = sycl ]; then
    [ -x "$SYCL_BIN" ] || { echo "(skip sycl — not built)" >&2; continue; }
    bin=$SYCL_BIN; pre="source $ONEAPI >/dev/null 2>&1; export GGML_SYCL_DISABLE_OPT=1 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1;"; envsel="ONEAPI_DEVICE_SELECTOR=level_zero:0,1"
  else
    [ -x "$VK_BIN" ] || { echo "(skip vulkan — not built)" >&2; continue; }
    bin=$VK_BIN; pre=""; envsel="GGML_VK_VISIBLE_DEVICES=1,2"
  fi
  tag="step37_${backend}_moe${ncpu}"
  echo ">>> $tag" >&2
  pf=$(mktemp); ( while :; do "$POWER" >>"$pf" 2>/dev/null; sleep 1; done ) & sp=$!
  e0=$(gpu_uj); t0=$(date +%s.%N)
  # -ngl 99 = all non-expert layers on GPU; --n-cpu-moe N = N MoE expert-layers in RAM;
  # -sm layer splits the GPU-resident weights across both B70s. ctx 2048 keeps KV tiny.
  bash -c "$pre $envsel $bin -m '$MODEL' -p 512 -n 128 -ngl 99 --n-cpu-moe $ncpu -sm layer -c 2048 $extra -o json" \
    >"$RAW/$tag.json" 2>"$RAW/$tag.err"
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
  note=$(grep -qiE "error|out of memory|abort|not supported" "$RAW/$tag.err" && echo "ERR(see $tag.err)" || echo "ok")
  tjg=$(awk -v t="$tg" -v w="$gpuw" 'BEGIN{if((w+0)>0&&t!="NA"&&t+0>0)printf "%.3f",t/w; else printf "NA"}')
  printf "| %s | %s | 2048 | %s | %s | %s | %s | %s | %s |\n" "$backend" "$ncpu" "$pp" "$tg" "$wallw" "$gpuw" "$tjg" "$note" | tee -a "$OUT"
done
echo "DONE — raw JSON/err in $RAW/. If all rows ERR with 'not supported', step35 lacks GPU kernels on this backend (expected risk)." >&2
