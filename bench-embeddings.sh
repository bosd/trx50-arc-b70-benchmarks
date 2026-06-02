#!/usr/bin/env bash
# Embedding throughput on a single Arc B70 — the numbers that decide which embedder
# to run in the Qdrant/OCA-index RAG pipeline. Almost no public Arc embedding data.
# Measures tokens/s encoded (llama-bench -embd) + GPU power -> tokens/Joule, and
# derives docs/s for a 512-token document.
GGUF=/var/lib/models/gguf
VK=$HOME/llama.cpp/build-vulkan/bin/llama-bench
SYCL=$HOME/llama.cpp/build-sycl/bin/llama-bench
ONEAPI=/opt/intel/oneapi/setvars.sh
POWER=/usr/local/bin/server-power
OUT=${1:-embeddings-results.md}
RAW=raw-embed; mkdir -p "$RAW"
EN=(/sys/class/hwmon/hwmon2/energy2_input /sys/class/hwmon/hwmon4/energy2_input)
gpu_uj(){ s=0; for f in "${EN[@]}"; do v=$(cat "$f" 2>/dev/null); s=$((s+${v:-0})); done; echo $s; }
findg(){ ls "$GGUF/$1"/*Q8_0*.gguf "$GGUF/$1"/*.gguf 2>/dev/null | head -1; }

# name|dir|backend
MODELS=(
  "Qwen3-Embedding-0.6B|qwen3-embed-0.6b|sycl"
  "Qwen3-Embedding-4B|qwen3-embed-4b|sycl"
  "Qwen3-Embedding-8B|qwen3-embed-8b|sycl"
)
echo "| Model | Backend | enc tok/s | docs/s (512tok) | GPU W | tok/J | docs/kJ |" | tee "$OUT"
echo "|---|---|---|---|---|---|---|" | tee -a "$OUT"
for m in "${MODELS[@]}"; do
  IFS='|' read -r name dir backend <<< "$m"
  f=$(findg "$dir"); [ -n "$f" ] || { echo "(skip $name — missing)" >&2; continue; }
  if [ "$backend" = vulkan ]; then bin=$VK; pre=""; envsel="GGML_VK_VISIBLE_DEVICES=1"; extra="-sm none"
  else bin=$SYCL; pre="source $ONEAPI >/dev/null 2>&1;"; envsel=""; extra="-dev SYCL0 -sm none"; fi
  echo ">>> $name" >&2
  pf=$(mktemp); ( while :; do "$POWER" >>"$pf" 2>/dev/null; sleep 1; done ) & sp=$!
  e0=$(gpu_uj); t0=$(date +%s.%N)
  # -embd 1 = embedding mode; -p 512 measures encode throughput of 512-token batches; -n 0 no gen.
  bash -c "$pre $envsel $bin -m '$f' -embd 1 -p 512 -n 0 -ngl 99 $extra -o json" >"$RAW/$name.json" 2>"$RAW/$name.err"
  t1=$(date +%s.%N); e1=$(gpu_uj); kill $sp 2>/dev/null; wait $sp 2>/dev/null
  gpuw=$(awk -v a=$e0 -v b=$e1 -v t0=$t0 -v t1=$t1 'BEGIN{dt=t1-t0;if(dt>0)printf "%.0f",(b-a)/1e6/dt}'); rm -f "$pf"
  enc=$(python3 -c "import json;d=json.load(open('$RAW/$name.json'));v=[x['avg_ts'] for x in d if x['n_prompt']>0];print(f'{v[0]:.0f}' if v else 'NA')" 2>/dev/null || echo NA)
  docs=$(awk -v e="$enc" 'BEGIN{if(e!="NA")printf "%.1f",e/512; else printf "NA"}')
  tokj=$(awk -v e="$enc" -v w="$gpuw" 'BEGIN{if((w+0)>0&&e!="NA")printf "%.1f",e/w; else printf "NA"}')
  dkj=$(awk -v d="$docs" -v w="$gpuw" 'BEGIN{if((w+0)>0&&d!="NA")printf "%.1f",d*1000/w; else printf "NA"}')
  printf "| %s | %s | %s | %s | %s | %s | %s |\n" "$name" "$backend" "$enc" "$docs" "$gpuw" "$tokj" "$dkj" | tee -a "$OUT"
done
echo "DONE — tok/J is the RAG-cost metric; docs/kJ = 512-token documents embedded per kilojoule." >&2
