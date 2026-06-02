#!/usr/bin/env bash
# Quality-per-Joule — the metric tokens/s can't game. Runs a real GSM8K eval through
# each model and divides CORRECT ANSWERS by the energy spent getting them (wall +
# GPU, measured with the Shelly plug + xe hwmon). A tiny fast model can win tokens/J
# while being wrong; this rewards useful work per Joule.
#
# Per model: load via llama-server (SYCL), answer N GSM8K questions (0-shot CoT,
# parse the final integer), integrate energy over the run, report:
#   accuracy, wall kJ, GPU kJ, correct-per-kJ(wall), J-per-correct(wall).
GGUF=/var/lib/models/gguf
SRV=$HOME/llama.cpp/build-sycl/bin/llama-server
ONEAPI=/opt/intel/oneapi/setvars.sh
POWER=/usr/local/bin/server-power
# NOTE: no `set -u` — sourcing oneAPI's setvars.sh references unbound vars and would
# abort the shell before the server even starts (silent empty server log -> 0 correct).
N=${N:-40}                      # GSM8K questions
PORT=8099
OUT=${1:-quality-per-joule-results.md}
RAW=raw-qpj; mkdir -p "$RAW"
EN=(/sys/class/hwmon/hwmon2/energy2_input /sys/class/hwmon/hwmon4/energy2_input)
gpu_uj(){ local s=0 v f; for f in "${EN[@]}"; do v=$(cat "$f" 2>/dev/null); s=$((s+${v:-0})); done; echo "$s"; }

# --- fetch GSM8K test set once ---
DATA=$RAW/gsm8k_test.jsonl
if [ ! -s "$DATA" ]; then
  echo "fetching GSM8K test set..." >&2
  export PATH=$HOME/.local/bin:$PATH
  hf download openai/gsm8k --repo-type dataset --include "main/test-*.parquet" --local-dir "$RAW/gsm8k" >/dev/null 2>&1
  python3 - "$RAW/gsm8k" "$DATA" <<'PY'
import sys,glob,json
try:
    import pyarrow.parquet as pq
except Exception:
    sys.exit("pyarrow missing")
f=glob.glob(sys.argv[1]+"/**/test-*.parquet",recursive=True)[0]
t=pq.read_table(f).to_pydict()
with open(sys.argv[2],"w") as o:
    for q,a in zip(t["question"],t["answer"]):
        gold=a.split("####")[-1].strip().replace(",","")
        o.write(json.dumps({"q":q,"gold":gold})+"\n")
PY
fi
[ -s "$DATA" ] || { echo "no GSM8K data — install pyarrow: pip install --user --break-system-packages pyarrow" >&2; exit 1; }

# name|gguf glob (relative to $GGUF)
MODELS=(
  "Qwen3-4B-Q4_K_M|Qwen_Qwen3-4B-Q4_K_M.gguf"
  "Qwen3-4B-BF16|Qwen_Qwen3-4B-bf16.gguf"
  "Falcon-Mamba-7B|falcon-mamba-7b/*Q4_K_M*.gguf"
  "RWKV6-World-7B|rwkv6-world-7b/*Q4_K_M*.gguf"
  "Jamba-Reasoning-3B|jamba-reason-3b/*Q4_K_M*.gguf"
  "Qwen3.6-35B-A3B-Q5|Qwen_Qwen3.6-35B-A3B-Q5_K_M.gguf"
)

echo "| Model | N | correct | acc % | wall kJ | GPU kJ | correct/kJ(wall) | J/correct |" | tee "$OUT"
echo "|---|---|---|---|---|---|---|---|" | tee -a "$OUT"
for m in "${MODELS[@]}"; do
  IFS='|' read -r name glob <<< "$m"
  model=$(ls $GGUF/$glob 2>/dev/null | head -1)
  [ -n "$model" ] || { echo "(skip $name — missing)" >&2; continue; }
  echo ">>> $name" >&2
  # launch server (source oneAPI inline — no set -u, so setvars can't abort us)
  source "$ONEAPI" >/dev/null 2>&1
  "$SRV" -m "$model" -ngl 99 -dev SYCL0 -sm none -c 4096 --host 127.0.0.1 --port $PORT -t 8 >"$RAW/${name}.srv" 2>&1 &
  srv=$!
  up=0; for i in $(seq 1 90); do curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && { up=1; break; }; sleep 2; done
  [ "$up" = 1 ] || { echo "(skip $name — server didn't come up)" >&2; kill $srv 2>/dev/null; wait $srv 2>/dev/null; continue; }
  # energy + wall sampler over the eval
  pf=$(mktemp); ( while :; do "$POWER" >>"$pf" 2>/dev/null; sleep 1; done ) & sp=$!
  e0=$(gpu_uj); t0=$(date +%s.%N)
  correct=0; n=0
  while IFS= read -r line && [ "$n" -lt "$N" ]; do
    q=$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["q"])' "$line")
    gold=$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["gold"])' "$line")
    prompt="Question: ${q}\nSolve step by step, then give the final answer on its own line as: #### <number>"
    resp=$(curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys;print(json.dumps({"prompt":sys.argv[1],"n_predict":512,"temperature":0.0,"stop":["Question:"]}))' "$prompt")" \
      2>/dev/null | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("content",""))
except: print("")')
    pred=$(printf '%s' "$resp" | grep -oE '####\s*-?[0-9,]+' | tail -1 | grep -oE '\-?[0-9,]+' | tr -d ',')
    [ -z "$pred" ] && pred=$(printf '%s' "$resp" | grep -oE '\-?[0-9,]+' | tail -1 | tr -d ',')
    [ "$pred" = "$gold" ] && correct=$((correct+1))
    n=$((n+1))
  done < "$DATA"
  t1=$(date +%s.%N); e1=$(gpu_uj)
  kill $sp 2>/dev/null; wait $sp 2>/dev/null; kill $srv 2>/dev/null; wait $srv 2>/dev/null
  # wall kJ = mean watts * seconds; GPU kJ = energy delta
  wallkj=$(awk -v t0="$t0" -v t1="$t1" '{s+=$1;c++} END{dt=t1-t0; if(c)printf "%.1f",(s/c)*dt/1000; else printf "NA"}' "$pf"); rm -f "$pf"
  gpukj=$(awk -v a="$e0" -v b="$e1" 'BEGIN{printf "%.1f",(b-a)/1e6/1000}')
  acc=$(awk -v c="$correct" -v n="$n" 'BEGIN{if(n)printf "%.1f",100*c/n; else printf "0"}')
  cpkj=$(awk -v c="$correct" -v k="$wallkj" 'BEGIN{if(k+0>0)printf "%.3f",c/k; else printf "NA"}')
  jpc=$(awk -v c="$correct" -v k="$wallkj" 'BEGIN{if(c>0)printf "%.0f",k*1000/c; else printf "NA"}')
  printf "| %s | %s | %s | %s | %s | %s | %s | %s |\n" "$name" "$n" "$correct" "$acc" "$wallkj" "$gpukj" "$cpkj" "$jpc" | tee -a "$OUT"
done
echo "DONE — correct/kJ(wall) is the frontier metric; J/correct is its inverse (lower=better)." >&2
