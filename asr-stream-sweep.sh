#!/usr/bin/env bash
cd /home/bosd/asr-bench/NeMo/examples/asr/asr_cache_aware_streaming
source /home/bosd/asr-bench/.venv/bin/activate
SCRIPT=speech_to_text_cache_aware_streaming_infer.py
declare -A MS=([0]=80 [1]=160 [3]=320 [6]=560 [13]=1120)
echo "R chunk_ms took_s"
for R in 0 1 3 6 13; do
  out=$(timeout 320 python $SCRIPT model_path=/home/bosd/asr-bench/nemotron-3.5-asr-streaming-0.6b.nemo dataset_manifest=/home/bosd/asr-bench/manifest.json cuda=-1 target_lang=en att_context_size=[56,$R] output_path=/tmp/out_$R.json 2>/tmp/sw_$R.err)
  took=$(echo "$out" | grep -oE "took: [0-9.]+s" | grep -oE "[0-9.]+")
  [ -z "$took" ] && took=$(grep -oE "took: [0-9.]+s" /tmp/sw_$R.err | grep -oE "[0-9.]+")
  echo "$R ${MS[$R]} ${took:-FAIL}"
done
echo SWEEPDONE
