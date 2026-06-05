#!/usr/bin/env bash
# Run the small-model suite on the x99 GPU (RX 5700 now, or B60 after the swap). Vulkan via the
# prebuilt llama-b9536. RUN WITH THE DESKTOP STOPPED on the 8GB RX 5700 (sudo systemctl isolate
# multi-user.target) to free its ~6GB; on a 24GB B60 the desktop fits alongside. Fetches via curl.
set +u
M=$HOME/llama-vulkan/models; mkdir -p "$M"
total=$(($(cat /sys/class/drm/card0/device/mem_info_vram_total)/1048576))
used=$(($(cat /sys/class/drm/card0/device/mem_info_vram_used)/1048576))
free=$((total-used)); echo "card0 free VRAM: ${free} MiB of ${total}" >&2
# name | hf_repo | file
SUITE=(
 "qwen3-0.6b|unsloth/Qwen3-0.6B-GGUF|Qwen3-0.6B-Q4_K_M.gguf"
 "qwen3-4b|unsloth/Qwen3-4B-GGUF|Qwen3-4B-Q4_K_M.gguf"
 "mimo-7b-rl|quantflex/MiMo-7B-RL-nomtp-GGUF|MiMo-7B-RL-nomtp-Q4_K_M.gguf"
 "lfm2.5-8b-a1b|unsloth/LFM2.5-8B-A1B-GGUF|LFM2.5-8B-A1B-UD-Q4_K_M.gguf"
 "mistral-nemo-12b|bartowski/Mistral-Nemo-Instruct-2407-GGUF|Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"
 "gemma4-12b|ggml-org/gemma-4-12B-it-GGUF|gemma-4-12B-it-Q4_K_M.gguf"
 # "mac-1|<REPO>|<FILE>"   # pending a valid GGUF; CJzafir/Mac-1 is 2.37GB safetensors, qwen3_5 arch (unsupported)
)
echo "| Model | pp512 t/s | tg128 t/s | size GiB | active W | t/J |"
echo "|---|---|---|---|---|---|"
for row in "${SUITE[@]}"; do
  IFS='|' read -r name repo file <<< "$row"
  [ -f "$M/$file" ] || curl -sL -o "$M/$file" "https://huggingface.co/$repo/resolve/main/$file"
  szg=$(( $(stat -c%s "$M/$file" 2>/dev/null || echo 0)/1073741824 ))
  if [ "$((szg*1024))" -gt "$free" ]; then echo "| $name | SKIP — ~${szg}GiB > ${free}MiB free | | | | |"; continue; fi
  bash $HOME/llama-vulkan/bench-rx5700.sh "$M/$file" "$name"
done
