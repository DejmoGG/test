#!/usr/bin/env bash
set -euo pipefail

# One-time lazy download of weights on cold start.
# Re-run is a no-op thanks to the checks.

WEIGHTS_DIR=${WEIGHTS_DIR:-/workspace/weights}
mkdir -p "$WEIGHTS_DIR"

check_download () {
  local path="$1"
  if [ ! -e "$path/.ok" ]; then
    echo "Downloading $2 -> $path"
    huggingface-cli download "$2" --local-dir "$path" ${HF_TOKEN:+--token $HF_TOKEN}
    touch "$path/.ok"
  else
    echo "Already present: $2"
  fi
}

# Base model + audio encoder + InfiniteTalk weights
check_download "$WEIGHTS_DIR/Wan2.1-I2V-14B-480P" "Wan-AI/Wan2.1-I2V-14B-480P"
check_download "$WEIGHTS_DIR/chinese-wav2vec2-base" "TencentGameMate/chinese-wav2vec2-base"
# safetensors revision
if [ ! -f "$WEIGHTS_DIR/chinese-wav2vec2-base/model.safetensors" ]; then
  huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 \
    --local-dir "$WEIGHTS_DIR/chinese-wav2vec2-base" ${HF_TOKEN:+--token $HF_TOKEN}
fi
check_download "$WEIGHTS_DIR/InfiniteTalk" "MeiGen-AI/InfiniteTalk"

echo "Starting RunPod handler…"
exec python -u /workspace/handler.py
