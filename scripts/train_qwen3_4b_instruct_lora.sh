#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTORCH_ENABLE_MPS_FALLBACK=1 .venv312/bin/praact train \
  outputs/Qwen--Qwen3-4B-Instruct-2507 \
  --train-json "data/starting kit text2picto/train.json" \
  --valid-json "data/starting kit text2picto/valid.json" \
  --output-dir outputs/train-qwen3-4b-instruct-lora \
  --prompt-file prompts/telegraphic_instruction.txt \
  --chat-template \
  --dtype fp32 \
  --max-length 256 \
  --epochs 1 \
  --learning-rate 1e-4 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --logging-steps 10 \
  --eval-steps 200 \
  --save-steps 200 \
  --save-total-limit 2 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --gradient-checkpointing
