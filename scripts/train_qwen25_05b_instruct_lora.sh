#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTORCH_ENABLE_MPS_FALLBACK=1 .venv312/bin/praact train \
  outputs/Qwen--Qwen2.5-0.5B-Instruct \
  --train-json "data/starting kit text2picto/train.json" \
  --valid-json "data/starting kit text2picto/valid.json" \
  --output-dir outputs/train-qwen25-05b-instruct-lora \
  --prompt-file prompts/telegraphic_instruction.txt \
  --chat-template \
  --dtype fp32 \
  --max-length 256 \
  --epochs 1 \
  --learning-rate 2e-4 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --logging-steps 10 \
  --eval-steps 200 \
  --save-steps 200 \
  --save-total-limit 2 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --gradient-checkpointing
