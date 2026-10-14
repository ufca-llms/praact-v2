#!/usr/bin/env bash
set -Eeuo pipefail

# Train all non-Gemma models from Hugging Face and analyze the complete
# validation base with top-25 candidate predictions.
#
# The models are processed sequentially to keep disk usage predictable. The
# 4 epochs do not create four copies of the model; intermediate checkpoints
# are disabled and only the final model is retained.

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv312/bin/python}"
DATA_ROOT="${DATA_ROOT:-data/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/all-models-4ep}"
RESULTS_DIR="${RESULTS_DIR:-outputs/all-models-4ep-candidates}"
NUM_GPUS="${NUM_GPUS:-2}"

TRAIN_JSON="$DATA_ROOT/starting kit text2picto/train.json"
VALID_JSON="$DATA_ROOT/starting kit text2picto/valid.json"
ARASAAC_JSON="$DATA_ROOT/arasaac_en.json"

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUTPUT_ROOT" "$RESULTS_DIR"

if [[ ! -f "$TRAIN_JSON" || ! -f "$VALID_JSON" || ! -f "$ARASAAC_JSON" ]]; then
  echo "Dados não encontrados. Verifique DATA_ROOT=$DATA_ROOT" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python não encontrado: $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
  TRAIN_LAUNCH=(
    "$PYTHON_BIN" -m torch.distributed.run
    --standalone
    --nproc_per_node "$NUM_GPUS"
  )
else
  TRAIN_LAUNCH=("$PYTHON_BIN")
fi

declare -a MODEL_SPECS=(
  # These are per-GPU values. With NUM_GPUS=2, the effective batches are
  # 128/256 for 3B and 256/512 for 1.5B, without overloading one A100.
  "Llama-3.2-3B-Instruct|meta-llama/Llama-3.2-3B-Instruct|64|128"
  "Qwen2.5-3B-Instruct|Qwen/Qwen2.5-3B-Instruct|64|128"
  "SmolLM3-3B|HuggingFaceTB/SmolLM3-3B|64|128"
  "Qwen2.5-1.5B-Instruct|Qwen/Qwen2.5-1.5B-Instruct|128|256"
  "DeepSeek-R1-Distill-Qwen-1.5B|deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B|128|256"
)

declare -a ANALYSIS_MODELS=()

train_one() {
  local tag="$1"
  local model_id="$2"
  local train_batch="$3"
  local eval_batch="$4"
  local base_dir="$OUTPUT_ROOT/${tag}-pictogram-id-tokenizer"
  local train_dir="$OUTPUT_ROOT/${tag}-pictogram-id-ft"
  local marker="$train_dir/.training-complete"

  echo
  echo "============================================================"
  echo "Modelo: $tag"
  echo "Hugging Face: $model_id"
  echo "Train batch: $train_batch | Eval batch: $eval_batch"
  echo "============================================================"

  if [[ ! -f "$base_dir/pictogram_vocabulary.json" ]]; then
    "$PYTHON_BIN" -m praact_pictogram_prediction.cli prepare-model \
      "$ARASAAC_JSON" \
      "$model_id" \
      --output-dir "$base_dir" \
      --output-vocabulary pictogram-id \
      --token-format raw-id \
      --dtype bf16 \
      --device cuda
  else
    echo "Tokenizer/model preparado já existe; pulando prepare-model."
  fi

  if [[ ! -f "$marker" ]]; then
    "${TRAIN_LAUNCH[@]}" -m praact_pictogram_prediction.cli train \
      "$base_dir" \
      --train-json "$TRAIN_JSON" \
      --valid-json "$VALID_JSON" \
      --output-dir "$train_dir" \
      --dtype bf16 \
      --max-length 128 \
      --epochs 4 \
      --learning-rate 5e-5 \
      --weight-decay 0.01 \
      --warmup-ratio 0.03 \
      --per-device-train-batch-size "$train_batch" \
      --per-device-eval-batch-size "$eval_batch" \
      --gradient-accumulation-steps 1 \
      --logging-steps 200 \
      --eval-steps 1000 \
      --save-steps 1000000 \
      --save-total-limit 1 \
      --device auto

    touch "$marker"
  else
    echo "Treinamento já marcado como concluído; pulando train."
  fi

  ANALYSIS_MODELS+=("$tag=$train_dir")
}

for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r tag model_id train_batch eval_batch <<< "$spec"
  train_one "$tag" "$model_id" "$train_batch" "$eval_batch"
done

analysis_args=(
  --input-json "$VALID_JSON"
  --arasaac-json "$ARASAAC_JSON"
  --output-dir "$RESULTS_DIR"
  --top-k 25
  --batch-size 512
  --dtype bf16
)

for model_spec in "${ANALYSIS_MODELS[@]}"; do
  analysis_args+=(--model "$model_spec")
done

echo
echo "============================================================"
echo "Analisando candidatos top-25 na base completa"
echo "============================================================"

"$PYTHON_BIN" scripts/analyze_candidate_base.py "${analysis_args[@]}"

echo
echo "Concluído. Resultados em: $RESULTS_DIR"
find "$RESULTS_DIR" -maxdepth 1 -type f -printf '%f\n' | sort
