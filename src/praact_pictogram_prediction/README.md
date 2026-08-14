# Praact Pictogram Prediction

Tools for training causal language models directly over ARASAAC pictogram ID
sequences.

This package is separate from the text-to-picto CLI. It replaces the base
model tokenizer with a pictogram-level tokenizer, then trains the model to
predict the next pictogram ID.

## 1. Prepare A Pictogram-ID Model

Start from a Hugging Face causal LM or a local model directory, such as
`outputs/gpt2`.

```bash
PYTHONPATH=src .venv312/bin/python -m praact_pictogram_prediction.cli prepare-model \
  data/arasaac_en.json \
  outputs/gpt2 \
  --output-dir outputs/gpt2-pictogram-id-tokenizer \
  --output-vocabulary pictogram-id \
  --token-format raw-id
```

This creates a new tokenizer whose vocabulary is:

```text
[PAD], [UNK], [BOS], [EOS], 2247, 2248, 2249, ...
```

The model embeddings are resized to this new vocabulary. Each pictogram ID
embedding is initialized as the mean of the base model embeddings for the
pictogram keywords/captions from `data/arasaac_en.json`.

The output directory contains:

- `tokenizer.json` and tokenizer config files
- model weights/config
- `pictogram_vocabulary.json`

## 2. Train

Train on the starting kit JSON, using the `pictos` field as the target causal
sequence.

```bash
PYTHONPATH=src .venv312/bin/python -m praact_pictogram_prediction.cli train \
  outputs/gpt2-pictogram-id-tokenizer \
  --train-json "data/starting kit text2picto/train.json" \
  --valid-json "data/starting kit text2picto/valid.json" \
  --output-dir outputs/gpt2-pictogram-id-ft \
  --epochs 1 \
  --per-device-train-batch-size 8 \
  --gradient-accumulation-steps 1 \
  --learning-rate 5e-5 \
  --device cpu
```

The training objective is standard causal LM next-token prediction over
pictogram IDs. For example, a sequence like:

```text
6632 5441 6456 2527
```

teaches the model to predict each next pictogram in the sequence.

By default, examples containing pictogram IDs that are not in the prepared
tokenizer are skipped. To fail instead, add:

```bash
--fail-on-unknown-pictos
```

For a quick smoke test:

```bash
PYTHONPATH=src .venv312/bin/python -m praact_pictogram_prediction.cli train \
  outputs/gpt2-pictogram-id-tokenizer \
  --train-json "data/starting kit text2picto/train.json" \
  --output-dir /tmp/praact-pictogram-train-smoke \
  --max-train-samples 16 \
  --epochs 1 \
  --per-device-train-batch-size 4 \
  --logging-steps 1 \
  --save-steps 100 \
  --device cpu
```

## 3. Score The Next Pictogram

Use `score` to rank likely next pictograms for a prompt sequence.

```bash
PYTHONPATH=src .venv312/bin/python -m praact_pictogram_prediction.cli score \
  outputs/gpt2-pictogram-id-ft \
  --prompt "6632 5441 6456" \
  --top-k 10 \
  --device cpu
```

Example prompt:

```text
6632 = I / me
5441 = want
6456 = eat
```

The command returns JSON with labels, token IDs, ARASAAC pictogram IDs,
keywords, and probabilities.

## 4. Evaluate

Evaluate on `valid.json`:

```bash
PYTHONPATH=src .venv312/bin/python -m praact_pictogram_prediction.cli evaluate \
  outputs/gpt2-pictogram-id-ft \
  --eval-json "data/starting kit text2picto/valid.json" \
  --batch-size 16 \
  --top-k 1,3,5,10 \
  --device cpu
```

Metrics:

- `loss`: mean causal LM negative log likelihood per predicted pictogram
- `perplexity`: `exp(loss)`
- `accuracy@k`: fraction of prediction positions where the true next pictogram
  is in the top `k`
- `mrr`: mean reciprocal rank of the true next pictogram
- `mean_rank` and `median_rank`: average and median rank of the true next
  pictogram

Example result from `outputs/gpt2-pictogram-id-ft`:

```json
{
  "examples": 4364,
  "skipped_examples": 33,
  "prediction_targets": 27693,
  "loss": 4.121391804682422,
  "perplexity": 61.64498035755783,
  "accuracy@1": 0.31036724081897954,
  "accuracy@3": 0.44769436319647565,
  "accuracy@5": 0.5063373415664608,
  "accuracy@10": 0.5829270934893295,
  "mrr": 0.4029901402788586,
  "mean_rank": 220.03639909002274,
  "median_rank": 5.0
}
```

## CLI Summary

```bash
PYTHONPATH=src .venv312/bin/python -m praact_pictogram_prediction.cli --help
```

Available commands:

- `prepare-model`: replace the tokenizer with a pictogram-level tokenizer
- `train`: train causal next-pictogram prediction
- `score`: rank the next pictogram for a prompt
- `evaluate`: compute validation metrics

