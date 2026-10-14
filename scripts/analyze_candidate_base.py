"""Analyze next-pictogram candidates for an entire labeled dataset.

The model is loaded once and prompts are scored in batches. This reproduces
the single-prompt score semantics: probabilities are normalized only over the
allowed pictogram vocabulary.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("data/data/starting kit text2picto/valid.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gemma-candidates-results/full-base"),
    )
    parser.add_argument(
        "--arasaac-json",
        type=Path,
        default=Path("data/data/arasaac_en.json"),
        help="ARASAAC vocabulary used to display pictogram names.",
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="NAME=PATH",
        help="Model to analyze. Repeat this option for multiple models.",
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def clean_markdown(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ").strip()


def load_examples(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}.")

    labeled: list[dict[str, Any]] = []
    for index, example in enumerate(payload, start=1):
        if not isinstance(example, dict):
            continue
        pictos = example.get("pictos")
        if not isinstance(pictos, list) or len(pictos) < 2:
            continue
        labeled.append(
            {
                "source_index": index,
                "text": example.get("text") or example.get("src", ""),
                "pictos": [str(value) for value in pictos],
                "tgt": example.get("tgt", ""),
            }
        )
    return labeled


def load_pictogram_names(
    arasaac_path: Path,
    examples: list[dict[str, Any]],
) -> dict[str, str]:
    """Build ID -> display name, with dataset alignment as a fallback.

    Some IDs present in the dataset are no longer present in the current
    ARASAAC export. For those IDs, the aligned ``pictos``/``tgt`` fields from
    the dataset provide the original name used by the task.
    """
    names: dict[str, str] = {}

    if arasaac_path.exists():
        with arasaac_path.open(encoding="utf-8") as file:
            arasaac_items = json.load(file)
        for item in arasaac_items:
            keywords = [
                entry.get("keyword")
                for entry in item.get("keywords", [])
                if isinstance(entry, dict) and entry.get("keyword")
            ]
            if keywords:
                names[str(item["_id"])] = keywords[0]

    for example in examples:
        target_words = str(example.get("tgt", "")).split()
        pictos = example["pictos"]
        if len(target_words) != len(pictos):
            continue
        for pictogram_id, word in zip(pictos, target_words, strict=True):
            if word and pictogram_id not in names:
                names[pictogram_id] = word

    return names


def display_pictogram(
    pictogram_id: int | str,
    names: dict[str, str],
    fallback_keywords: list[str] | None = None,
) -> str:
    key = str(pictogram_id)
    name = names.get(key)
    if not name and fallback_keywords:
        name = ", ".join(fallback_keywords)
    return f"ID {key} — {name or 'nome não disponível'}"


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this fast batch analysis.")
    return torch.device("cuda")


def score_model(
    model_name: str,
    model_dir: Path,
    examples: list[dict[str, Any]],
    output_dir: Path,
    top_k: int,
    batch_size: int,
    dtype_name: str,
    pictogram_names: dict[str, str],
) -> None:
    device = resolve_device()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16

    print(f"\n[{model_name}] carregando {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    metadata_path = model_dir / "pictogram_vocabulary.json"
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    allowed_token_ids = [
        int(token_id) for token_id in metadata["allowed_token_ids"]
    ]
    allowed_token_ids_tensor = torch.tensor(
        allowed_token_ids,
        dtype=torch.long,
        device=device,
    )
    metadata_entries = metadata["entries"]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0

    results: list[dict[str, Any]] = []
    total_batches = (len(examples) + batch_size - 1) // batch_size

    for batch_number, start in enumerate(
        range(0, len(examples), batch_size),
        start=1,
    ):
        batch_examples = examples[start : start + batch_size]
        encoded_prompts = [
            tokenizer.encode(
                " ".join(example["pictos"][:-1]),
                add_special_tokens=False,
            )
            for example in batch_examples
        ]

        if any(not token_ids for token_ids in encoded_prompts):
            raise ValueError("At least one prompt produced no tokens.")

        max_length = max(len(token_ids) for token_ids in encoded_prompts)
        input_ids = torch.full(
            (len(encoded_prompts), max_length),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(encoded_prompts), max_length),
            dtype=torch.long,
            device=device,
        )

        lengths: list[int] = []
        for row, token_ids in enumerate(encoded_prompts):
            length = len(token_ids)
            lengths.append(length)
            input_ids[row, :length] = torch.tensor(
                token_ids,
                dtype=torch.long,
                device=device,
            )
            attention_mask[row, :length] = 1

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            row_indices = torch.arange(len(batch_examples), device=device)
            last_positions = torch.tensor(lengths, device=device) - 1
            last_logits = outputs.logits[row_indices, last_positions]
            allowed_logits = last_logits.index_select(1, allowed_token_ids_tensor)
            probabilities = torch.softmax(allowed_logits, dim=-1)
            values, local_indices = torch.topk(
                probabilities,
                k=min(top_k, probabilities.shape[-1]),
                dim=-1,
            )

        for row, example in enumerate(batch_examples):
            expected_id = example["pictos"][-1]
            predictions: list[dict[str, Any]] = []
            correct_rank: int | None = None

            for position, (probability, local_index) in enumerate(
                zip(values[row].tolist(), local_indices[row].tolist(), strict=True),
                start=1,
            ):
                entry = metadata_entries[int(local_index)]
                prediction = {
                    "rank": position,
                    "label": entry["label"],
                    "token": entry["token"],
                    "token_id": entry["token_id"],
                    "pictogram_id": entry["pictogram_id"],
                    "keywords": entry["keywords"],
                    "pictogram_name": ", ".join(entry["keywords"]),
                    "probability": float(probability),
                }
                predictions.append(prediction)
                if str(entry["pictogram_id"]) == expected_id:
                    correct_rank = position

            results.append(
                {
                    "example": example["source_index"],
                    "text": example["text"],
                    "prompt_ids": example["pictos"][:-1],
                    "prompt": " ".join(
                        display_pictogram(pictogram_id, pictogram_names)
                        for pictogram_id in example["pictos"][:-1]
                    ),
                    "expected_pictogram_id": int(expected_id),
                    "expected_pictogram_name": pictogram_names.get(
                        expected_id,
                        "nome não disponível",
                    ),
                    "correct_rank": correct_rank,
                    "hit_at_25": correct_rank is not None,
                    "predictions": predictions,
                }
            )

        print(
            f"[{model_name}] batch {batch_number}/{total_batches} "
            f"({min(start + batch_size, len(examples))}/{len(examples)})",
            flush=True,
        )

    ranks = [
        result["correct_rank"]
        for result in results
        if result["correct_rank"] is not None
    ]

    markdown: list[str] = [
        f"# Candidatos do {model_name} — base completa",
        "",
        f"Modelo: `{model_dir}`  ",
        f"Base: `{args.input_json}`  ",
        f"Exemplos avaliados: {len(results)}  ",
        f"Candidatos por exemplo: top-{top_k}  ",
        f"Batch de inferência: {batch_size}  ",
        "",
        "## Resumo",
        "",
        "| Métrica | Resultado |",
        "|---|---:|",
        f"| Exemplos | {len(results)} |",
        f"| Respostas no top-{top_k} | {len(ranks)} |",
        f"| Accuracy@{top_k} | {len(ranks) / len(results):.6f} |",
    ]
    if ranks:
        markdown.extend(
            [
                f"| Rank médio quando encontrado | {mean(ranks):.2f} |",
                f"| Melhor rank | {min(ranks)} |",
                f"| Pior rank | {max(ranks)} |",
            ]
        )

    markdown.extend(["", "---", ""])

    for result in results:
        markdown.extend(
            [
                f"## Exemplo {result['example']}",
                "",
                f"**Frase:** {clean_markdown(result['text'])}  ",
                f"**Prompt:** `{result['prompt']}`  ",
                f"**Resposta correta:** {display_pictogram(result['expected_pictogram_id'], pictogram_names)}  ",
                (
                    f"**Rank da resposta correta:** {result['correct_rank']}"
                    if result["correct_rank"] is not None
                    else f"**Resposta correta não apareceu no top-{top_k}.**"
                ),
                "",
                "| Rank | ID e palavra | Texto | Probabilidade | Correto |",
                "|---:|---:|---|---:|:---:|",
            ]
        )

        for prediction in result["predictions"]:
            prediction_id = prediction["pictogram_id"]
            keywords = ", ".join(prediction.get("keywords", []))
            is_correct = str(prediction_id) == str(result["expected_pictogram_id"])
            markdown.append(
                f"| {prediction['rank']} | "
                f"{display_pictogram(prediction_id, pictogram_names, prediction.get('keywords'))} | "
                f"{clean_markdown(keywords)} | "
                f"{prediction['probability']:.8f} | "
                f"{'✅' if is_correct else ''} |"
            )
        markdown.extend(["", "---", ""])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{model_name}_top25_full_base.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"{model_name}_top25_full_base.json").write_text(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "input_json": str(args.input_json),
                "examples_analyzed": len(results),
                "top_k": top_k,
                "batch_size": batch_size,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[{model_name}] arquivos gravados em {output_dir}")
    del model
    torch.cuda.empty_cache()


args = parse_args()
examples = load_examples(args.input_json)
if not examples:
    raise RuntimeError(f"No labeled examples found in {args.input_json}.")
pictogram_names = load_pictogram_names(args.arasaac_json, examples)

if args.model:
    models: dict[str, Path] = {}
    for model_spec in args.model:
        if "=" not in model_spec:
            raise ValueError("Each --model value must have the form NAME=PATH.")
        model_name, model_path = model_spec.split("=", 1)
        if not model_name or not model_path:
            raise ValueError("Each --model value must have the form NAME=PATH.")
        models[model_name] = Path(model_path)
else:
    models = {
        "Gemma2": Path(os.environ["GEMMA2_TRAIN_DIR"]),
        "Gemma4": Path(os.environ["GEMMA4_TRAIN_DIR"]),
    }

for model_name, model_dir in models.items():
    score_model(
        model_name=model_name,
        model_dir=model_dir,
        examples=examples,
        output_dir=args.output_dir,
        top_k=args.top_k,
        batch_size=args.batch_size,
        dtype_name=args.dtype,
        pictogram_names=pictogram_names,
    )
