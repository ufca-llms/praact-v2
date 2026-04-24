#!/usr/bin/env python3
"""Run expanded Praact models on the next pictogram prediction task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from praact.decoding import build_reverse_keyword_lookup
from praact.decoding import generate_hypotheses_batch
from praact.decoding import load_model_for_decoding
from praact.decoding import load_prompt_template
from praact.decoding import render_prompt
from praact.decoding import save_generation_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more expanded Praact models on the next pictogram task.",
    )
    parser.add_argument(
        "model_paths",
        nargs="+",
        type=Path,
        help="One or more expanded model directories.",
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=Path("data/ImageCLEF2026_NextPictogramPredicti/test_next_picto.json"),
        help="Task dataset JSON. Defaults to the provided next pictogram test file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/next-pictogram"),
        help="Directory where one prediction file per model will be saved.",
    )
    parser.add_argument(
        "--input-field",
        default="tgt",
        help="Field from the task JSON used as the sequence context.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("prompts/next_pictogram_prediction.txt"),
        help="Prompt template file. Use {sentence} as placeholder.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size used during generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1,
        help="Maximum new tokens to generate per sample.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="fp32",
        help="Torch dtype used to load the model.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Execution device used during generation.",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Format prompts with tokenizer.apply_chat_template().",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty applied to repeated generated tokens.",
    )
    return parser.parse_args()


def load_task_dataset(dataset_path: Path, input_field: str) -> list[dict[str, str]]:
    with dataset_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Expected task JSON root to be a list.")

    dataset: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each task item must be an object.")
        if "id" not in item:
            raise ValueError("Each task item must include 'id'.")
        if input_field not in item:
            raise ValueError(f"Each task item must include '{input_field}'.")
        if not isinstance(item[input_field], str):
            raise ValueError(f"Task field '{input_field}' must be a string.")

        dataset.append(
            {
                "id": str(item["id"]),
                "context": item[input_field],
            }
        )

    return dataset


def chunk_list(items: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def build_output_filename(model_path: Path) -> str:
    return f"{model_path.name}_next_pictogram_predictions.json"


def run_model(
    model_path: Path,
    dataset: list[dict[str, str]],
    prompt_template: str,
    output_path: Path,
    dtype: str,
    device: str,
    use_chat_template: bool,
    batch_size: int,
    max_new_tokens: int,
    repetition_penalty: float,
) -> None:
    loaded = load_model_for_decoding(
        model_path=model_path,
        dtype=dtype,
        device=device,
    )
    tokenizer = loaded["tokenizer"]
    model = loaded["model"]
    metadata = loaded["metadata"]
    allowed_token_ids = metadata["allowed_token_ids"]
    token_id_to_keyword = build_reverse_keyword_lookup(metadata["keyword_to_token_id"])

    batches = chunk_list(dataset, batch_size)
    outputs: list[dict[str, str]] = []

    with tqdm(total=len(dataset), desc=model_path.name, unit="sample") as progress:
        for batch in batches:
            prompts = [
                render_prompt(prompt_template, item["context"])
                for item in batch
            ]
            hypotheses = generate_hypotheses_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                allowed_token_ids=allowed_token_ids,
                token_id_to_keyword=token_id_to_keyword,
                max_new_tokens=max_new_tokens,
                use_chat_template=use_chat_template,
                repetition_penalty=repetition_penalty,
            )
            for item, hypothesis in zip(batch, hypotheses, strict=False):
                outputs.append({"id": item["id"], "hyp": hypothesis})
            progress.update(len(batch))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_generation_outputs(output_path, outputs)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    prompt_template = load_prompt_template(args.prompt_file)
    dataset = load_task_dataset(args.dataset_json, args.input_field)

    for model_path in args.model_paths:
        output_path = args.output_dir / build_output_filename(model_path)
        run_model(
            model_path=model_path,
            dataset=dataset,
            prompt_template=prompt_template,
            output_path=output_path,
            dtype=args.dtype,
            device=args.device,
            use_chat_template=args.chat_template,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
        print(output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
