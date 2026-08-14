"""Training utilities for pictogram-id causal language models."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

from praact_pictogram_prediction.model_preparation import resolve_torch_dtype


def load_pictogram_dataset(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Expected dataset JSON root to be a list.")

    records: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Expected every dataset item to be an object.")
        if "pictos" not in item:
            raise ValueError("Each dataset item must contain a 'pictos' field.")
        if not isinstance(item["pictos"], list):
            raise ValueError("Each 'pictos' field must be a list.")
        records.append(item)

    return records


def maybe_limit_dataset(
    dataset: list[dict[str, Any]],
    max_samples: int | None,
) -> list[dict[str, Any]]:
    if max_samples is None:
        return dataset
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1 when provided.")
    return dataset[:max_samples]


def pictogram_ids_to_text(pictogram_ids: list[Any]) -> str:
    return " ".join(str(pictogram_id) for pictogram_id in pictogram_ids)


def tokenize_pictogram_sequence(
    tokenizer: Any,
    pictogram_ids: list[Any],
    max_length: int,
    skip_unknown_pictos: bool,
) -> dict[str, list[int]] | None:
    tokens = [str(pictogram_id) for pictogram_id in pictogram_ids]
    if not tokens:
        return None

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    unknown_ids = [
        token
        for token, token_id in zip(tokens, token_ids, strict=False)
        if token_id == tokenizer.unk_token_id
    ]
    if unknown_ids:
        if skip_unknown_pictos:
            return None
        missing = ", ".join(unknown_ids[:10])
        raise ValueError(f"Unknown pictogram IDs in training example: {missing}")

    if tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)

    token_ids = token_ids[:max_length]
    if len(token_ids) < 2:
        return None

    return {
        "input_ids": token_ids,
        "attention_mask": [1] * len(token_ids),
        "labels": list(token_ids),
    }


class PictogramSequenceDataset(Dataset):
    def __init__(self, features: list[dict[str, list[int]]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]


@dataclass
class CausalPictogramCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for feature in features:
            padding_length = max_length - len(feature["input_ids"])
            batch_input_ids.append(
                feature["input_ids"] + [self.pad_token_id] * padding_length
            )
            batch_attention_mask.append(
                feature["attention_mask"] + [0] * padding_length
            )
            batch_labels.append(feature["labels"] + [-100] * padding_length)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def build_pictogram_sequence_dataset(
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_length: int,
    skip_unknown_pictos: bool,
) -> tuple[PictogramSequenceDataset, int]:
    features: list[dict[str, list[int]]] = []
    skipped_count = 0

    for record in records:
        feature = tokenize_pictogram_sequence(
            tokenizer=tokenizer,
            pictogram_ids=record["pictos"],
            max_length=max_length,
            skip_unknown_pictos=skip_unknown_pictos,
        )
        if feature is None:
            skipped_count += 1
            continue
        features.append(feature)

    if not features:
        raise ValueError("No training features were produced from the dataset.")

    return PictogramSequenceDataset(features), skipped_count


def train_pictogram_model(
    model_path: Path,
    output_dir: Path,
    train_json: Path,
    valid_json: Path | None,
    dtype: str,
    max_length: int,
    num_train_epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    save_total_limit: int,
    seed: int,
    use_cpu: bool,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    skip_unknown_pictos: bool,
    report_to: str | None,
    run_name: str | None,
    wandb_project: str | None,
) -> dict[str, Any]:
    train_records = maybe_limit_dataset(
        load_pictogram_dataset(train_json),
        max_train_samples,
    )
    valid_records = (
        maybe_limit_dataset(load_pictogram_dataset(valid_json), max_eval_samples)
        if valid_json is not None
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        raise ValueError("Prepared pictogram tokenizer must define a pad token.")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=resolve_torch_dtype(dtype),
    )
    model.config.use_cache = False

    train_dataset, skipped_train_count = build_pictogram_sequence_dataset(
        tokenizer=tokenizer,
        records=train_records,
        max_length=max_length,
        skip_unknown_pictos=skip_unknown_pictos,
    )

    eval_dataset = None
    skipped_eval_count = 0
    if valid_records is not None:
        eval_dataset, skipped_eval_count = build_pictogram_sequence_dataset(
            tokenizer=tokenizer,
            records=valid_records,
            max_length=max_length,
            skip_unknown_pictos=skip_unknown_pictos,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    if wandb_project is not None:
        os.environ["WANDB_PROJECT"] = wandb_project

    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / effective_batch_size))
    total_training_steps = max(1, math.ceil(num_train_epochs * steps_per_epoch))
    warmup_steps = int(total_training_steps * warmup_ratio)

    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=eval_dataset is not None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_steps=eval_steps if eval_dataset is not None else None,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        report_to=report_to or "none",
        run_name=run_name,
        remove_unused_columns=False,
        seed=seed,
        bf16=(dtype == "bf16"),
        fp16=(dtype == "fp16"),
        save_only_model=False,
        use_cpu=use_cpu,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalPictogramCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    source_metadata_path = model_path / "pictogram_vocabulary.json"
    if source_metadata_path.exists():
        target_metadata_path = output_dir / source_metadata_path.name
        target_metadata_path.write_text(
            source_metadata_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    return {
        "output_dir": output_dir,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
        "skipped_train_examples": skipped_train_count,
        "skipped_eval_examples": skipped_eval_count,
    }
