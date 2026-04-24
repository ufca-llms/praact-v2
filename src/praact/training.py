"""Training utilities for supervised fine-tuning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

from praact.decoding import load_prompt_template
from praact.decoding import render_prompt
from praact.model_expansion import resolve_torch_dtype

DEFAULT_TRAINING_PROMPT_TEMPLATE = """Transform this sentence into a telegraphic sentence used in Augmentative and Alternative Communication.
Use short content words only.
Avoid articles, auxiliary verbs, and unnecessary function words.

Sentence: {sentence}
Telegraphic:"""

DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_supervised_dataset(json_path: Path) -> list[dict[str, str]]:
    with json_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Expected supervised dataset JSON root to be a list.")

    validated_payload: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each dataset item must be an object.")
        if "src" not in item or "tgt" not in item:
            raise ValueError("Each supervised dataset item must include 'src' and 'tgt'.")

        source = item["src"]
        target = item["tgt"]
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("'src' and 'tgt' must both be strings.")

        validated_payload.append(
            {
                "id": str(item.get("id", "")),
                "src": source,
                "tgt": target,
            }
        )

    return validated_payload


def maybe_limit_dataset(
    dataset: list[dict[str, str]],
    max_samples: int | None,
) -> list[dict[str, str]]:
    if max_samples is None:
        return dataset
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1 when provided.")
    return dataset[:max_samples]


def ensure_training_tokenizer_defaults(tokenizer: Any) -> None:
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError(
                "Tokenizer has no pad_token, eos_token, or unk_token available for padding."
            )


def build_prompt_text(source_text: str, prompt_template: str | None) -> str:
    template = prompt_template or DEFAULT_TRAINING_PROMPT_TEMPLATE
    return render_prompt(template, source_text)


def build_training_texts(
    tokenizer: Any,
    prompt_text: str,
    target_text: str,
    use_chat_template: bool,
) -> tuple[str, str, bool]:
    if use_chat_template:
        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": target_text},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        return prompt_only, full_text, False

    prompt_prefix = prompt_text if prompt_text.endswith((" ", "\n", "\t")) else f"{prompt_text} "
    full_text = prompt_prefix + target_text
    if tokenizer.eos_token is not None:
        full_text = f"{full_text}{tokenizer.eos_token}"
    return prompt_prefix, full_text, True


def tokenize_completion_example(
    tokenizer: Any,
    prompt_text: str,
    target_text: str,
    use_chat_template: bool,
    max_length: int,
) -> dict[str, list[int]] | None:
    prompt_only, full_text, add_special_tokens = build_training_texts(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        target_text=target_text,
        use_chat_template=use_chat_template,
    )

    prompt_encoding = tokenizer(
        prompt_only,
        add_special_tokens=add_special_tokens,
        truncation=True,
        max_length=max_length,
    )
    full_encoding = tokenizer(
        full_text,
        add_special_tokens=add_special_tokens,
        truncation=True,
        max_length=max_length,
    )

    input_ids = full_encoding["input_ids"]
    attention_mask = full_encoding["attention_mask"]
    prompt_length = min(len(prompt_encoding["input_ids"]), len(input_ids))
    labels = list(input_ids)
    labels[:prompt_length] = [-100] * prompt_length

    if all(label == -100 for label in labels):
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class CompletionDataset(Dataset):
    def __init__(self, features: list[dict[str, list[int]]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]


@dataclass
class CompletionOnlyCollator:
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


def build_completion_dataset(
    tokenizer: Any,
    dataset: list[dict[str, str]],
    prompt_template: str | None,
    use_chat_template: bool,
    max_length: int,
) -> CompletionDataset:
    features: list[dict[str, list[int]]] = []

    for item in dataset:
        prompt_text = build_prompt_text(item["src"], prompt_template)
        feature = tokenize_completion_example(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            target_text=item["tgt"],
            use_chat_template=use_chat_template,
            max_length=max_length,
        )
        if feature is None:
            continue
        features.append(feature)

    if not features:
        raise ValueError("No training features were produced from the dataset.")

    return CompletionDataset(features)


def build_lora_model(
    model: Any,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> Any:
    try:
        from peft import LoraConfig
        from peft import TaskType
        from peft import get_peft_model
    except ModuleNotFoundError as exc:
        raise ValueError(
            "LoRA training requires the 'peft' package. Install project dependencies again."
        ) from exc

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=DEFAULT_LORA_TARGET_MODULES,
        ensure_weight_tying=True,
    )

    lora_model = get_peft_model(model, lora_config)
    try:
        lora_model.print_trainable_parameters()
    except Exception:
        pass
    return lora_model


def train_model(
    model_path: Path,
    output_dir: Path,
    train_json: Path,
    valid_json: Path | None,
    prompt_file: Path | None,
    use_chat_template: bool,
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
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    gradient_checkpointing: bool,
    use_cpu: bool,
    max_train_samples: int | None,
    max_eval_samples: int | None,
) -> Path:
    prompt_template = None
    if prompt_file is not None:
        prompt_template = load_prompt_template(prompt_file)

    train_records = maybe_limit_dataset(
        load_supervised_dataset(train_json),
        max_train_samples,
    )
    valid_records = (
        maybe_limit_dataset(load_supervised_dataset(valid_json), max_eval_samples)
        if valid_json is not None
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ensure_training_tokenizer_defaults(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=resolve_torch_dtype(dtype),
    )

    if use_lora:
        model = build_lora_model(
            model=model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False

    train_dataset = build_completion_dataset(
        tokenizer=tokenizer,
        dataset=train_records,
        prompt_template=prompt_template,
        use_chat_template=use_chat_template,
        max_length=max_length,
    )
    eval_dataset = None
    if valid_records is not None:
        eval_dataset = build_completion_dataset(
            tokenizer=tokenizer,
            dataset=valid_records,
            prompt_template=prompt_template,
            use_chat_template=use_chat_template,
            max_length=max_length,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

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
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
        bf16=(dtype == "bf16"),
        fp16=(dtype == "fp16"),
        gradient_checkpointing=gradient_checkpointing,
        save_only_model=False,
        use_cpu=use_cpu,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir
