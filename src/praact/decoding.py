"""Restricted decoding utilities for Praact models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessor
from transformers import LogitsProcessorList

from praact.model_expansion import resolve_torch_dtype


class AllowedTokensLogitsProcessor(LogitsProcessor):
    """Masks logits so generation can only produce a predefined token subset."""

    def __init__(self, allowed_token_ids: list[int]) -> None:
        if not allowed_token_ids:
            raise ValueError("allowed_token_ids must not be empty.")
        self.allowed_token_ids = allowed_token_ids

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        masked_scores = torch.full_like(scores, float("-inf"))
        masked_scores[..., self.allowed_token_ids] = scores[..., self.allowed_token_ids]
        return masked_scores


def load_praact_vocab_metadata(model_path: Path) -> dict[str, Any]:
    metadata_path = model_path / "praact_vocab.json"
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    required_keys = {
        "added_token_ids",
        "existing_token_ids",
        "allowed_token_ids",
        "keyword_to_token_id",
    }
    missing_keys = required_keys - metadata.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"praact_vocab.json is missing required keys: {missing}.")

    return metadata


def load_generation_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Expected the dataset JSON root to be a list.")

    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Expected every dataset item to be an object.")
        if "id" not in item or "src" not in item:
            raise ValueError("Each dataset item must include 'id' and 'src'.")

    return payload


def build_reverse_keyword_lookup(keyword_to_token_id: dict[str, int]) -> dict[int, str]:
    return {token_id: keyword for keyword, token_id in keyword_to_token_id.items()}


def decode_generated_token_ids(
    token_ids: list[int],
    token_id_to_keyword: dict[int, str],
) -> str:
    keywords: list[str] = []
    for token_id in token_ids:
        keyword = token_id_to_keyword.get(token_id)
        if keyword is None:
            continue
        keywords.append(keyword.replace("_", " "))

    return " ".join(keywords)


def generate_hypothesis(
    model: Any,
    tokenizer: Any,
    prompt: str,
    allowed_token_ids: list[int],
    token_id_to_keyword: dict[int, str],
    max_new_tokens: int,
) -> str:
    device = model.device
    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    encoded_prompt = {name: tensor.to(device) for name, tensor in encoded_prompt.items()}

    prompt_length = encoded_prompt["input_ids"].shape[1]
    logits_processor = LogitsProcessorList(
        [AllowedTokensLogitsProcessor(allowed_token_ids)]
    )

    with torch.no_grad():
        generated = model.generate(
            **encoded_prompt,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_token_ids = generated[0, prompt_length:].tolist()
    return decode_generated_token_ids(generated_token_ids, token_id_to_keyword)


def load_model_for_decoding(model_path: Path, dtype: str = "auto") -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=resolve_torch_dtype(dtype),
    )
    metadata = load_praact_vocab_metadata(model_path)

    return {
        "tokenizer": tokenizer,
        "model": model,
        "metadata": metadata,
    }


def generate_from_dataset(
    model: Any,
    tokenizer: Any,
    dataset: list[dict[str, Any]],
    allowed_token_ids: list[int],
    token_id_to_keyword: dict[int, str],
    max_new_tokens: int,
) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []

    for item in dataset:
        hypothesis = generate_hypothesis(
            model=model,
            tokenizer=tokenizer,
            prompt=item["src"],
            allowed_token_ids=allowed_token_ids,
            token_id_to_keyword=token_id_to_keyword,
            max_new_tokens=max_new_tokens,
        )
        outputs.append({"id": item["id"], "hyp": hypothesis})

    return outputs


def save_generation_outputs(output_path: Path, outputs: list[dict[str, str]]) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(outputs, file, ensure_ascii=False, indent=2)
        file.write("\n")
