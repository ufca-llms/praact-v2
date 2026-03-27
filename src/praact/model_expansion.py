"""Core logic for expanding a causal language model vocabulary."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_keyword(value: str) -> str:
    cleaned = value.strip()
    if " " not in cleaned:
        return cleaned
    return "_".join(cleaned.split())


def denormalize_keyword(keyword: str) -> str:
    return keyword.replace("_", " ")


def load_items(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Expected the root JSON value to be a list of items.")

    invalid_items = [item for item in payload if not isinstance(item, dict)]
    if invalid_items:
        raise ValueError("Expected every item in the JSON array to be an object.")

    return payload


def extract_keywords(items: list[dict[str, Any]]) -> list[str]:
    keywords: list[str] = []

    for item in items:
        raw_keywords = item.get("keywords", [])
        if not isinstance(raw_keywords, list):
            continue

        for keyword_entry in raw_keywords:
            if isinstance(keyword_entry, dict):
                keyword_value = keyword_entry.get("keyword")
            elif isinstance(keyword_entry, str):
                keyword_value = keyword_entry
            else:
                continue

            if not isinstance(keyword_value, str):
                continue

            normalized = normalize_keyword(keyword_value)
            if normalized:
                keywords.append(normalized)

    return keywords


def deduplicate_keywords(keywords: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_keywords: list[str] = []

    for keyword in keywords:
        if keyword in seen:
            continue
        seen.add(keyword)
        unique_keywords.append(keyword)

    return unique_keywords


def build_output_dir(model_id: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir

    safe_model_id = model_id.replace("/", "--")
    return Path("outputs") / safe_model_id


def resolve_torch_dtype(dtype: str) -> torch.dtype | str:
    if dtype == "auto":
        return "auto"
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but no CUDA device is available.")
        return "cuda"

    if device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested, but MPS is not available.")
        return "mps"

    if device == "cpu":
        return "cpu"

    raise ValueError(f"Unsupported device: {device}")


def build_praact_vocab_metadata(
    keywords: list[str],
    added_keywords: list[str],
    model_id: str,
    keyword_to_token_id: dict[str, int],
) -> dict[str, Any]:
    added_keyword_set = set(added_keywords)
    added_token_ids: list[int] = []
    existing_token_ids: list[int] = []

    for keyword in keywords:
        token_id = keyword_to_token_id[keyword]
        if keyword in added_keyword_set:
            added_token_ids.append(token_id)
        else:
            existing_token_ids.append(token_id)

    allowed_token_ids = existing_token_ids + added_token_ids
    generation_token_ids: list[int] = []

    for keyword in keywords:
        if not is_generation_keyword(keyword):
            continue
        generation_token_ids.append(keyword_to_token_id[keyword])

    return {
        "model_id": model_id,
        "added_token_ids": added_token_ids,
        "existing_token_ids": existing_token_ids,
        "allowed_token_ids": allowed_token_ids,
        "generation_token_ids": generation_token_ids,
        "keyword_to_token_id": keyword_to_token_id,
    }


def is_generation_keyword(keyword: str) -> bool:
    denormalized = denormalize_keyword(keyword).strip()
    if not denormalized:
        return False

    for character in denormalized:
        if unicodedata.category(character).startswith("L"):
            return True

    return False


def save_praact_vocab_metadata(output_dir: Path, metadata: dict[str, Any]) -> Path:
    metadata_path = output_dir / "praact_vocab.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return metadata_path


def compute_mean_embedding(
    token_ids: list[int],
    embedding_matrix: torch.Tensor,
) -> torch.Tensor:
    if not token_ids:
        raise ValueError("Cannot initialize a token from an empty subtoken list.")

    subtoken_embeddings = embedding_matrix[token_ids]
    return subtoken_embeddings.mean(dim=0)


def resolve_existing_token_id(tokenizer: Any, keyword: str) -> int | None:
    surface_form = denormalize_keyword(keyword)
    candidate_texts = [f" {surface_form}", surface_form]

    for candidate_text in candidate_texts:
        token_ids = tokenizer.encode(candidate_text, add_special_tokens=False)
        if len(token_ids) == 1:
            return int(token_ids[0])

    literal_token_id = tokenizer.convert_tokens_to_ids(keyword)
    if literal_token_id is not None and literal_token_id >= 0:
        return int(literal_token_id)

    return None


def add_missing_keywords_to_model(
    model_id: str,
    keywords: list[str],
    dtype: str = "auto",
    device: str = "auto",
) -> dict[str, Any]:
    resolved_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=resolve_torch_dtype(dtype),
    )
    model = model.to(resolved_device)

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise ValueError("The model does not expose an input embedding layer.")

    input_embedding_weights = input_embeddings.weight.detach().clone()
    output_embeddings = model.get_output_embeddings()
    output_embedding_weights = None
    if output_embeddings is not None:
        output_embedding_weights = output_embeddings.weight.detach().clone()

    missing_keywords: list[str] = []
    existing_keywords: list[str] = []
    initialization_token_ids: dict[str, list[int]] = {}
    keyword_to_token_id: dict[str, int] = {}

    for keyword in keywords:
        existing_token_id = resolve_existing_token_id(tokenizer, keyword)
        if existing_token_id is not None:
            keyword_to_token_id[keyword] = existing_token_id
            existing_keywords.append(keyword)
            continue

        subtoken_ids = tokenizer.encode(
            denormalize_keyword(keyword),
            add_special_tokens=False,
        )
        if not subtoken_ids:
            continue

        missing_keywords.append(keyword)
        initialization_token_ids[keyword] = subtoken_ids

    added_token_count = tokenizer.add_tokens(missing_keywords)
    if added_token_count != len(missing_keywords):
        raise ValueError(
            "Failed to add all missing keywords to the tokenizer vocabulary."
        )

    model.resize_token_embeddings(len(tokenizer))

    if missing_keywords:
        input_embeddings = model.get_input_embeddings()
        input_weights = input_embeddings.weight.data

        output_embeddings = model.get_output_embeddings()
        output_weights = None
        if output_embeddings is not None:
            output_weights = output_embeddings.weight.data

        for keyword in missing_keywords:
            token_id = tokenizer.convert_tokens_to_ids(keyword)
            if token_id is None or token_id < 0:
                raise ValueError(f"Tokenizer did not return a valid id for {keyword!r}.")
            keyword_to_token_id[keyword] = int(token_id)

            mean_input_embedding = compute_mean_embedding(
                initialization_token_ids[keyword],
                input_embedding_weights,
            )
            input_weights[token_id] = mean_input_embedding

            if output_weights is not None and output_embedding_weights is not None:
                mean_output_embedding = compute_mean_embedding(
                    initialization_token_ids[keyword],
                    output_embedding_weights,
                )
                output_weights[token_id] = mean_output_embedding

    return {
        "tokenizer": tokenizer,
        "model": model,
        "missing_keywords": missing_keywords,
        "existing_keywords": existing_keywords,
        "keyword_to_token_id": keyword_to_token_id,
        "device": resolved_device,
    }
