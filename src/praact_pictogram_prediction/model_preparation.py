"""Prepare causal language models for ARASAAC pictogram prediction."""

from __future__ import annotations

import json
import inspect
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast


METADATA_FILENAME = "pictogram_vocabulary.json"
SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
}


@dataclass(frozen=True)
class PictogramEntry:
    label: str
    token: str
    pictogram_id: int | None
    keywords: list[str]
    token_id: int | None = None


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


def load_arasaac_items(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Expected ARASAAC JSON root to be a list.")

    items: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Expected every ARASAAC item to be an object.")
        items.append(item)
    return items


def extract_item_keywords(item: dict[str, Any]) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()

    raw_keywords = item.get("keywords", [])
    if not isinstance(raw_keywords, list):
        return keywords

    for keyword_entry in raw_keywords:
        if isinstance(keyword_entry, dict):
            keyword = keyword_entry.get("keyword")
        elif isinstance(keyword_entry, str):
            keyword = keyword_entry
        else:
            continue

        if not isinstance(keyword, str):
            continue

        cleaned = " ".join(keyword.strip().split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        keywords.append(cleaned)

    return keywords


def build_pictogram_token(label: str, token_format: str) -> str:
    if token_format == "raw-id":
        return label
    if token_format == "bracketed-id":
        return f"<picto:{label}>"
    raise ValueError("token_format must be 'raw-id' or 'bracketed-id'.")


def resize_token_embeddings_without_random_init(model: Any, vocabulary_size: int) -> None:
    signature = inspect.signature(model.resize_token_embeddings)
    if "mean_resizing" in signature.parameters:
        model.resize_token_embeddings(vocabulary_size, mean_resizing=False)
        return

    model.resize_token_embeddings(vocabulary_size)


def build_pictogram_tokenizer(entries: list[PictogramEntry]) -> PreTrainedTokenizerFast:
    vocab: dict[str, int] = {
        SPECIAL_TOKENS["pad_token"]: 0,
        SPECIAL_TOKENS["unk_token"]: 1,
        SPECIAL_TOKENS["bos_token"]: 2,
        SPECIAL_TOKENS["eos_token"]: 3,
    }

    for entry in entries:
        if entry.token in vocab:
            raise ValueError(f"Duplicate tokenizer token: {entry.token!r}.")
        vocab[entry.token] = len(vocab)

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=SPECIAL_TOKENS["unk_token"]))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=SPECIAL_TOKENS["pad_token"],
        unk_token=SPECIAL_TOKENS["unk_token"],
        bos_token=SPECIAL_TOKENS["bos_token"],
        eos_token=SPECIAL_TOKENS["eos_token"],
    )


def build_arasaac_entries(
    items: list[dict[str, Any]],
    output_vocabulary: str,
    keyword_strategy: str,
    token_format: str,
    aac_only: bool = False,
) -> list[PictogramEntry]:
    if output_vocabulary not in {"pictogram-id", "keyword"}:
        raise ValueError("output_vocabulary must be 'pictogram-id' or 'keyword'.")
    if keyword_strategy not in {"primary", "all"}:
        raise ValueError("keyword_strategy must be 'primary' or 'all'.")

    entries: list[PictogramEntry] = []
    seen_labels: set[str] = set()

    for item in items:
        if aac_only and item.get("aac") is not True:
            continue

        keywords = extract_item_keywords(item)
        if not keywords:
            continue

        pictogram_id_value = item.get("_id")
        pictogram_id = (
            int(pictogram_id_value) if isinstance(pictogram_id_value, int) else None
        )

        if output_vocabulary == "pictogram-id":
            if pictogram_id is None:
                continue
            label = str(pictogram_id)
            entry_keywords = keywords if keyword_strategy == "all" else keywords[:1]
            entries.append(
                PictogramEntry(
                    label=label,
                    token=build_pictogram_token(label, token_format),
                    pictogram_id=pictogram_id,
                    keywords=entry_keywords,
                )
            )
            continue

        selected_keywords = keywords if keyword_strategy == "all" else keywords[:1]
        for keyword in selected_keywords:
            if keyword in seen_labels:
                continue
            seen_labels.add(keyword)
            entries.append(
                PictogramEntry(
                    label=keyword,
                    token=keyword,
                    pictogram_id=pictogram_id,
                    keywords=[keyword],
                )
            )

    if not entries:
        raise ValueError("No pictogram vocabulary entries were produced.")

    return entries


def encode_text(
    tokenizer: Any,
    embedding_matrix: torch.Tensor,
    text: str,
    prefix_space: bool = False,
) -> torch.Tensor:
    tokenization_text = f" {text}" if prefix_space else text
    token_ids = tokenizer.encode(tokenization_text, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Tokenizer produced no tokens for {text!r}.")

    token_tensor = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=embedding_matrix.device,
    )
    return embedding_matrix.index_select(0, token_tensor).mean(dim=0)


def encode_special_token(
    base_tokenizer: Any,
    embedding_matrix: torch.Tensor,
    token_text: str | None,
    fallback: torch.Tensor,
) -> torch.Tensor:
    if token_text is None:
        return fallback

    token_ids = base_tokenizer.encode(token_text, add_special_tokens=False)
    if not token_ids:
        return fallback

    token_tensor = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=embedding_matrix.device,
    )
    return embedding_matrix.index_select(0, token_tensor).mean(dim=0)


def encode_entry(
    tokenizer: Any,
    embedding_matrix: torch.Tensor,
    entry: PictogramEntry,
    prefix_space: bool,
) -> torch.Tensor:
    vectors = [
        encode_text(
            tokenizer=tokenizer,
            embedding_matrix=embedding_matrix,
            text=keyword,
            prefix_space=prefix_space,
        )
        for keyword in entry.keywords
    ]
    return torch.stack(vectors).mean(dim=0)


def entries_with_token_ids(
    tokenizer: Any,
    entries: list[PictogramEntry],
) -> list[PictogramEntry]:
    resolved_entries: list[PictogramEntry] = []
    for entry in entries:
        token_id = tokenizer.convert_tokens_to_ids(entry.token)
        if token_id is None or token_id < 0:
            raise ValueError(f"Tokenizer did not return a valid id for {entry.token!r}.")
        resolved_entries.append(
            PictogramEntry(
                label=entry.label,
                token=entry.token,
                pictogram_id=entry.pictogram_id,
                keywords=entry.keywords,
                token_id=int(token_id),
            )
        )
    return resolved_entries


def initialize_replaced_token_weights(
    model: Any,
    tokenizer: Any,
    entries: list[PictogramEntry],
    entry_vectors: dict[str, torch.Tensor],
    special_vectors: dict[str, torch.Tensor],
) -> None:
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise ValueError("The model does not expose an input embedding layer.")

    input_weights = input_embeddings.weight.data
    output_embeddings = model.get_output_embeddings()
    output_weights = (
        output_embeddings.weight.data if output_embeddings is not None else None
    )

    for token_name, token_text in SPECIAL_TOKENS.items():
        token_id = tokenizer.convert_tokens_to_ids(token_text)
        if token_id is None or token_id < 0:
            raise ValueError(f"Tokenizer did not return a valid id for {token_text!r}.")

        vector = special_vectors[token_name].to(
            input_weights.device,
            dtype=input_weights.dtype,
        )
        input_weights[int(token_id)] = vector
        if output_weights is not None:
            output_weights[int(token_id)] = vector.to(
                output_weights.device,
                dtype=output_weights.dtype,
            )

    for entry in entries:
        if entry.token_id is None:
            raise ValueError(f"Entry {entry.label!r} has no token_id.")

        vector = entry_vectors[entry.label].to(
            input_weights.device,
            dtype=input_weights.dtype,
        )
        input_weights[entry.token_id] = vector

        if output_weights is not None:
            output_weights[entry.token_id] = vector.to(
                output_weights.device,
                dtype=output_weights.dtype,
            )


def save_metadata(
    output_dir: Path,
    model_id: str,
    output_vocabulary: str,
    keyword_strategy: str,
    token_format: str,
    prefix_space: bool,
    entries: list[PictogramEntry],
) -> Path:
    metadata_path = output_dir / METADATA_FILENAME
    metadata = {
        "model_id": model_id,
        "output_vocabulary": output_vocabulary,
        "keyword_strategy": keyword_strategy,
        "token_format": token_format,
        "prefix_space": prefix_space,
        "tokenizer_type": "wordlevel-pictogram-id",
        "special_tokens": SPECIAL_TOKENS,
        "class_count": len(entries),
        "allowed_token_ids": [entry.token_id for entry in entries],
        "label_to_token_id": {entry.label: entry.token_id for entry in entries},
        "token_to_label": {entry.token: entry.label for entry in entries},
        "entries": [asdict(entry) for entry in entries],
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return metadata_path


def prepare_pictogram_prediction_model(
    arasaac_json: Path,
    model_id: str,
    output_dir: Path,
    output_vocabulary: str = "pictogram-id",
    keyword_strategy: str = "primary",
    token_format: str = "bracketed-id",
    aac_only: bool = False,
    prefix_space: bool = False,
    dtype: str = "auto",
    device: str = "cpu",
) -> dict[str, Any]:
    items = load_arasaac_items(arasaac_json)
    entries = build_arasaac_entries(
        items=items,
        output_vocabulary=output_vocabulary,
        keyword_strategy=keyword_strategy,
        token_format=token_format,
        aac_only=aac_only,
    )

    resolved_device = resolve_device(device)
    base_tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=resolve_torch_dtype(dtype),
    )
    model = model.to(resolved_device)

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise ValueError("The model does not expose an input embedding layer.")

    original_input_weights = input_embeddings.weight.detach().clone()
    fallback_special_vector = original_input_weights.mean(dim=0)
    with torch.no_grad():
        entry_vectors = {
            entry.label: encode_entry(
                tokenizer=base_tokenizer,
                embedding_matrix=original_input_weights,
                entry=entry,
                prefix_space=prefix_space,
            )
            for entry in entries
        }
        special_vectors = {
            "pad_token": encode_special_token(
                base_tokenizer,
                original_input_weights,
                base_tokenizer.pad_token,
                fallback_special_vector,
            ),
            "unk_token": encode_special_token(
                base_tokenizer,
                original_input_weights,
                base_tokenizer.unk_token,
                fallback_special_vector,
            ),
            "bos_token": encode_special_token(
                base_tokenizer,
                original_input_weights,
                base_tokenizer.bos_token,
                fallback_special_vector,
            ),
            "eos_token": encode_special_token(
                base_tokenizer,
                original_input_weights,
                base_tokenizer.eos_token,
                fallback_special_vector,
            ),
        }

    tokenizer = build_pictogram_tokenizer(entries)
    resize_token_embeddings_without_random_init(model, len(tokenizer))
    resolved_entries = entries_with_token_ids(tokenizer, entries)
    initialize_replaced_token_weights(
        model=model,
        tokenizer=tokenizer,
        entries=resolved_entries,
        entry_vectors=entry_vectors,
        special_vectors=special_vectors,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    metadata_path = save_metadata(
        output_dir=output_dir,
        model_id=model_id,
        output_vocabulary=output_vocabulary,
        keyword_strategy=keyword_strategy,
        token_format=token_format,
        prefix_space=prefix_space,
        entries=resolved_entries,
    )

    return {
        "class_count": len(resolved_entries),
        "vocabulary_size": len(tokenizer),
        "metadata_path": metadata_path,
        "device": resolved_device,
    }


def load_pictogram_metadata(model_path: Path) -> dict[str, Any]:
    metadata_path = model_path / METADATA_FILENAME
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    required_keys = {"allowed_token_ids", "entries", "label_to_token_id"}
    missing_keys = required_keys - metadata.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"{METADATA_FILENAME} is missing required keys: {missing}.")

    return metadata


def score_pictogram_tokens(
    model_path: Path,
    prompt: str,
    dtype: str = "auto",
    device: str = "auto",
    top_k: int = 10,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    resolved_device = resolve_device(device)
    metadata = load_pictogram_metadata(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=resolve_torch_dtype(dtype),
    )
    model = model.to(resolved_device)

    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    encoded_prompt = {
        name: tensor.to(resolved_device) for name, tensor in encoded_prompt.items()
    }
    allowed_token_ids = [int(token_id) for token_id in metadata["allowed_token_ids"]]

    with torch.no_grad():
        outputs = model(**encoded_prompt)
        next_token_logits = outputs.logits[0, -1, allowed_token_ids]
        probabilities = torch.softmax(next_token_logits, dim=-1)
        top_values, top_indices = torch.topk(
            probabilities,
            k=min(top_k, probabilities.shape[-1]),
        )

    entries = metadata["entries"]
    predictions: list[dict[str, Any]] = []
    for probability, local_index in zip(
        top_values.tolist(),
        top_indices.tolist(),
        strict=False,
    ):
        entry = entries[int(local_index)]
        predictions.append(
            {
                "label": entry["label"],
                "token": entry["token"],
                "token_id": entry["token_id"],
                "pictogram_id": entry["pictogram_id"],
                "keywords": entry["keywords"],
                "probability": float(probability),
            }
        )

    return {
        "prompt": prompt,
        "output_vocabulary": metadata["output_vocabulary"],
        "predictions": predictions,
    }
