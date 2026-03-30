"""Restricted decoding utilities for Praact models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessor
from transformers import LogitsProcessorList

from praact.model_expansion import resolve_device
from praact.model_expansion import resolve_torch_dtype


class AllowedTokensLogitsProcessor(LogitsProcessor):
    """Masks logits so generation can only produce a predefined token subset."""

    def __init__(self, allowed_token_ids: list[int], extra_token_ids: list[int] | None = None) -> None:
        if not allowed_token_ids and not extra_token_ids:
            raise ValueError("allowed_token_ids must not be empty.")
        combined_ids = list(allowed_token_ids)
        if extra_token_ids:
            combined_ids.extend(extra_token_ids)
        self.allowed_token_ids = sorted(set(combined_ids))

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
        "generation_token_ids",
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


def load_prompt_template(prompt_path: Path) -> str:
    with prompt_path.open(encoding="utf-8") as file:
        return file.read().strip()


def render_prompt(template: str, sentence: str | None = None) -> str:
    if "{sentence}" in template:
        if sentence is None:
            raise ValueError("Prompt template requires a sentence, but none was provided.")
        return template.replace("{sentence}", sentence)

    if sentence is None:
        return template

    return sentence


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


def format_prompt_for_generation(
    tokenizer: Any,
    prompt: str,
    use_chat_template: bool,
) -> str:
    if not use_chat_template:
        return prompt

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("This tokenizer does not provide apply_chat_template().")

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_generation_logits_processor(
    tokenizer: Any,
    allowed_token_ids: list[int],
) -> LogitsProcessorList:
    eos_token_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)

    return LogitsProcessorList(
        [
            AllowedTokensLogitsProcessor(
                allowed_token_ids,
                extra_token_ids=eos_token_ids,
            )
        ]
    )


def describe_top_tokens(
    scores: torch.FloatTensor,
    tokenizer: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    top_values, top_indices = torch.topk(scores, k=min(top_k, scores.shape[-1]))
    items: list[dict[str, Any]] = []

    for value, index in zip(top_values.tolist(), top_indices.tolist(), strict=False):
        items.append(
            {
                "token_id": int(index),
                "token": tokenizer.convert_ids_to_tokens(int(index)),
                "score": float(value),
            }
        )

    return items


def inspect_first_step(
    model: Any,
    tokenizer: Any,
    prompt: str,
    allowed_token_ids: list[int],
    use_chat_template: bool,
    top_k: int = 10,
) -> dict[str, Any]:
    device = model.device
    formatted_prompt = format_prompt_for_generation(
        tokenizer=tokenizer,
        prompt=prompt,
        use_chat_template=use_chat_template,
    )
    encoded_prompt = tokenizer(formatted_prompt, return_tensors="pt")
    encoded_prompt = {name: tensor.to(device) for name, tensor in encoded_prompt.items()}

    with torch.no_grad():
        outputs = model(**encoded_prompt)

    raw_scores = outputs.logits[0, -1, :].detach().clone()
    masked_scores = build_generation_logits_processor(
        tokenizer=tokenizer,
        allowed_token_ids=allowed_token_ids,
    )(encoded_prompt["input_ids"], raw_scores.unsqueeze(0))[0]

    allowed_scores: list[dict[str, Any]] = []
    for token_id in allowed_token_ids:
        allowed_scores.append(
            {
                "token_id": int(token_id),
                "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                "score": float(masked_scores[token_id].item()),
            }
        )

    allowed_scores.sort(key=lambda item: item["score"], reverse=True)

    eos_score = None
    if tokenizer.eos_token_id is not None:
        eos_score = float(masked_scores[tokenizer.eos_token_id].item())

    return {
        "formatted_prompt": formatted_prompt,
        "raw_top_tokens": describe_top_tokens(raw_scores, tokenizer, top_k),
        "masked_top_tokens": describe_top_tokens(masked_scores, tokenizer, top_k),
        "top_allowed_tokens": allowed_scores[:top_k],
        "eos_token_id": tokenizer.eos_token_id,
        "eos_score": eos_score,
    }


def generate_hypothesis(
    model: Any,
    tokenizer: Any,
    prompt: str,
    allowed_token_ids: list[int],
    token_id_to_keyword: dict[int, str],
    max_new_tokens: int,
    use_chat_template: bool = False,
    repetition_penalty: float = 1.1,
) -> str:
    device = model.device
    formatted_prompt = format_prompt_for_generation(
        tokenizer=tokenizer,
        prompt=prompt,
        use_chat_template=use_chat_template,
    )
    encoded_prompt = tokenizer(formatted_prompt, return_tensors="pt")
    encoded_prompt = {name: tensor.to(device) for name, tensor in encoded_prompt.items()}

    prompt_length = encoded_prompt["input_ids"].shape[1]
    logits_processor = build_generation_logits_processor(tokenizer, allowed_token_ids)

    with torch.no_grad():
        generated = model.generate(
            **encoded_prompt,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_token_ids = generated[0, prompt_length:].tolist()
    return decode_generated_token_ids(generated_token_ids, token_id_to_keyword)


def chunk_list(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def generate_hypotheses_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    allowed_token_ids: list[int],
    token_id_to_keyword: dict[int, str],
    max_new_tokens: int,
    use_chat_template: bool = False,
    repetition_penalty: float = 1.1,
) -> list[str]:
    device = model.device
    formatted_prompts = [
        format_prompt_for_generation(
            tokenizer=tokenizer,
            prompt=prompt,
            use_chat_template=use_chat_template,
        )
        for prompt in prompts
    ]

    encoded_batch = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
    )
    encoded_batch = {name: tensor.to(device) for name, tensor in encoded_batch.items()}
    prompt_lengths = encoded_batch["attention_mask"].sum(dim=1).tolist()
    logits_processor = build_generation_logits_processor(tokenizer, allowed_token_ids)

    with torch.no_grad():
        generated = model.generate(
            **encoded_batch,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    hypotheses: list[str] = []
    for row_index, prompt_length in enumerate(prompt_lengths):
        generated_token_ids = generated[row_index, int(prompt_length) :].tolist()
        hypotheses.append(
            decode_generated_token_ids(generated_token_ids, token_id_to_keyword)
        )

    return hypotheses


def load_model_for_decoding(
    model_path: Path,
    dtype: str = "auto",
    device: str = "auto",
) -> dict[str, Any]:
    resolved_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer has no pad_token_id and no eos_token to reuse as padding."
            )
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=resolve_torch_dtype(dtype),
    )
    model = model.to(resolved_device)
    metadata = load_praact_vocab_metadata(model_path)

    return {
        "tokenizer": tokenizer,
        "model": model,
        "metadata": metadata,
        "device": resolved_device,
    }


def generate_from_dataset(
    model: Any,
    tokenizer: Any,
    dataset: list[dict[str, Any]],
    allowed_token_ids: list[int],
    token_id_to_keyword: dict[int, str],
    max_new_tokens: int,
    use_chat_template: bool = False,
    repetition_penalty: float = 1.1,
    batch_size: int = 1,
) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    batches = chunk_list(dataset, batch_size)
    print(f"Processing {len(dataset)} samples in {len(batches)} batches..., bs={batch_size}")

    with tqdm(total=len(batches), desc="Generating", unit="batch") as progress:
        for batch in batches:
            prompts = [item["src"] for item in batch]
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
            progress.update(1)

    return outputs


def save_generation_outputs(output_path: Path, outputs: list[dict[str, str]]) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(outputs, file, ensure_ascii=False, indent=2)
        file.write("\n")
