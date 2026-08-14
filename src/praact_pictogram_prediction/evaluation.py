"""Evaluation metrics for pictogram-id causal language models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from praact_pictogram_prediction.model_preparation import resolve_device
from praact_pictogram_prediction.model_preparation import resolve_torch_dtype
from praact_pictogram_prediction.training import CausalPictogramCollator
from praact_pictogram_prediction.training import build_pictogram_sequence_dataset
from praact_pictogram_prediction.training import load_pictogram_dataset
from praact_pictogram_prediction.training import maybe_limit_dataset


def summarize_ranks(ranks: list[int]) -> dict[str, float]:
    sorted_ranks = sorted(ranks)
    count = len(sorted_ranks)
    midpoint = count // 2
    if count % 2:
        median_rank = float(sorted_ranks[midpoint])
    else:
        median_rank = (sorted_ranks[midpoint - 1] + sorted_ranks[midpoint]) / 2

    return {
        "mean_rank": sum(sorted_ranks) / count,
        "median_rank": median_rank,
    }


def update_ranking_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    top_k_values: list[int],
    counters: dict[str, float],
    ranks: list[int],
) -> None:
    valid_mask = labels != -100
    if not valid_mask.any():
        return

    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask]
    max_k = max(top_k_values)
    _, top_indices = torch.topk(
        valid_logits,
        k=min(max_k, valid_logits.shape[-1]),
        dim=-1,
    )

    for k in top_k_values:
        matches = top_indices[:, :k].eq(valid_labels.unsqueeze(-1)).any(dim=-1)
        counters[f"accuracy@{k}"] += int(matches.sum().item())

    true_logits = valid_logits.gather(1, valid_labels.unsqueeze(-1))
    batch_ranks = valid_logits.gt(true_logits).sum(dim=-1) + 1
    ranks.extend(int(rank) for rank in batch_ranks.tolist())
    counters["mrr"] += float((1.0 / batch_ranks.float()).sum().item())


def evaluate_pictogram_model(
    model_path: Path,
    eval_json: Path,
    dtype: str = "auto",
    device: str = "auto",
    batch_size: int = 8,
    max_length: int = 128,
    max_eval_samples: int | None = None,
    skip_unknown_pictos: bool = True,
    top_k_values: list[int] | None = None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if max_length < 2:
        raise ValueError("max_length must be at least 2.")

    top_k_values = top_k_values or [1, 3, 5, 10]
    top_k_values = sorted(set(top_k_values))
    if any(k < 1 for k in top_k_values):
        raise ValueError("All top-k values must be at least 1.")

    resolved_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        raise ValueError("Prepared pictogram tokenizer must define a pad token.")

    records = maybe_limit_dataset(
        load_pictogram_dataset(eval_json),
        max_eval_samples,
    )
    dataset, skipped_examples = build_pictogram_sequence_dataset(
        tokenizer=tokenizer,
        records=records,
        max_length=max_length,
        skip_unknown_pictos=skip_unknown_pictos,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CausalPictogramCollator(tokenizer.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=resolve_torch_dtype(dtype),
    )
    model = model.to(resolved_device)
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    counters = {f"accuracy@{k}": 0.0 for k in top_k_values}
    counters["mrr"] = 0.0
    ranks: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {name: tensor.to(resolved_device) for name, tensor in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            valid_token_count = int(shift_labels.ne(-100).sum().item())
            total_nll += float(loss.item())
            total_tokens += valid_token_count

            update_ranking_metrics(
                logits=shift_logits,
                labels=shift_labels,
                top_k_values=top_k_values,
                counters=counters,
                ranks=ranks,
            )

    if total_tokens == 0:
        raise ValueError("No valid prediction targets were produced for evaluation.")

    mean_loss = total_nll / total_tokens
    metrics: dict[str, Any] = {
        "examples": len(dataset),
        "skipped_examples": skipped_examples,
        "prediction_targets": total_tokens,
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
    }

    for k in top_k_values:
        metrics[f"accuracy@{k}"] = counters[f"accuracy@{k}"] / total_tokens

    metrics["mrr"] = counters["mrr"] / total_tokens
    metrics.update(summarize_ranks(ranks))
    return metrics
