"""Evaluation utilities for ImageCLEF ToPicto style metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sacrebleu


def load_predictions(predictions_path: Path) -> list[dict[str, str]]:
    with predictions_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Predictions JSON root must be a list.")

    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each prediction must be an object.")
        if "id" not in item or "hyp" not in item:
            raise ValueError("Each prediction must include 'id' and 'hyp'.")

    return payload


def load_references(references_path: Path) -> list[dict[str, str]]:
    with references_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("References JSON root must be a list.")

    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each reference must be an object.")
        if "id" not in item or "tgt" not in item:
            raise ValueError("Each reference must include 'id' and 'tgt'.")

    return payload


def tokenize_picto_terms(text: str) -> list[str]:
    return text.strip().split()


def count_chunks(matches: list[tuple[int, int]]) -> int:
    if not matches:
        return 0

    chunks = 1
    previous_hyp_index, previous_ref_index = matches[0]

    for hyp_index, ref_index in matches[1:]:
        if hyp_index != previous_hyp_index + 1 or ref_index != previous_ref_index + 1:
            chunks += 1
        previous_hyp_index = hyp_index
        previous_ref_index = ref_index

    return chunks


def meteor_score_tokens(
    hypothesis_tokens: list[str],
    reference_tokens: list[str],
    gamma: float = 0.5,
    beta: float = 3.0,
) -> float:
    if not hypothesis_tokens or not reference_tokens:
        return 0.0

    used_reference_indices: set[int] = set()
    matches: list[tuple[int, int]] = []

    for hyp_index, hyp_token in enumerate(hypothesis_tokens):
        for ref_index, ref_token in enumerate(reference_tokens):
            if ref_index in used_reference_indices:
                continue
            if hyp_token == ref_token:
                matches.append((hyp_index, ref_index))
                used_reference_indices.add(ref_index)
                break

    if not matches:
        return 0.0

    matches_count = len(matches)
    precision = matches_count / len(hypothesis_tokens)
    recall = matches_count / len(reference_tokens)
    fmean = (10 * precision * recall) / (recall + 9 * precision)
    chunks = count_chunks(matches)
    penalty = gamma * ((chunks / matches_count) ** beta)

    return fmean * (1 - penalty) * 100


def edit_distance(hypothesis_tokens: list[str], reference_tokens: list[str]) -> int:
    rows = len(reference_tokens) + 1
    cols = len(hypothesis_tokens) + 1
    dp = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        dp[row][0] = row
    for col in range(cols):
        dp[0][col] = col

    for row in range(1, rows):
        for col in range(1, cols):
            cost = 0 if reference_tokens[row - 1] == hypothesis_tokens[col - 1] else 1
            dp[row][col] = min(
                dp[row - 1][col] + 1,
                dp[row][col - 1] + 1,
                dp[row - 1][col - 1] + cost,
            )

    return dp[-1][-1]


def pictoer_score(
    hypotheses: list[list[str]],
    references: list[list[str]],
) -> float:
    total_errors = 0
    total_reference_tokens = 0

    for hypothesis_tokens, reference_tokens in zip(hypotheses, references, strict=False):
        total_errors += edit_distance(hypothesis_tokens, reference_tokens)
        total_reference_tokens += len(reference_tokens)

    if total_reference_tokens == 0:
        return 0.0

    return (total_errors / total_reference_tokens) * 100


def align_predictions_with_references(
    predictions: list[dict[str, str]],
    references: list[dict[str, str]],
) -> tuple[list[str], list[str]]:
    predictions_by_id = {item["id"]: item["hyp"] for item in predictions}
    references_by_id = {item["id"]: item["tgt"] for item in references}

    missing_prediction_ids = sorted(set(references_by_id) - set(predictions_by_id))
    if missing_prediction_ids:
        preview = ", ".join(missing_prediction_ids[:5])
        raise ValueError(f"Missing predictions for reference ids such as: {preview}")

    ordered_ids = [item["id"] for item in references]
    ordered_hypotheses = [predictions_by_id[item_id] for item_id in ordered_ids]
    ordered_references = [references_by_id[item_id] for item_id in ordered_ids]

    return ordered_hypotheses, ordered_references


def evaluate_predictions(
    predictions_path: Path,
    references_path: Path,
) -> dict[str, Any]:
    predictions = load_predictions(predictions_path)
    references = load_references(references_path)
    hypotheses_text, references_text = align_predictions_with_references(
        predictions,
        references,
    )

    bleu = sacrebleu.corpus_bleu(
        hypotheses_text,
        [references_text],
        tokenize="none",
        force=True,
    )
    hypothesis_tokens = [tokenize_picto_terms(text) for text in hypotheses_text]
    reference_tokens = [tokenize_picto_terms(text) for text in references_text]
    meteor = sum(
        meteor_score_tokens(hypothesis, reference)
        for hypothesis, reference in zip(hypothesis_tokens, reference_tokens, strict=False)
    ) / len(hypothesis_tokens)
    pictoer = pictoer_score(hypothesis_tokens, reference_tokens)

    return {
        "num_samples": len(hypotheses_text),
        "sacrebleu": bleu.score,
        "meteor": meteor,
        "pictoer": pictoer,
    }
