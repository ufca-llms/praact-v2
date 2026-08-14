"""CLI for Praact pictogram prediction model preparation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from praact_pictogram_prediction.evaluation import evaluate_pictogram_model
from praact_pictogram_prediction.model_preparation import (
    prepare_pictogram_prediction_model,
)
from praact_pictogram_prediction.model_preparation import score_pictogram_tokens
from praact_pictogram_prediction.training import train_pictogram_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="praact-pictogram",
        description="Praact pictogram prediction tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-model",
        help="Replace the tokenizer with an ARASAAC pictogram-id tokenizer.",
    )
    prepare_parser.add_argument(
        "json_path",
        type=Path,
        help="Path to ARASAAC JSON, such as data/arasaac_en.json.",
    )
    prepare_parser.add_argument(
        "model_id",
        help="Hugging Face causal model identifier used as the base model.",
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the replaced tokenizer, model, and metadata are saved.",
    )
    prepare_parser.add_argument(
        "--output-vocabulary",
        choices=["pictogram-id", "keyword"],
        default="pictogram-id",
        help="Use pictogram _id values or keyword strings as output tokens.",
    )
    prepare_parser.add_argument(
        "--keyword-strategy",
        choices=["primary", "all"],
        default="primary",
        help="Use only the first keyword per pictogram or average all keywords.",
    )
    prepare_parser.add_argument(
        "--token-format",
        choices=["raw-id", "bracketed-id"],
        default="bracketed-id",
        help="Represent pictogram-id tokens as raw IDs or as <picto:ID> tokens.",
    )
    prepare_parser.add_argument(
        "--aac-only",
        action="store_true",
        help="Only include ARASAAC entries marked with aac=true.",
    )
    prepare_parser.add_argument(
        "--prefix-space",
        action="store_true",
        help=(
            "Prefix captions with a space before tokenization, useful for "
            "GPT-style BPE tokenizers."
        ),
    )
    prepare_parser.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Torch dtype used to load the model.",
    )
    prepare_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="cpu",
        help="Execution device used to prepare the model.",
    )

    score_parser = subparsers.add_parser(
        "score",
        help="Rank pictogram tokens for the next generated position.",
    )
    score_parser.add_argument(
        "model_path",
        type=Path,
        help="Path to a prepared pictogram prediction model.",
    )
    score_parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt whose next token should be scored against pictogram tokens.",
    )
    score_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of pictogram tokens to return.",
    )
    score_parser.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Torch dtype used to load the model.",
    )
    score_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Execution device used during scoring.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train a prepared pictogram-id causal LM on pictogram sequences.",
    )
    train_parser.add_argument(
        "model_path",
        type=Path,
        help="Path to the prepared pictogram tokenizer/model directory.",
    )
    train_parser.add_argument(
        "--train-json",
        type=Path,
        required=True,
        help="JSON dataset containing a 'pictos' list per example.",
    )
    train_parser.add_argument(
        "--valid-json",
        type=Path,
        help="Optional validation JSON dataset containing 'pictos'.",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the trained checkpoint will be saved.",
    )
    train_parser.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="fp32",
        help="Torch dtype used to load the model for training.",
    )
    train_parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum pictogram sequence length including EOS.",
    )
    train_parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
        help="Number of training epochs.",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate used by Trainer.",
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay used by Trainer.",
    )
    train_parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio used by Trainer.",
    )
    train_parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8,
        help="Per-device training batch size.",
    )
    train_parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=8,
        help="Per-device evaluation batch size.",
    )
    train_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    train_parser.add_argument(
        "--logging-steps",
        type=int,
        default=25,
        help="Logging interval in steps.",
    )
    train_parser.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="Evaluation interval in steps.",
    )
    train_parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Checkpoint save interval in steps.",
    )
    train_parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints kept on disk.",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used during training.",
    )
    train_parser.add_argument(
        "--max-train-samples",
        type=int,
        help="Optional cap on the number of training examples.",
    )
    train_parser.add_argument(
        "--max-eval-samples",
        type=int,
        help="Optional cap on the number of validation examples.",
    )
    train_parser.add_argument(
        "--fail-on-unknown-pictos",
        action="store_true",
        help="Fail instead of skipping examples containing IDs outside the tokenizer.",
    )
    train_parser.add_argument(
        "--device",
        choices=["auto", "cpu"],
        default="auto",
        help="Use CPU explicitly or leave device selection to Trainer.",
    )
    train_parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging during training.",
    )
    train_parser.add_argument(
        "--wandb-project",
        help="Optional WANDB_PROJECT value used when --wandb is enabled.",
    )
    train_parser.add_argument(
        "--run-name",
        help="Optional training run name.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a pictogram-id causal LM on a JSON dataset.",
    )
    evaluate_parser.add_argument(
        "model_path",
        type=Path,
        help="Path to a trained pictogram tokenizer/model directory.",
    )
    evaluate_parser.add_argument(
        "--eval-json",
        type=Path,
        required=True,
        help="JSON dataset containing a 'pictos' list per example.",
    )
    evaluate_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Evaluation batch size.",
    )
    evaluate_parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum pictogram sequence length including EOS.",
    )
    evaluate_parser.add_argument(
        "--max-eval-samples",
        type=int,
        help="Optional cap on the number of evaluation examples.",
    )
    evaluate_parser.add_argument(
        "--top-k",
        default="1,3,5,10",
        help="Comma-separated k values for accuracy@k.",
    )
    evaluate_parser.add_argument(
        "--fail-on-unknown-pictos",
        action="store_true",
        help="Fail instead of skipping examples containing IDs outside the tokenizer.",
    )
    evaluate_parser.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Torch dtype used to load the model.",
    )
    evaluate_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Execution device used during evaluation.",
    )
    return parser


def run_prepare_model(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        result = prepare_pictogram_prediction_model(
            arasaac_json=args.json_path,
            model_id=args.model_id,
            output_dir=args.output_dir,
            output_vocabulary=args.output_vocabulary,
            keyword_strategy=args.keyword_strategy,
            token_format=args.token_format,
            aac_only=args.aac_only,
            prefix_space=args.prefix_space,
            dtype=args.dtype,
            device=args.device,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))
    except Exception as exc:
        parser.error(str(exc))

    json.dump(
        {
            "class_count": result["class_count"],
            "vocabulary_size": result["vocabulary_size"],
            "device": result["device"],
            "metadata": str(result["metadata_path"]),
        },
        fp=sys.stdout,
        ensure_ascii=False,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


def run_score(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        result = score_pictogram_tokens(
            model_path=args.model_path,
            prompt=args.prompt,
            dtype=args.dtype,
            device=args.device,
            top_k=args.top_k,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))
    except Exception as exc:
        parser.error(str(exc))

    json.dump(result, fp=sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def run_train(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        result = train_pictogram_model(
            model_path=args.model_path,
            output_dir=args.output_dir,
            train_json=args.train_json,
            valid_json=args.valid_json,
            dtype=args.dtype,
            max_length=args.max_length,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
            use_cpu=(args.device == "cpu"),
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            skip_unknown_pictos=not args.fail_on_unknown_pictos,
            report_to="wandb" if args.wandb else "none",
            run_name=args.run_name,
            wandb_project=args.wandb_project,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))
    except Exception as exc:
        parser.error(str(exc))

    json.dump(result, fp=sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


def parse_top_k(value: str) -> list[int]:
    try:
        top_k_values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--top-k must be a comma-separated list of integers.") from exc

    if not top_k_values:
        raise ValueError("--top-k must contain at least one integer.")
    if any(k < 1 for k in top_k_values):
        raise ValueError("All --top-k values must be at least 1.")
    return top_k_values


def run_evaluate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        result = evaluate_pictogram_model(
            model_path=args.model_path,
            eval_json=args.eval_json,
            dtype=args.dtype,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_eval_samples=args.max_eval_samples,
            skip_unknown_pictos=not args.fail_on_unknown_pictos,
            top_k_values=parse_top_k(args.top_k),
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))
    except Exception as exc:
        parser.error(str(exc))

    json.dump(result, fp=sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-model":
        return run_prepare_model(args, parser)

    if args.command == "score":
        return run_score(args, parser)

    if args.command == "train":
        return run_train(args, parser)

    if args.command == "evaluate":
        return run_evaluate(args, parser)

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
