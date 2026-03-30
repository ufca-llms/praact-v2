"""Command-line interface for the Praact package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from praact.decoding import build_reverse_keyword_lookup
from praact.decoding import generate_from_dataset
from praact.decoding import generate_hypothesis
from praact.decoding import inspect_first_step
from praact.decoding import load_generation_dataset
from praact.decoding import load_model_for_decoding
from praact.decoding import load_prompt_template
from praact.decoding import render_prompt
from praact.decoding import save_generation_outputs
from praact.evaluation import evaluate_predictions
from praact.model_expansion import add_missing_keywords_to_model
from praact.model_expansion import build_output_dir
from praact.model_expansion import build_praact_vocab_metadata
from praact.model_expansion import deduplicate_keywords
from praact.model_expansion import extract_keywords
from praact.model_expansion import load_items
from praact.model_expansion import save_praact_vocab_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="praact",
        description="Praact command-line tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    expand_parser = subparsers.add_parser(
        "expand",
        help="Expand a causal language model with Praact keywords.",
    )
    expand_parser.add_argument(
        "json_path",
        type=Path,
        help="Path to a JSON file with the same structure as data/arasaac_en.json.",
    )
    expand_parser.add_argument(
        "model_id",
        help="Hugging Face causal model identifier.",
    )
    expand_parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory where the updated tokenizer and model will be saved. "
            "Defaults to outputs/<model-id>."
        ),
    )
    expand_parser.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Torch dtype used to load the model.",
    )
    expand_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Execution device used to load and update the model.",
    )

    decode_parser = subparsers.add_parser(
        "decode",
        help="Generate text with logits restricted to the Praact vocabulary.",
    )
    decode_parser.add_argument(
        "model_path",
        type=Path,
        help="Path to the expanded model directory containing praact_vocab.json.",
    )
    decode_parser.add_argument(
        "--prompt",
        help="Single input prompt to complete.",
    )
    decode_parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Path to a prompt template file. Use {sentence} as placeholder.",
    )
    decode_parser.add_argument(
        "--input-json",
        type=Path,
        help="JSON dataset with items containing 'id' and 'src'.",
    )
    decode_parser.add_argument(
        "--output-json",
        type=Path,
        help="Where to save batch generation results as a JSON list.",
    )
    decode_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum number of new tokens to generate.",
    )
    decode_parser.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Torch dtype used to load the model.",
    )
    decode_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Execution device used during generation.",
    )
    decode_parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Format the prompt with tokenizer.apply_chat_template().",
    )
    decode_parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty applied to repeated generated tokens.",
    )
    decode_parser.add_argument(
        "--debug-first-step",
        action="store_true",
        help="Print first-step logits before and after masking.",
    )
    decode_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used for dataset generation.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate predictions with ImageCLEF ToPicto style metrics.",
    )
    evaluate_parser.add_argument(
        "predictions_json",
        type=Path,
        help="Path to a JSON file with predictions containing 'id' and 'hyp'.",
    )
    evaluate_parser.add_argument(
        "references_json",
        type=Path,
        help="Path to a JSON file with references containing 'id' and 'tgt'.",
    )
    return parser


def run_expand(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        items = load_items(args.json_path)
    except FileNotFoundError:
        parser.error(f"JSON file not found: {args.json_path}")
    except ValueError as exc:
        parser.error(str(exc))

    keywords = deduplicate_keywords(extract_keywords(items))
    output_dir = build_output_dir(args.model_id, args.output_dir)

    try:
        result = add_missing_keywords_to_model(
            args.model_id,
            keywords,
            dtype=args.dtype,
            device=args.device,
        )
    except Exception as exc:
        parser.error(str(exc))

    output_dir.mkdir(parents=True, exist_ok=True)
    result["tokenizer"].save_pretrained(output_dir)
    result["model"].save_pretrained(output_dir)
    vocab_metadata = build_praact_vocab_metadata(
        keywords=keywords,
        added_keywords=result["missing_keywords"],
        model_id=args.model_id,
        keyword_to_token_id=result["keyword_to_token_id"],
    )
    vocab_metadata_path = save_praact_vocab_metadata(output_dir, vocab_metadata)

    json.dump(
        {
            "existing_keyword_count": len(result["existing_keywords"]),
            "added_keyword_count": len(result["missing_keywords"]),
        },
        fp=sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


def run_decode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if bool(args.prompt) == bool(args.input_json):
        parser.error("Use exactly one of --prompt or --input-json.")

    if args.input_json and args.output_json is None:
        parser.error("--output-json is required when using --input-json.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    prompt_template = None
    if args.prompt_file is not None:
        try:
            prompt_template = load_prompt_template(args.prompt_file)
        except FileNotFoundError as exc:
            parser.error(str(exc))

    try:
        loaded = load_model_for_decoding(
            args.model_path,
            dtype=args.dtype,
            device=args.device,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))
    except Exception as exc:
        parser.error(str(exc))

    tokenizer = loaded["tokenizer"]
    model = loaded["model"]
    metadata = loaded["metadata"]
    allowed_token_ids = metadata["allowed_token_ids"]
    token_id_to_keyword = build_reverse_keyword_lookup(metadata["keyword_to_token_id"])

    if args.prompt:
        final_prompt = render_prompt(prompt_template, args.prompt) if prompt_template else args.prompt

        if args.debug_first_step:
            debug_info = inspect_first_step(
                model=model,
                tokenizer=tokenizer,
                prompt=final_prompt,
                allowed_token_ids=allowed_token_ids,
                use_chat_template=args.chat_template,
            )
            json.dump(debug_info, fp=sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
            return 0

        hypothesis = generate_hypothesis(
            model=model,
            tokenizer=tokenizer,
            prompt=final_prompt,
            allowed_token_ids=allowed_token_ids,
            token_id_to_keyword=token_id_to_keyword,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=args.chat_template,
            repetition_penalty=args.repetition_penalty,
        )
        sys.stdout.write(f"{hypothesis}\n")
        return 0

    try:
        dataset = load_generation_dataset(args.input_json)
        if prompt_template is not None:
            dataset = [
                {
                    **item,
                    "src": render_prompt(prompt_template, item["src"]),
                }
                for item in dataset
            ]
        outputs = generate_from_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            allowed_token_ids=allowed_token_ids,
            token_id_to_keyword=token_id_to_keyword,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=args.chat_template,
            repetition_penalty=args.repetition_penalty,
            batch_size=args.batch_size,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))
    except Exception as exc:
        parser.error(str(exc))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    save_generation_outputs(args.output_json, outputs)
    sys.stdout.write(f"{args.output_json}\n")
    return 0


def run_evaluate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        metrics = evaluate_predictions(args.predictions_json, args.references_json)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))

    json.dump(metrics, fp=sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "expand":
        return run_expand(args, parser)

    if args.command == "decode":
        return run_decode(args, parser)

    if args.command == "evaluate":
        return run_evaluate(args, parser)

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
