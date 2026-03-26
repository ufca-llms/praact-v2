"""Command-line interface for the Praact package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from praact.decoding import build_reverse_keyword_lookup
from praact.decoding import generate_from_dataset
from praact.decoding import generate_hypothesis
from praact.decoding import load_generation_dataset
from praact.decoding import load_model_for_decoding
from praact.decoding import save_generation_outputs
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
        result = add_missing_keywords_to_model(args.model_id, keywords, dtype=args.dtype)
    except Exception as exc:
        parser.error(str(exc))

    output_dir.mkdir(parents=True, exist_ok=True)
    result["tokenizer"].save_pretrained(output_dir)
    result["model"].save_pretrained(output_dir)
    vocab_metadata = build_praact_vocab_metadata(
        tokenizer=result["tokenizer"],
        keywords=keywords,
        added_keywords=result["missing_keywords"],
        model_id=args.model_id,
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

    try:
        loaded = load_model_for_decoding(args.model_path, dtype=args.dtype)
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
        hypothesis = generate_hypothesis(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            allowed_token_ids=allowed_token_ids,
            token_id_to_keyword=token_id_to_keyword,
            max_new_tokens=args.max_new_tokens,
        )
        sys.stdout.write(f"{hypothesis}\n")
        return 0

    try:
        dataset = load_generation_dataset(args.input_json)
        outputs = generate_from_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            allowed_token_ids=allowed_token_ids,
            token_id_to_keyword=token_id_to_keyword,
            max_new_tokens=args.max_new_tokens,
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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "expand":
        return run_expand(args, parser)

    if args.command == "decode":
        return run_decode(args, parser)

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
