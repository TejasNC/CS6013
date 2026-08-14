#!/usr/bin/env python3
"""Run GPQA Diamond and/or MMLU-Pro evaluations with a shared HF model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DEFAULT_MODEL,
    configure_logging,
    load_model_and_tokenizer,
    run_multiple_choice_eval,
    save_results,
    summarize_predictions,
)
from evaluate_gpqa import build_gpqa_examples
from evaluate_mmlu_pro import build_mmlu_pro_examples, per_category_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["gpqa", "mmlu_pro"],
        choices=["gpqa", "mmlu_pro"],
        help="Which benchmarks to run",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Per-benchmark example limit")
    parser.add_argument("--seed", type=int, default=0, help="GPQA choice-shuffle seed")
    parser.add_argument("--mmlu-split", default="test", choices=["test", "validation"])
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable library logging (transformers/datasets/HF warnings)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
    )

    if "gpqa" in args.benchmarks:
        examples = build_gpqa_examples(limit=args.limit, seed=args.seed)
        print(f"[gpqa] Evaluating {len(examples)} examples")
        predictions = run_multiple_choice_eval(
            examples,
            model,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            valid_letters=["A", "B", "C", "D"],
            desc="GPQA Diamond",
        )
        summary = summarize_predictions(predictions)
        print(
            f"[gpqa] accuracy={summary['accuracy']:.4f} "
            f"({summary['num_correct']}/{summary['num_examples']})"
        )
        save_results(
            args.output_dir / "gpqa_diamond.json",
            benchmark="gpqa_diamond",
            model_name=args.model,
            summary=summary,
            predictions=predictions,
            extra={"seed": args.seed, "limit": args.limit},
        )

    if "mmlu_pro" in args.benchmarks:
        examples = build_mmlu_pro_examples(split=args.mmlu_split, limit=args.limit)
        print(f"[mmlu_pro] Evaluating {len(examples)} examples")
        predictions = run_multiple_choice_eval(
            examples,
            model,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            desc="MMLU-Pro",
        )
        summary = summarize_predictions(predictions)
        category_stats = per_category_accuracy(predictions)
        print(
            f"[mmlu_pro] accuracy={summary['accuracy']:.4f} "
            f"({summary['num_correct']}/{summary['num_examples']})"
        )
        save_results(
            args.output_dir / "mmlu_pro.json",
            benchmark="mmlu_pro",
            model_name=args.model,
            summary=summary,
            predictions=predictions,
            extra={
                "split": args.mmlu_split,
                "limit": args.limit,
                "per_category": category_stats,
            },
        )


if __name__ == "__main__":
    main()
