#!/usr/bin/env python3
"""Evaluate an instruction-tuned Hugging Face model on MMLU-Pro."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DEFAULT_MODEL,
    EvalExample,
    build_mcq_prompt,
    configure_logging,
    load_model_and_tokenizer,
    run_multiple_choice_eval,
    save_results,
    summarize_predictions,
)


def build_mmlu_pro_examples(
    split: str = "test",
    categories: list[str] | None = None,
    limit: int | None = None,
) -> list[EvalExample]:
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    if categories:
        allowed = {c.lower() for c in categories}
        dataset = dataset.filter(lambda x: str(x["category"]).lower() in allowed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    examples: list[EvalExample] = []
    for doc in dataset:
        options = [str(opt).strip() for opt in doc["options"] if str(opt).strip()]
        # Prefer the letter in `answer`; fall back to answer_index.
        if doc.get("answer"):
            gold = str(doc["answer"]).strip().upper()
        else:
            gold = chr(ord("A") + int(doc["answer_index"]))

        prompt = build_mcq_prompt(doc["question"], options)
        examples.append(
            EvalExample(
                example_id=str(doc["question_id"]),
                prompt=prompt,
                gold=gold,
                metadata={
                    "num_choices": len(options),
                    "category": doc.get("category"),
                    "src": doc.get("src"),
                    "choices": options,
                },
            )
        )
    return examples


def per_category_accuracy(predictions) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for pred in predictions:
        category = str(pred.metadata.get("category", "unknown"))
        buckets[category].append(pred.correct)

    stats: dict[str, dict[str, float | int]] = {}
    for category, flags in sorted(buckets.items()):
        n = len(flags)
        n_correct = sum(1 for x in flags if x)
        stats[category] = {
            "num_examples": n,
            "num_correct": n_correct,
            "accuracy": (n_correct / n) if n else 0.0,
        }
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional category filter, e.g. --categories math physics",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/mmlu_pro.json"),
        help="Where to write JSON results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable library logging (transformers/datasets/HF warnings)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    examples = build_mmlu_pro_examples(
        split=args.split,
        categories=args.categories,
        limit=args.limit,
    )
    print(f"Loaded {len(examples)} MMLU-Pro examples from split={args.split}")

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
    )
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
        f"MMLU-Pro accuracy: {summary['accuracy']:.4f} "
        f"({summary['num_correct']}/{summary['num_examples']}), "
        f"parse_rate={summary['parse_rate']:.4f}"
    )
    for category, stats in category_stats.items():
        print(
            f"  {category}: {stats['accuracy']:.4f} "
            f"({stats['num_correct']}/{stats['num_examples']})"
        )

    save_results(
        args.output,
        benchmark="mmlu_pro",
        model_name=args.model,
        summary=summary,
        predictions=predictions,
        extra={
            "split": args.split,
            "categories": args.categories,
            "limit": args.limit,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "per_category": category_stats,
        },
    )


if __name__ == "__main__":
    main()
