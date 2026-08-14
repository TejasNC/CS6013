#!/usr/bin/env python3
"""Evaluate an instruction-tuned Hugging Face model on GPQA Diamond."""

from __future__ import annotations

import argparse
import random
import sys
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


def preprocess_gpqa_example(doc: dict, rng: random.Random) -> tuple[list[str], str]:
    choices = [
        str(doc["Incorrect Answer 1"]).strip(),
        str(doc["Incorrect Answer 2"]).strip(),
        str(doc["Incorrect Answer 3"]).strip(),
        str(doc["Correct Answer"]).strip(),
    ]
    rng.shuffle(choices)
    gold_index = choices.index(str(doc["Correct Answer"]).strip())
    gold = chr(ord("A") + gold_index)
    return choices, gold


def build_gpqa_examples(
    limit: int | None = None,
    seed: int = 0,
) -> list[EvalExample]:
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    examples: list[EvalExample] = []
    for i, doc in enumerate(dataset):
        # Per-example RNG so changing --limit does not reshuffle earlier items.
        rng = random.Random(seed + i)
        choices, gold = preprocess_gpqa_example(doc, rng)
        prompt = build_mcq_prompt(doc["Question"], choices)
        examples.append(
            EvalExample(
                example_id=str(doc.get("Record ID", i)),
                prompt=prompt,
                gold=gold,
                metadata={
                    "num_choices": 4,
                    "subdomain": doc.get("Subdomain"),
                    "domain": doc.get("High-level domain"),
                    "choices": choices,
                },
            )
        )
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples")
    parser.add_argument("--seed", type=int, default=0, help="Seed for shuffling answer choices")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/gpqa_diamond.json"),
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
    examples = build_gpqa_examples(limit=args.limit, seed=args.seed)
    print(f"Loaded {len(examples)} GPQA Diamond examples")

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
        valid_letters=["A", "B", "C", "D"],
        desc="GPQA Diamond",
    )
    summary = summarize_predictions(predictions)
    print(
        f"GPQA Diamond accuracy: {summary['accuracy']:.4f} "
        f"({summary['num_correct']}/{summary['num_examples']}), "
        f"parse_rate={summary['parse_rate']:.4f}"
    )
    save_results(
        args.output,
        benchmark="gpqa_diamond",
        model_name=args.model,
        summary=summary,
        predictions=predictions,
        extra={
            "seed": args.seed,
            "limit": args.limit,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
        },
    )


if __name__ == "__main__":
    main()
