"""Evaluate a Qwen3.5-4B on GPQA Diamond using the vLLM API server.

End-to-end flow
----------------
1. `load_gpqa_diamond_examples` pulls a dataset (via `datasets.load_dataset`)
    and converts each row into an `EvalExample`(a prompt built from a fixed 
    instruction template -- see `math_instruction` in the experiment config.
2. `run_math_eval` drives the evaluation:
   - `generate_responses` sends each prompt to the vLLM server as a
     single-turn chat completion, with up to `max_concurrency` requests in
     flight at once (configurable sampling parameters: temperature, top_p,
     top_k, min_p, repetition/presence penalties, and an optional
     "enable_thinking" chat-template flag), and collects the raw model
     text.
   - For each response, `extract_boxed_answer` pulls out the model's final
     answer from a LaTeX `\\boxed{...}` expression (scanning for the last
     one, and handling nested braces).
   - `answers_match` compares the extracted prediction against the gold
     letter: both are run through `normalize_answer`, which strips
     surrounding whitespace/`$`/LaTeX wrapping (e.g. "$A$", "(A)", "Option
     A") down to a bare uppercase letter, and the two are then compared
     directly for equality -- there is no symbolic/mathematical
     equivalence step, since the answer space is just the four discrete
     choices A/B/C/D.
   - Each example's prediction, normalized/parsed forms, gold answer, and
     correctness are collected into an `EvalPrediction` record, with verbose
     per-example logging to stdout (flushed immediately so nothing is lost
     if stdout is redirected to a file, e.g. under SLURM).

"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import traceback
import warnings
import torch

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from openai import AsyncOpenAI
from datasets import load_dataset


# The task is multiple-choice: the model is instructed (see
# `math_instruction` in the experiment config) to give its final answer as
# \boxed{X} where X is exactly one of these four letters.
_VALID_CHOICES = ("A", "B", "C", "D")


@dataclass
class EvalExample:
    example_id: str
    prompt: str
    gold: str
    metadata: dict[str, Any]


@dataclass
class EvalPrediction:
    example_id: str
    gold: str
    prediction: str | None
    prediction_normalized: str | None
    prediction_parsed: list[str]
    gold_normalized: str
    gold_parsed: list[str]
    correct: bool
    response: str
    metadata: dict[str, Any]


def normalize_answer(answer: str) -> str:

    text = answer.strip()

    # Remove surrounding $ delimiters some models still add out of habit.
    text = text.strip("$").strip()

    # Strip LaTeX escape/grouping characters that might wrap a bare letter,
    # e.g. "\(A\)" or "{A}".
    text = re.sub(r"[\\{}()]", "", text)

    text = text.strip()

    # Prefer a standalone A-D token (so "Option A" or "A." resolve to "A"
    # rather than accidentally matching a letter inside another word).
    match = re.search(r"\b([A-Da-d])\b", text)

    if match:
        return match.group(1).upper()

    # Fall back to the first A-D letter appearing anywhere in the text.
    for char in text.upper():
        if char in _VALID_CHOICES:
            return char

    return ""


def parse_answer(
    answer: str | None,
) -> tuple[str | None, list[str]]:

    if answer is None:
        return None, []

    normalized = normalize_answer(answer)

    if normalized not in _VALID_CHOICES:
        return normalized, []

    return normalized, [normalized]


def build_math_prompt(
    problem: str,
    math_instruction: str,
) -> str:

    return (
        f"{math_instruction}\n\n"
        f"Problem:\n"
        f"{problem.strip()}"
    )


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} answer."""

    i = text.rfind(r"\boxed")

    if i < 0:
        return None

    start = text.find("{", i)

    if start < 0:
        return None

    depth = 0

    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1

            if depth == 0:
                return text[start + 1:j].strip()

    return None


def answers_match(
    prediction: str | None,
    gold: str,
) -> tuple[
    bool,
    str | None,
    list[str],
    str,
    list[str],
]:

    gold_normalized = normalize_answer(gold)

    gold_parsed = (
        [gold_normalized]
        if gold_normalized in _VALID_CHOICES
        else []
    )

    if prediction is None:
        return (
            False,
            None,
            [],
            gold_normalized,
            gold_parsed,
        )

    prediction_normalized = normalize_answer(prediction)

    prediction_parsed = (
        [prediction_normalized]
        if prediction_normalized in _VALID_CHOICES
        else []
    )

    correct = (
        prediction_normalized in _VALID_CHOICES
        and prediction_normalized == gold_normalized
    )

    return (
        correct,
        prediction_normalized or None,
        prediction_parsed,
        gold_normalized,
        gold_parsed,
    )


def get_torch_dtype(dtype: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    return mapping[dtype]


def build_chat_messages(
    prompt: str,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": prompt,
        }
    ]


async def _generate_one_response(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
    prompt: str,
    *,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float | None,
    presence_penalty: float | None,
    repetition_penalty: float | None,
    enable_thinking: bool,
) -> str:

    async with semaphore:

        # NOTE: flush=True is required on every print here. When stdout is
        # redirected to a file (e.g. under SLURM), Python switches from
        # line-buffering to full block-buffering, so unflushed prints can
        # sit invisible in a buffer for a long time. Because many of these
        # tasks run concurrently, lines from different requests may
        # interleave in the log -- that's expected.
        print(flush=True)
        print("=" * 80, flush=True)
        print(f"Generating response {index}/{total}", flush=True)
        print("=" * 80, flush=True)

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                seed=42,
                extra_body={
                    "top_k": top_k,
                    "min_p": min_p,
                    "repetition_penalty": repetition_penalty,
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking,
                    },
                },
            )
        except Exception:
            print(
                f"FAILED to generate response {index}/{total}",
                flush=True,
            )
            traceback.print_exc()
            raise

        text = response.choices[0].message.content

        if text is None:
            text = ""

        text = text.strip()

        print(flush=True)
        print(f"Model Response {index}/{total}:", flush=True)
        print(text, flush=True)
        print(flush=True)

        return text


async def generate_responses(
    client: AsyncOpenAI,
    model_name: str,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float | None,
    presence_penalty: float | None,
    repetition_penalty: float | None,
    enable_thinking: bool,
    max_concurrency: int = 8,
) -> list[str]:

    total = len(prompts)

    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        _generate_one_response(
            client,
            semaphore,
            i,
            total,
            prompt,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            enable_thinking=enable_thinking,
        )
        for i, prompt in enumerate(prompts, start=1)
    ]

    responses = await asyncio.gather(*tasks)

    return list(responses)

def load_gpqa_diamond_examples(
    math_instruction: str,
    dataset_path: str,
    limit: int | None = None,
    seed: int = 42,
) -> list[EvalExample]:
 
    print(
        f"Loading dataset: {dataset_path} ",
        flush=True,
    )

    dataset = load_dataset(
        dataset_path,
        split="train",
    )

    print(
        f"Dataset size: {len(dataset)}",
        flush=True,
    )

    if limit is not None:
        dataset = dataset.select(
            range(min(limit, len(dataset)))
        )

        print(
            f"Using first {len(dataset)} examples.",
            flush=True,
        )

    examples = []

    for row_idx, row in enumerate(dataset):

        # NOTE: wrapped in try/except so that a missing/renamed column
        # (e.g. GPQA's exact column names changing between dataset
        # revisions) raises a clear, immediately-flushed error instead of
        # silently killing the process with the traceback only visible in
        # a separate .err log file.
        try:
            question = row["Question"]

            # Index 0 is always the correct answer here, before shuffling.
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]

            # Deterministic per-row shuffle: seeding by (seed, row_idx)
            # rather than advancing one shared Random instance across the
            # loop means each row's shuffle only depends on its own
            # content/position, not on how many rows came before it.
            rng = random.Random(f"{seed}:{row_idx}")

            original_indices = list(range(len(choices)))
            rng.shuffle(original_indices)

            shuffled_choices = [
                choices[i]
                for i in original_indices
            ]

            # The correct answer started at original index 0; find which
            # shuffled slot it landed in to get the gold letter.
            correct_position = original_indices.index(0)
            gold_letter = "ABCD"[correct_position]

            options_block = "\n".join(
                f"{letter}) {choice.strip()}"
                for letter, choice in zip(
                    "ABCD",
                    shuffled_choices,
                )
            )

            problem = (
                f"{question.strip()}\n\n"
                f"{options_block}"
            )

            examples.append(
                EvalExample(
                    example_id=str(row_idx),
                    prompt=build_math_prompt(
                        problem,
                        math_instruction,
                    ),
                    gold=gold_letter,
                    metadata={
                        "problem_type": row.get(
                            "Subdomain",
                            None,
                        ),
                    },
                )
            )

        except Exception:
            print(
                f"FAILED to parse row {row_idx}: {row!r}",
                flush=True,
            )
            traceback.print_exc()
            raise

    print(
        f"Created {len(examples)} EvalExample objects.",
        flush=True,
    )

    return examples


async def run_math_eval(
    examples: Sequence[EvalExample],
    client: AsyncOpenAI,
    *,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    enable_thinking: bool,
    max_concurrency: int = 1,
) -> list[EvalPrediction]:

    # --------------------------------------------------------
    # Generate model responses
    # --------------------------------------------------------

    prompts = [
        example.prompt
        for example in examples
    ]

    responses = await generate_responses(
        client,
        model_name,
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        enable_thinking=enable_thinking,
        max_concurrency=max_concurrency,
    )

    # --------------------------------------------------------
    # Evaluate responses
    # --------------------------------------------------------

    predictions: list[EvalPrediction] = []

    for example, response in zip(
        examples,
        responses,
    ):

        # Extract the final answer from \boxed{...}
        prediction = extract_boxed_answer(
            response
        )

        # Normalize, parse, and compare prediction vs gold.
        (
            correct,
            prediction_normalized,
            prediction_parsed,
            gold_normalized,
            gold_parsed,
        ) = answers_match(
            prediction=prediction,
            gold=example.gold,
        )

        # ----------------------------------------------------
        # Print detailed evaluation information
        # ----------------------------------------------------

        print(flush=True)
        print("=" * 80, flush=True)
        print(
            f"Example {example.example_id}",
            flush=True,
        )

        print(
            f"Prediction: "
            f"{prediction!r}",
            flush=True,
        )

        print(
            f"Gold: "
            f"{example.gold!r}",
            flush=True,
        )

        print(
            f"Prediction normalized: "
            f"{prediction_normalized!r}",
            flush=True,
        )

        print(
            f"Gold normalized: "
            f"{gold_normalized!r}",
            flush=True,
        )

        print(
            f"Prediction parsed: "
            f"{prediction_parsed!r}",
            flush=True,
        )

        print(
            f"Gold parsed: "
            f"{gold_parsed!r}",
            flush=True,
        )

        print(
            f"Correct: "
            f"{correct}",
            flush=True,
        )

        print("=" * 80, flush=True)

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        predictions.append(
            EvalPrediction(
                example_id=example.example_id,
                gold=example.gold,
                prediction=prediction,
                prediction_normalized=prediction_normalized,
                prediction_parsed=prediction_parsed,
                gold_normalized=gold_normalized,
                gold_parsed=gold_parsed,
                correct=correct,
                response=response,
                metadata=example.metadata,
            )
        )

    return predictions


def summarize_predictions(
    predictions: Sequence[EvalPrediction],
) -> dict[str, Any]:

    n = len(predictions)

    n_correct = sum(1 for prediction in predictions if prediction.correct)

    n_parsed = sum(1 for prediction in predictions if prediction.prediction is not None)

    return {
        "num_examples": n,
        "num_correct": n_correct,
        "accuracy": (n_correct / n if n else 0.0),
        "num_parsed": n_parsed,
        "parse_rate": (n_parsed / n if n else 0.0),
    }


def save_results(
    output_path: str | Path,
    *,
    benchmark: str,
    model_name: str,
    summary: dict[str, Any],
    predictions: Sequence[EvalPrediction],
    extra: dict[str, Any] | None = None,
) -> None:
    """Save evaluation results as JSON."""

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "benchmark": benchmark,
        "model_name": model_name,
        "summary": summary,
        "extra": extra or {},
        "predictions": [asdict(prediction) for prediction in predictions],
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Wrote results to {path}", flush=True)