"""Shared utilities for Hugging Face instruction-model evaluation."""

from __future__ import annotations

import json
import logging
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

BOXED_INSTRUCTION = (
    "Answer the following multiple-choice question. Reason carefully, but keep "
    "the reasoning concise. After reasoning, output the final answer on its own "
    "last line using exactly this format: \\boxed{{X}} where X is one of "
    "({letters}). Do not write anything after that line. Example last line: "
    "\\boxed{{A}}."
)

BOXED_RE = re.compile(r"\\boxed\s*\{\s*([A-Za-z])\s*\}")
STRICT_ANSWER_LINE_RE = re.compile(
    r"^\s*(?:final\s+answer|answer)\s*(?:is|:)\s*\(?([A-Za-z])\)?\s*\.?\s*$",
    re.IGNORECASE,
)
LETTER_ONLY_RE = re.compile(r"^\s*\(?([A-Za-z])\)?\s*$")


def configure_logging(verbose: bool = False) -> None:
    """Silence library logs by default; enable with --verbose."""
    if verbose:
        level = logging.INFO
    else:
        level = logging.ERROR
        warnings.filterwarnings("ignore")

    logging.basicConfig(level=level)
    for name in (
        "transformers",
        "datasets",
        "huggingface_hub",
        "httpx",
        "httpcore",
        "torch",
        "accelerate",
    ):
        logging.getLogger(name).setLevel(level)

    try:
        from transformers.utils import logging as transformers_logging

        if verbose:
            transformers_logging.set_verbosity_info()
            transformers_logging.enable_progress_bar()
        else:
            transformers_logging.set_verbosity_error()
            transformers_logging.disable_progress_bar()
    except Exception:
        pass

    try:
        from datasets.utils import logging as datasets_logging

        if verbose:
            datasets_logging.set_verbosity_info()
            datasets_logging.enable_progress_bar()
        else:
            datasets_logging.set_verbosity_error()
            datasets_logging.disable_progress_bar()
    except Exception:
        pass


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
    correct: bool
    response: str
    metadata: dict[str, Any]


def format_choices(choices: Sequence[str]) -> str:
    lines = []
    for i, choice in enumerate(choices):
        letter = chr(ord("A") + i)
        lines.append(f"{letter}. {choice.strip()}")
    return "\n".join(lines)


def build_mcq_prompt(question: str, choices: Sequence[str]) -> str:
    letters = ", ".join(chr(ord("A") + i) for i in range(len(choices)))
    return (
        f"{BOXED_INSTRUCTION.format(letters=letters)}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Choices:\n{format_choices(choices)}\n"
    )


def extract_boxed_letter(text: str, valid_letters: Sequence[str] | None = None) -> str | None:
    """Extract the choice letter from a model response, preferring \\boxed{}."""

    def _valid(letter: str) -> str | None:
        letter = letter.upper()
        if valid_letters is None or letter in valid_letters:
            return letter
        return None

    matches = BOXED_RE.findall(text)
    if matches:
        letter = _valid(matches[-1])
        if letter is not None:
            return letter

    # Only inspect the last few lines so mid-reasoning phrases are ignored.
    tail_lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()][-5:]
    for line in reversed(tail_lines):
        boxed = BOXED_RE.search(line)
        if boxed:
            letter = _valid(boxed.group(1))
            if letter is not None:
                return letter
        m = STRICT_ANSWER_LINE_RE.match(line)
        if m:
            letter = _valid(m.group(1))
            if letter is not None:
                return letter
        m = LETTER_ONLY_RE.match(line)
        if m:
            letter = _valid(m.group(1))
            if letter is not None:
                return letter
    return None


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    batch_size: int = 4,
    desc: str = "Generating",
) -> list[str]:
    responses: list[str] = []
    do_sample = temperature > 0
    pbar = tqdm(total=len(prompts), desc=desc, unit="ex")

    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            chat_texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in batch_prompts
            ]
            inputs = tokenizer(
                chat_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature

            outputs = model.generate(**inputs, **gen_kwargs)
            prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
            for i, output in enumerate(outputs):
                generated = output[int(prompt_lens[i]) :]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                responses.append(text)
            pbar.update(len(batch_prompts))
    finally:
        pbar.close()

    return responses


def run_multiple_choice_eval(
    examples: Sequence[EvalExample],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    batch_size: int = 4,
    valid_letters: Sequence[str] | None = None,
    desc: str = "Evaluating",
) -> list[EvalPrediction]:
    prompts = [ex.prompt for ex in examples]
    responses = generate_responses(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
        desc=desc,
    )

    predictions: list[EvalPrediction] = []
    for ex, response in zip(examples, responses):
        letters = valid_letters
        if letters is None:
            # Infer from metadata when present.
            n_choices = ex.metadata.get("num_choices")
            if n_choices is not None:
                letters = [chr(ord("A") + i) for i in range(int(n_choices))]
        pred = extract_boxed_letter(response, letters)
        predictions.append(
            EvalPrediction(
                example_id=ex.example_id,
                gold=ex.gold,
                prediction=pred,
                correct=pred is not None and pred.upper() == ex.gold.upper(),
                response=response,
                metadata=ex.metadata,
            )
        )
    return predictions


def summarize_predictions(predictions: Iterable[EvalPrediction]) -> dict[str, Any]:
    preds = list(predictions)
    n = len(preds)
    n_correct = sum(1 for p in preds if p.correct)
    n_parsed = sum(1 for p in preds if p.prediction is not None)
    return {
        "num_examples": n,
        "num_correct": n_correct,
        "accuracy": (n_correct / n) if n else 0.0,
        "num_parsed": n_parsed,
        "parse_rate": (n_parsed / n) if n else 0.0,
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
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": benchmark,
        "model_name": model_name,
        "summary": summary,
        "extra": extra or {},
        "predictions": [asdict(p) for p in predictions],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote results to {path}")
