# Evaluation scripts (GPQA Diamond + MMLU-Pro)

Zero-shot generative evaluation of Hugging Face instruction models with a
clear prompt that asks for a boxed choice letter, e.g. `\boxed{A}`.

## Default model

`Qwen/Qwen3-4B-Instruct-2507`

## Datasets

| Benchmark | Dataset | Subset / split |
| --- | --- | --- |
| GPQA Diamond | `Idavidrein/gpqa` | `gpqa_diamond` / `train` (198 Qs) |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | `test` (default) |

### GPQA Diamond category distribution

**High-level domain** (198 total)

| Category | Count |
| --- | ---: |
| Chemistry | 93 |
| Physics | 86 |
| Biology | 19 |

**Subdomain**

| Subdomain | Count |
| --- | ---: |
| Organic Chemistry | 72 |
| Quantum Mechanics | 25 |
| Chemistry (general) | 20 |
| Physics (general) | 19 |
| Molecular Biology | 15 |
| High-energy particle physics | 14 |
| Astrophysics | 13 |
| Relativistic Mechanics | 7 |
| Electromagnetism and Photonics | 6 |
| Genetics | 4 |
| Inorganic Chemistry | 1 |
| Optics and Acoustics | 1 |
| Condensed Matter Physics | 1 |

### MMLU-Pro category distribution

**Test split** (12,032 total; default eval split)

| Category | Count |
| --- | ---: |
| math | 1351 |
| physics | 1299 |
| chemistry | 1132 |
| law | 1101 |
| engineering | 969 |
| other | 924 |
| economics | 844 |
| health | 818 |
| psychology | 798 |
| business | 789 |
| biology | 717 |
| philosophy | 499 |
| computer science | 410 |
| history | 381 |

**Validation split** (70 total): 5 examples per category across the same 14 categories.

## Prompt format

Each question is sent through the model's chat template as:

```text
Answer the following multiple-choice question. Reason carefully, but keep
the reasoning concise. After reasoning, output the final answer on its own
last line using exactly this format: \boxed{X} where X is one of (A, B, C, D).
Do not write anything after that line. Example last line: \boxed{A}.

Question:
...

Choices:
A. ...
B. ...
...
```

Answers are extracted preferentially from `\boxed{X}`, with light fallbacks if
the model does not follow the format.

## Usage

From the project root (with the project venv activated):

```bash
# Quick smoke test (2 examples each)
python evaluation/run_eval.py --limit 2 --batch-size 1

# Full GPQA Diamond
python evaluation/evaluate_gpqa.py

# Full MMLU-Pro (slow; ~12k questions)
python evaluation/evaluate_mmlu_pro.py

# MMLU-Pro subset / category filter
python evaluation/evaluate_mmlu_pro.py --limit 100 --categories math physics

# Custom model
python evaluation/run_eval.py --model Qwen/Qwen3-4B-Instruct-2507 --benchmarks gpqa

# Enable transformers/datasets/HF library logging (off by default)
python evaluation/evaluate_gpqa.py --verbose

# Evaluate a compression submission (compress -> restore -> 10-sample eval)
python evaluation/eval_submission.py \
  --submission sample_submission/code.py \
  --model Qwen/Qwen3-4B-Instruct-2507
```

Library logging (transformers, datasets, Hugging Face Hub warnings) is **off by
default**. Pass `--verbose` to turn it on. Eval progress bars (`tqdm`) stay enabled.

Results are written under `evaluation/results/` as JSON with overall accuracy,
parse rate, and per-example predictions.

Default generation length is `--max-new-tokens 4096` so the model has room to
finish reasoning and emit `\boxed{X}`.
