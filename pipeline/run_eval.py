"""Evaluate Qwen3.5-4B on the GPQA Diamond dataset using the vLLM API server.

End-to-end flow
----------------
1. `parse_args` reads a single required CLI argument, `--config`, pointing
   to an experiment YAML file.
2. `load_config` loads that YAML file into a plain `dict` (`cfg`).
3. Loads the GPQA Diamond dataset via `load_gpqa_diamond_examples`, using
   `cfg["dataset_path"]` as the dataset name/path and `cfg["limit"]` to
   optionally cap the number of examples (e.g. for a quick smoke test).
4. Runs the evaluation via `run_math_eval`, forwarding the model name and
   all sampling/decoding parameters from the config: `max_new_tokens`,
   `temperature`, `top_p`, `top_k`, and the optional `min_p`,
   `presence_penalty`, `repetition_penalty` (each with a fallback default
   if absent from the config), plus `enable_thinking` mode.

Usage
-----
    python run_eval.py --config path/to/config.yaml

Concurrency
-----------
Generation against the vLLM server is concurrent, not one-request-at-a-time:
`common.run_math_eval` issues up to `cfg["max_concurrency"]` chat-completion
requests in flight at once (default 8 if not set in the config), which lets
vLLM's continuous batching actually batch multiple sequences together
instead of processing a single request, waiting for it, then sending the
next one. Because the whole request pipeline is now `async`, `main()` is a
coroutine and is driven via `asyncio.run(main())` at the bottom of this
file, and the OpenAI client used is `AsyncOpenAI` rather than `OpenAI`.
Raise `max_concurrency` in the config to push more sequences through the
server at once (bounded by the server's own batch/KV-cache limits); lower
it if you see server-side OOMs or excessive queuing.

"""
from __future__ import annotations

import argparse
import asyncio
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    load_gpqa_diamond_examples,
    run_math_eval,
    save_results,
    summarize_predictions,
)


def load_config(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Qwen3.5-4B on GPQA Diamond."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML config.",
    )

    return parser.parse_args()


async def main():

    args = parse_args()

    cfg = load_config(args.config)

    max_concurrency = cfg.get("max_concurrency", 8)

    print("=" * 80, flush=True)
    print("GPQA Diamond vLLM API Evaluation", flush=True)
    print("=" * 80, flush=True)

    print(f"Experiment:      {cfg['exp_name']}", flush=True)
    print(f"Model:           {cfg['model_name']}", flush=True)
    print(f"Server:          {cfg['vllm_base_url']}", flush=True)
    print(f"Dataset:         {cfg['dataset_path']}", flush=True)
    print(f"Max concurrency: {max_concurrency}", flush=True)
    print(flush=True)

    # --------------------------------------------------------
    # Check vLLM API server
    # --------------------------------------------------------

    print("=" * 80, flush=True)
    print("Connecting to vLLM API server", flush=True)
    print("=" * 80, flush=True)

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=cfg["vllm_base_url"],
        api_key=cfg.get("api_key", "EMPTY"),
        timeout=cfg["request_timeout"],
    )

    models = await client.models.list()

    print("vLLM API server is available.", flush=True)

    print("Available models:", flush=True)
    for model in models.data:
        print(f"  - {model.id}", flush=True)

    print(flush=True)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    examples = load_gpqa_diamond_examples(
        dataset_path=cfg["dataset_path"],
        math_instruction=cfg["math_instruction"],
        limit=cfg["limit"],
    )

    print(
        f"Loaded {len(examples)} GPQA Diamond examples.",
        flush=True,
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    predictions = await run_math_eval(
        examples,
        client,
        model_name=cfg["model_name"],
        max_new_tokens=cfg["max_new_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        min_p=cfg.get("min_p", 0.0),
        presence_penalty=cfg.get("presence_penalty", 1.5),
        repetition_penalty=cfg.get("repetition_penalty", 1.0),
        enable_thinking=cfg["enable_thinking"],
        max_concurrency=max_concurrency,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    summary = summarize_predictions(
        predictions
    )

    print(flush=True)
    print("=" * 80, flush=True)
    print("RESULTS", flush=True)
    print("=" * 80, flush=True)

    print(
        f"Accuracy:   {summary['accuracy']:.4f} "
        f"({summary['num_correct']}/"
        f"{summary['num_examples']})",
        flush=True,
    )

    print(
        f"Parse rate: {summary['parse_rate']:.4f} "
        f"({summary['num_parsed']}/"
        f"{summary['num_examples']})",
        flush=True,
    )

    print("=" * 80, flush=True)

    save_results(
        cfg["output"],
        benchmark="gpqa_diamond",
        model_name=cfg["model_name"],
        summary=summary,
        predictions=predictions,
        extra=cfg,
    )


if __name__ == "__main__":
    asyncio.run(main())