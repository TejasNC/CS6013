"""Math eval (HMMT Feb 2025 + AIME 2025) against a vLLM server.

Ours. Reuses the TA harness' generation path unchanged: `common.generate_responses`
(same sampling params, `enable_thinking`, seed handling, concurrency) and
`common.extract_boxed_answer`. Only the dataset and the grading differ, because
`common.py` hard-codes A-D multiple choice. `common.py` is NOT modified.

Reports, per source and overall:
  - mean accuracy over all (problem, sample) pairs  (= expected pass@1)
  - parse rate (a \\boxed{} answer was found)
  - truncation proxy: responses without any \\boxed answer

Usage
-----
    python pipeline/run_math_eval.py --config configs/eval_math.yaml
    python pipeline/run_math_eval.py --config configs/eval_math.yaml --samples 4 --sources aime_2025
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import extract_boxed_answer, generate_responses  # noqa: E402  (TA code, unmodified)
from math_grading import is_equiv  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=None, help="samples per problem (overrides config)")
    ap.add_argument("--sources", nargs="*", default=None, help="subset of: hmmt_feb_2025 aime_2025")
    ap.add_argument("--limit", type=int, default=None, help="cap problems per source (smoke test)")
    ap.add_argument("--output", type=Path, default=None, help="results JSON (overrides config)")
    return ap.parse_args()


async def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    from datasets import load_dataset
    from openai import AsyncOpenAI

    dpath = Path(cfg["dataset_path"])
    if (dpath / "dataset_dict.json").is_file():
        # TA-style `save_to_disk` dataset (e.g. the hmmt_feb_50 folder from the TA repo).
        from datasets import load_from_disk

        ds = load_from_disk(str(dpath))["train"]
    else:
        ds = load_dataset(str(dpath), split="train")
    rows = []
    for i, r in enumerate(ds):
        r = dict(r)
        r.setdefault("source", dpath.name)
        r.setdefault("example_id", f"{r['source']}-{r.get('id', r.get('problem_idx', i))}")
        r["answer"] = str(r["answer"])
        rows.append(r)
    if args.sources:
        rows = [r for r in rows if r["source"] in args.sources]
    limit = args.limit if args.limit is not None else cfg.get("limit")
    if limit:
        seen: dict[str, int] = defaultdict(int)
        kept = []
        for r in rows:
            if seen[r["source"]] < limit:
                kept.append(r)
                seen[r["source"]] += 1
        rows = kept

    n_samples = args.samples or cfg.get("samples", 1)
    instruction = cfg["math_instruction"].strip()
    jobs = [
        (r, k)
        for r in rows
        for k in range(n_samples)
    ]
    prompts = [f"{instruction}\n\nProblem:\n{r['problem'].strip()}" for r, _ in jobs]
    print(f"{len(rows)} problems x {n_samples} samples = {len(prompts)} generations", flush=True)

    client = AsyncOpenAI(
        base_url=cfg["vllm_base_url"],
        api_key=cfg.get("api_key", "EMPTY"),
        timeout=cfg["request_timeout"],
    )

    responses = await generate_responses(
        client,
        cfg["model_name"],
        prompts,
        max_new_tokens=cfg["max_new_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        min_p=cfg.get("min_p", 0.0),
        presence_penalty=cfg.get("presence_penalty", 1.5),
        repetition_penalty=cfg.get("repetition_penalty", 1.0),
        enable_thinking=cfg["enable_thinking"],
        max_concurrency=cfg.get("max_concurrency", 8),
    )

    records = []
    for (r, k), resp in zip(jobs, responses):
        pred = extract_boxed_answer(resp)
        records.append(
            {
                "example_id": r["example_id"],
                "source": r["source"],
                "sample": k,
                "gold": r["answer"],
                "prediction": pred,
                "correct": is_equiv(pred, r["answer"]),
                "response": resp,
            }
        )

    def summarize(rs: list[dict]) -> dict:
        n = len(rs)
        return {
            "n": n,
            "accuracy": sum(x["correct"] for x in rs) / n if n else 0.0,
            "parse_rate": sum(x["prediction"] is not None for x in rs) / n if n else 0.0,
        }

    by_source = defaultdict(list)
    for x in records:
        by_source[x["source"]].append(x)
    summary = {"overall": summarize(records), **{s: summarize(v) for s, v in by_source.items()}}

    print("\n" + "=" * 60 + "\nRESULTS (mean over samples)\n" + "=" * 60)
    for name, s in summary.items():
        print(f"{name:>16}: acc {s['accuracy']:.4f}  parse {s['parse_rate']:.4f}  (n={s['n']})")

    out = args.output or Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "config": cfg, "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    asyncio.run(main())
