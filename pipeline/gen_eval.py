#!/usr/bin/env python3
"""Sampling/eval harness with per-sample seeds, token counts and finish_reason.

Why not common.generate_responses: it hard-codes seed=42 for every request, so k samples of one
prompt are (near-)identical, and it drops finish_reason / token counts we need for the cap-rate.
Sampling params are otherwise identical to the TA config (temp 1.0, top_p .95, top_k 20, pp 1.5, thinking on).

usage: gen_eval.py --data hmmt_feb_50|aime_2025|hmmt_2025_local --k 4 --port 8000 --out file.jsonl [--keep-text]
"""
import argparse, asyncio, json, sys, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import extract_boxed_answer
from math_grading import is_equiv

ROOT = Path(__file__).resolve().parents[2]
# Exact TA instruction (their sample-math config; the real HMMT prompt is unpublished): includes "keep the reasoning concise".
import yaml
INSTR = yaml.safe_load((ROOT / "ta_eval/CS6013Fall_ProjectEvalMidSemScript/configs/eval_config.yaml").read_text())["math_instruction"]


def load(name):
    from datasets import load_dataset
    if name == "hmmt_feb_50":
        ds = load_dataset(str(ROOT / "ta_eval/CS6013Fall_ProjectEvalMidSemScript/datasets/hmmt_feb_50"), split="train")
        return [dict(id=r["id"], problem=r["problem"], answer=r["answer"]) for r in ds]
    ds = load_dataset(str(ROOT / "CS6013/datasets/math_eval"), split="train")
    src = {"aime_2025": "aime_2025", "hmmt_2025_local": "hmmt_feb_2025"}[name]
    return [dict(id=r["example_id"], problem=r["problem"], answer=r["answer"]) for r in ds if r["source"] == src]


async def one(client, sem, model, row, k, a, f):
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"{INSTR}\n\nProblem:\n{row['problem'].strip()}"}],
                max_tokens=a.max_new, temperature=1.0, top_p=0.95, presence_penalty=1.5,
                seed=zlib.crc32(f"{row['id']}|{k}|{a.seed_base}".encode()),
                extra_body={"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
                            "chat_template_kwargs": {"enable_thinking": True}})
        except Exception as e:
            print("FAIL", row["id"], k, repr(e)[:200], flush=True)
            return
        ch = r.choices[0]
        text = (ch.message.content or "").strip()
        pred = extract_boxed_answer(text)
        rec = dict(id=row["id"], k=k, gold=row["answer"], pred=pred, correct=bool(is_equiv(pred, row["answer"])),
                   ntok=r.usage.completion_tokens, finish=ch.finish_reason, closed_think="</think>" in text)
        if a.keep_text:
            rec["text"] = text
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", type=int, default=96)
    ap.add_argument("--max-new", type=int, default=32000)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--keep-text", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    a = ap.parse_args()
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{a.port}/v1", api_key="EMPTY", timeout=7200)
    rows = load(a.data)
    sem = asyncio.Semaphore(a.conc)
    with open(a.out, "a") as f:
        await asyncio.gather(*[one(client, sem, a.model, r, k, a, f) for r in rows for k in range(a.k)])


if __name__ == "__main__":
    asyncio.run(main())
