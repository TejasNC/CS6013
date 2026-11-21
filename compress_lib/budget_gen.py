"""Budget-forced short traces for length shaping.

Take teacher traces (TA prompt, any outcome), cut the thinking at a sentence boundary near a token budget, append the wrap-up phrase and
</think>, let the TEACHER (bf16, vLLM) write the final answer, keep answers that match the ground truth.

  python -m compress_lib.budget_gen --port 8000 --out budget_traces.jsonl [--cuts 4096,6144,8192,12288,16384] [--dry]

Output record: id, src, k, cut (requested budget), nprefix (tokens in the kept thinking prefix), prefix, answer, correct, wrap.
Only local-2025 problems that are NOT the three validation problems (kd.py val split) are used. No hmmt_feb_50, no AIME.
"""
import argparse, asyncio, json, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from common import extract_boxed_answer  # noqa: E402
from math_grading import is_equiv  # noqa: E402
from .data import RUNS, problems, _instr  # noqa: E402

WRAP = "Considering the limited time, I have to give the solution based on the thinking directly now.\n</think>\n\n"


def val_problem_set(recs):
    allp = sorted({(r["src"], r["id"]) for r in recs if not r["src"].startswith("teacher_old_")})
    random.Random(1234).shuffle(allp)
    return set(allp[:3])  # identical to kd.py's validation holdout


def cut_at_sentence(tok, ids, cut):
    text = tok.decode(ids[:cut])
    m = list(re.finditer(r"(?:[.!?]\)?\$?|\n\n)(?=\s)", text))
    if not m:
        return None
    ABBR = re.compile(r"(?:\be\.g|\bi\.e|\betc|\bvs|\bFig|\bapprox|\bNo|\bcf|\bresp|\bDr|\bMr)\.$")
    for mm in reversed(m):  # last boundary that does not follow an abbreviation (e.g., i.e., etc., vs., ...)
        end = mm.end()
        if end < 0.8 * len(text):  # no boundary near the cut: skip rather than cut mid-thought
            return None
        if not ABBR.search(text[:end].rstrip()):
            return text[:end].rstrip()
    return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=None); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", required=True); ap.add_argument("--cuts", default="4096,6144,8192,12288,16384")
    ap.add_argument("--conc", type=int, default=64); ap.add_argument("--dry", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    M = a.teacher or open(Path(__file__).resolve().parents[2] / "scratch_model_path.txt").read().strip()
    tok = AutoTokenizer.from_pretrained(M)
    probs = problems()
    recs = []
    for l in open(RUNS / "teacher_hmmt_2025_local.jsonl"):
        r = json.loads(l); r["src"] = "teacher_hmmt_2025_local"
        if "text" in r: recs.append(r)
    val = val_problem_set(recs)
    cuts = [int(x) for x in a.cuts.split(",")]
    jobs = []
    for r in recs:
        if (r["src"], r["id"]) in val: continue
        ids = tok(r["text"], add_special_tokens=False).input_ids
        think_ids = ids  # the trace text starts right after "<think>\n"; any "</think>" inside means the model had finished thinking
        if "</think>" in r["text"]:
            think_ids = tok(r["text"].split("</think>")[0], add_special_tokens=False).input_ids
        for cut in cuts:
            if len(think_ids) < cut + 300: continue  # the trace must still be thinking at the cut
            prefix = cut_at_sentence(tok, think_ids, cut)
            if prefix is None: continue
            jobs.append(dict(id=r["id"], src=r["src"], k=r["k"], cut=cut, prefix=prefix, gold=r["gold"]))
    by_cut = {c: sum(j["cut"] == c for j in jobs) for c in cuts}
    print("candidate jobs", len(jobs), "per cut", by_cut, flush=True)
    if a.dry: return
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{a.port}/v1", api_key="EMPTY", timeout=3600)
    sem = asyncio.Semaphore(a.conc); fout = open(a.out, "a")

    async def one(j):
        msg = [{"role": "user", "content": f"{_instr()}\n\nProblem:\n{probs[(j['src'], j['id'])].strip()}"}]
        p = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        prompt = p + j["prefix"] + "\n\n" + WRAP
        async with sem:
            try:
                x = await client.completions.create(model="Qwen/Qwen3.5-4B", prompt=prompt, max_tokens=2500, temperature=1.0, top_p=0.95,
                                                    presence_penalty=1.5, seed=random.Random(f"{j['id']}{j['k']}{j['cut']}").randrange(2**31),
                                                    extra_body={"top_k": 20, "min_p": 0.0})
            except Exception as e:
                print("FAIL", j["id"], j["cut"], repr(e)[:150], flush=True); return
        ans = x.choices[0].text
        pred = extract_boxed_answer(ans)
        rec = dict(id=j["id"], src=j["src"], k=j["k"], cut=j["cut"], nprefix=len(tok(j["prefix"], add_special_tokens=False).input_ids),
                   prefix=j["prefix"], answer=ans.strip(), pred=pred, correct=bool(is_equiv(pred, j["gold"])), finish=x.choices[0].finish_reason)
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()

    await asyncio.gather(*[one(j) for j in jobs])


if __name__ == "__main__":
    asyncio.run(main())
