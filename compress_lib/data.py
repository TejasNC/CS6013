"""Teacher-trace data: splits (calibration/KD vs held-out KL slice), tokenization.

Leak rule: a sample (id,k) goes to the KL slice iff crc32(f"{id}|{k}") % 5 == 0; those samples are NEVER used for
Hessians, Fisher, activity scores, allocation fits or KD. AIME is never loaded here.
"""
import json, zlib
from pathlib import Path
import torch

RUNS = Path("/mnt/nas/tejaschaudhari/emir_data/runs")
INSTR = ("Solve the following math problem. Reason step by step, and put your final\n"
         "answer within \\boxed{}. Write only the final answer inside the box.")


def is_heldout(rec) -> bool:
    return zlib.crc32(f"{rec['id']}|{rec['k']}|{rec.get('src','')}".encode()) % 5 == 0


def load_records(files=None):
    files = files or sorted(RUNS.glob("teacher_*.jsonl"))
    recs = []
    for f in files:
        src = Path(f).stem
        for l in open(f):
            r = json.loads(l); r["src"] = src
            if "text" in r and r["finish"] == "stop" and r["closed_think"]:
                recs.append(r)
    return recs


def problems():
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
    from gen_eval import load
    d = {}
    for name in ("hmmt_feb_50", "hmmt_2025_local"):  # HMMT only; AIME deliberately absent
        for r in load(name):
            d[(name_to_src(name), r["id"])] = r["problem"]
    return d


def name_to_src(name):
    return {"hmmt_feb_50": "teacher_hmmt_feb_50", "hmmt_2025_local": "teacher_hmmt_2025_local"}[name]


def tokenize(rec, tok, probs, max_len):
    msg = [{"role": "user", "content": f"{INSTR}\n\nProblem:\n{probs[(rec['src'], rec['id'])].strip()}"}]
    p = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(p, add_special_tokens=False).input_ids
    rid = tok(rec["text"], add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
    ids = (pid + rid)[:max_len]
    return torch.tensor(ids), len(pid)


def build(split: str, tok, max_len: int, max_seqs: int | None = None, correct_only: bool = False):
    probs = problems(); recs = load_records()
    recs = [r for r in recs if is_heldout(r) == (split == "kl")]
    if correct_only: recs = [r for r in recs if r["correct"]]
    recs.sort(key=lambda r: (r["id"], r["k"]))
    out = [tokenize(r, tok, probs, max_len) for r in recs[:max_seqs]]
    return out, recs[:max_seqs]
