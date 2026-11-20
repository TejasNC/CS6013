"""Teacher-trace data: splits (calibration/KD vs held-out KL slice), tokenization.

Leak rule: see is_heldout(); KL-slice samples are NEVER used for
Hessians, Fisher, activity scores, allocation fits or KD. AIME is never loaded here.
"""
import json, zlib
from pathlib import Path
import torch

RUNS = Path("/mnt/nas/tejaschaudhari/emir_data/runs")
def _instr():
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
    from gen_eval import INSTR
    return INSTR


def is_heldout(rec) -> bool:
    """KL slice = every hmmt_feb_50 sample (the official eval problems, never trained/calibrated on) plus ~20% of
    local HMMT-2025 samples. Calibration / Hessians / Fisher / KD use ONLY the remaining local-2025 samples."""
    if "feb_50" in rec.get("src", ""):
        return True
    if rec.get("src", "").startswith("teacher_old_"):  # legacy-prompt traces: only used for the frozen KL slice v1
        return zlib.crc32(f"{rec['id']}|{rec['k']}|{rec['src']}".encode()) % 5 == 0
    return zlib.crc32(f"{rec['id']}|{rec['k']}|{rec.get('src','')}".encode()) % 5 == 0


def load_records(files=None, include_unfinished=False):
    files = files or sorted(RUNS.glob("teacher_*.jsonl"))
    recs = []
    for f in files:
        src = Path(f).stem
        for l in open(f):
            r = json.loads(l); r["src"] = src
            if "text" in r and ((r["finish"] == "stop" and r["closed_think"]) or (include_unfinished and r["finish"] == "length")):
                recs.append(r)
    return recs


def problems():
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
    from gen_eval import load
    d = {}
    for name in ("hmmt_feb_50", "hmmt_2025_local"):  # HMMT only; AIME deliberately absent
        for r in load(name):
            d[(name_to_src(name), r["id"])] = r["problem"]
            d[(name_to_src(name).replace("teacher_", "teacher_old_"), r["id"])] = r["problem"]
    return d


def name_to_src(name):
    return {"hmmt_feb_50": "teacher_hmmt_feb_50", "hmmt_2025_local": "teacher_hmmt_2025_local"}[name]


LEGACY_INSTR = ("Solve the following math problem. Reason step by step, and put your final\n"
                "answer within \\boxed{}. Write only the final answer inside the box.")


def tokenize(rec, tok, probs, max_len, tail=False):
    msg = [{"role": "user", "content": f"{LEGACY_INSTR if rec['src'].startswith('teacher_old_') else _instr()}\n\nProblem:\n{probs[(rec['src'], rec['id'])].strip()}"}]
    p = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(p, add_special_tokens=False).input_ids
    rid = tok(rec["text"], add_special_tokens=False).input_ids
    if rec["finish"] == "stop":  # truncated (capped) traces have no end token
        rid = rid + [tok.convert_tokens_to_ids("<|im_end|>")]
    if tail and len(pid) + len(rid) > max_len:  # keep the prompt and the LAST tokens of the response (ends at </think> + answer)
        ids = pid + rid[len(rid) - (max_len - len(pid)):]
    else:
        ids = (pid + rid)[:max_len]
    return torch.tensor(ids), len(pid)


SLICE_FILE = RUNS / "kl_slice_v1.json"


def build(split: str, tok, max_len: int, max_seqs: int | None = None, correct_only: bool = False, include_unfinished: bool = False, tail: bool = False):
    """split 'kl' uses a FROZEN slice (kl_slice_v1.json, created on first use) so KL numbers stay comparable across runs."""
    probs = problems(); recs = load_records(include_unfinished=include_unfinished)
    if split == "kl" and SLICE_FILE.exists():
        pass  # frozen slice: membership comes from the file only
    else:
        recs = [r for r in recs if is_heldout(r) == (split == "kl")]
        if split != "kl" and SLICE_FILE.exists():  # frozen slice members are never calibration/KD data, whatever is_heldout says
            frozen = {tuple(x) for x in json.loads(SLICE_FILE.read_text())}
            recs = [r for r in recs if (r["src"], r["id"], r["k"]) not in frozen]
    if correct_only: recs = [r for r in recs if r["correct"]]
    recs.sort(key=lambda r: (r["src"], r["id"], r["k"]))
    if split == "kl":
        if not SLICE_FILE.exists():
            SLICE_FILE.write_text(json.dumps([[r["src"], r["id"], r["k"]] for r in recs[:max_seqs]]))
        keep = {tuple(x) for x in json.loads(SLICE_FILE.read_text())}
        recs = [r for r in recs if (r["src"], r["id"], r["k"]) in keep]
    out = [tokenize(r, tok, probs, max_len, tail) for r in recs[:max_seqs]]
    return out, recs[:max_seqs]
