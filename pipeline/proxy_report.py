#!/usr/bin/env python3
"""12k free-run proxy report. usage: proxy_report.py cand.jsonl [baseline.jsonl]  (baseline restricted to k < candidate's k count; same seeds)
Reports, for all 27 proxy problems and for the 24 outside {28, 15, 17} (the problems round 3's own KD held out): finished within the cap,
correct within the cap, correct among finished, mean finished length, 'Wait' per 1k tokens and the repeated-8-gram share."""
import json, re, sys
from collections import defaultdict
OUT3 = {"hmmt_feb_2025-28", "hmmt_feb_2025-15", "hmmt_feb_2025-17"}
def load(f, kmax=None):
    r = [json.loads(l) for l in open(f)]
    d = {}
    for x in r:
        if kmax is None or x["k"] < kmax: d[(x["id"], x["k"])] = x
    return list(d.values())
def rep8(ids):
    g = [tuple(ids[i:i + 8]) for i in range(max(len(ids) - 7, 0))]
    return 1 - len(set(g)) / max(len(g), 1)
def stats(rows):
    from transformers import AutoTokenizer
    global tok
    n = len(rows); fin = [r for r in rows if r["finish"] == "stop"]
    w = [1000 * len(re.findall(r"\bWait\b", r["text"])) / max(r["ntok"], 1) for r in rows if "text" in r]
    rp = [rep8(tok(r["text"], add_special_tokens=False).input_ids) for r in rows if "text" in r]
    return dict(n=n, finished=len(fin) / n, correct=sum(r["correct"] for r in rows) / n,
                correct_among_finished=sum(r["correct"] for r in fin) / max(len(fin), 1), mean_len_finished=sum(r["ntok"] for r in fin) / max(len(fin), 1),
                wait_per_1k=sum(w) / max(len(w), 1), rep8=sum(rp) / max(len(rp), 1))
if __name__ == "__main__":
    from transformers import AutoTokenizer
    M = open("/mnt/home/tejaschaudhari/emir/scratch_model_path.txt").read().strip(); tok = AutoTokenizer.from_pretrained(M)
    cand = load(sys.argv[1]); kmax = max(x["k"] for x in cand) + 1
    sets = [("cand", cand)] + ([("round3 baseline (same seeds, k<%d)" % kmax, load(sys.argv[2], kmax))] if len(sys.argv) > 2 else [])
    for name, rows in sets:
        for sub, rr in (("all 27", rows), ("24 outside {28,15,17}", [r for r in rows if r["id"] not in OUT3])):
            s = stats(rr); print(f"{name:38s} {sub:22s}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
