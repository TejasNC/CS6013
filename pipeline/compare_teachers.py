#!/usr/bin/env python3
"""Step-0 measurement: 9B vs 4B on the non-validation local HMMT problems (traces generated with the TA prompt and sampling).
usage: compare_teachers.py <9b traces jsonl> [--limit 16384]
4B reference = the 60 TA-prompt traces generated with a 32k cap (so finish rates within 8k / 16k are readable from ntok)."""
import json, sys, re, collections
VAL = {"hmmt_feb_2025-5", "hmmt_feb_2025-11", "hmmt_feb_2025-12"}
def stats(rows, name, cap):
    rows = [r for r in rows if r["id"] not in VAL]
    n = len(rows); fin = [r for r in rows if r["finish"] == "stop"]
    out = dict(name=name, n=n, correct_rate=sum(r["correct"] for r in rows) / n,
               finish_rate=len(fin) / n, correct_among_finished=(sum(r["correct"] for r in fin) / max(len(fin), 1)),
               mean_len_finished=sum(r["ntok"] for r in fin) / max(len(fin), 1))
    for lim in (4096, 8192, 12288, 16384):
        if lim <= cap:
            ok = [r for r in rows if r["finish"] == "stop" and r["ntok"] <= lim]
            out[f"finish<= {lim}"] = len(ok) / n; out[f"correct<= {lim}"] = sum(r["correct"] for r in ok) / n
    if rows and "text" in rows[0]:
        w = [1000 * len(re.findall(r"\bWait\b", r["text"])) / max(r["ntok"], 1) for r in rows]
        out["wait_per_1k"] = sum(w) / n
    return out
if __name__ == "__main__":
    ref = [json.loads(l) for l in open("/mnt/nas/tejaschaudhari/emir_data/runs/teacher_hmmt_2025_local.jsonl")][:60]  # first 60 lines = the 32k-cap TA-prompt run
    print(json.dumps(stats(ref, "4B (32k cap, 60 traces)", 32000), indent=1))
    nine = [json.loads(l) for l in open(sys.argv[1])]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    print(json.dumps(stats(nine, f"9B ({cap} cap)", cap), indent=1))
