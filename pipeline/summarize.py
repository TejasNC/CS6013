#!/usr/bin/env python3
"""Summarize gen_eval jsonl: accuracy (mean of per-problem means, SE over problems), length, 32k-cap rate.
usage: summarize.py run.jsonl [--vs other.jsonl]   (paired difference with SE over per-problem differences)"""
import json, sys, math, collections
def load(f):
    d = collections.defaultdict(list); L = []; cap = 0; n = 0
    for l in open(f):
        r = json.loads(l); d[r["id"]].append(r["correct"]); L.append(r["ntok"]); cap += r["finish"] == "length"; n += 1
    return {k: sum(v) / len(v) for k, v in d.items()}, L, cap, n
def stats(f):
    pm, L, cap, n = load(f); m = sum(pm.values()) / len(pm)
    se = math.sqrt(sum((v - m) ** 2 for v in pm.values()) / (len(pm) - 1) / len(pm)) if len(pm) > 1 else float("nan")
    return dict(acc=100 * m, se=100 * se, problems=len(pm), gens=n, mean_len=sum(L) / len(L), cap_rate=cap / n), pm
if __name__ == "__main__":
    s, pm = stats(sys.argv[1]); print(json.dumps(s))
    if "--vs" in sys.argv:
        s2, pm2 = stats(sys.argv[sys.argv.index("--vs") + 1]); ks = sorted(set(pm) & set(pm2))
        dd = [pm[k] - pm2[k] for k in ks]; md = sum(dd) / len(dd)
        se = math.sqrt(sum((x - md) ** 2 for x in dd) / (len(dd) - 1) / len(dd))
        print(f"paired diff {100*md:.2f} pts, SE {100*se:.2f} ({md/se if se else 0:.2f} SE) over {len(ks)} problems")
