"""Per-position forward KL(teacher||student) on STUDENT rollouts, binned by response position (0-1k, 1-4k, 4-8k); also stop-hazard comparison and Wait/repetition counts."""
import sys, json, os, torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import build_rollouts
M, S, ROLL = sys.argv[1:4]; N = int(os.environ.get("N", 16)); MAXLEN = int(os.environ.get("MAXLEN", 8800))
tok = AutoTokenizer.from_pretrained(M); end_think = tok.convert_tokens_to_ids("</think>")
rolls = build_rollouts(ROLL, tok, MAXLEN)[:N]
T = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()
St = AutoModelForCausalLM.from_pretrained(S, dtype=torch.bfloat16, device_map="cuda").eval()
bins = [(0, 1000), (1000, 4000), (4000, 8000)]; acc = {b: [0.0, 0] for b in bins}; haz_t = haz_s = 0.0; nt = 0
waits = []; reps = []
@torch.no_grad()
def fwd(m, ids):
    h = m.model(input_ids=ids[None].cuda()).last_hidden_state[0]; return h
for ids, plen, r in rolls:
    ht, hs = fwd(T, ids), fwd(St, ids); Wt, Ws = T.lm_head.weight, St.lm_head.weight
    for i in range(plen - 1, len(ids) - 1, 2048):
        hi = min(i + 2048, len(ids) - 1)
        lt = torch.log_softmax((ht[i:hi] @ Wt.T).float(), -1); ls = torch.log_softmax((hs[i:hi] @ Ws.T).float(), -1)
        kl = (lt.exp() * (lt - ls)).sum(-1)
        for j in range(hi - i):
            pos = i + j - (plen - 1)
            for b in bins:
                if b[0] <= pos < b[1]: acc[b][0] += kl[j].item(); acc[b][1] += 1
        haz_t += lt[:, end_think].exp().sum().item(); haz_s += ls[:, end_think].exp().sum().item(); nt += hi - i
    text = r["text"]; n = max(r["ntok"], 1)
    waits.append(1000 * len(re.findall(r"\bWait\b", text)) / n)
    g = ids[plen:].tolist(); grams = [tuple(g[k:k + 8]) for k in range(max(len(g) - 7, 0))]
    reps.append(1 - len(set(grams)) / max(len(grams), 1))
print("per-position forward KL on student rollouts:", {f"{b[0]}-{b[1]}": round(a[0] / max(a[1], 1), 4) for b, a in acc.items()}, flush=True)
print(f"stop hazard per 1k positions: teacher-scored {1000*haz_t/nt:.3f}  student {1000*haz_s/nt:.3f}  (n positions {nt})")
print(f"'Wait' per 1k tokens: {sum(waits)/len(waits):.2f}   repeated-8gram share: {sum(reps)/len(reps):.3f}   rollouts {len(rolls)}")
