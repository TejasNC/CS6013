"""Termination hazard: sum over response positions (before the true </think>) of P(</think>), teacher vs students, on full finished traces."""
import sys, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import build
M, *S = sys.argv[1:]
maxtok = int(os.environ.get("MAXTOK", 14000)); nmax = int(os.environ.get("NMAX", 6))
tok = AutoTokenizer.from_pretrained(M)
seqs, recs = build("cal", tok, maxtok + 700, None)
items = [(ids, plen, r) for (ids, plen), r in zip(seqs, recs) if r["finish"] == "stop" and r["correct"] and len(ids) < maxtok + 700
         and not r["src"].startswith("teacher_old_")]
items.sort(key=lambda x: len(x[0])); items = items[:nmax]
end_think = tok.convert_tokens_to_ids("</think>")
print("items", [(r["id"], len(i)) for i, _, r in items], flush=True)
models = {"teacher": AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()}
for s in S: models[s.split("/")[-1]] = AutoModelForCausalLM.from_pretrained(s, dtype=torch.bfloat16, device_map="cuda").eval()
lora = os.environ.get("LORA")
@torch.no_grad()
def hazards(m, ids, plen):
    x = ids[None].cuda(); h = m.model(input_ids=x).last_hidden_state[0]; W = m.lm_head.weight
    et = (ids == end_think).nonzero()[-1].item(); out = []
    for i in range(plen - 1, et, 4096):
        hi = min(i + 4096, et)  # positions predicting tokens plen..et ; token at position et is </think>, predicted at et-1
        lg = (h[i:hi] @ W.T).float(); lp = torch.log_softmax(lg, -1)[:, end_think]; out.append(lp)
    lp = torch.cat(out); p = lp.exp()
    early = p[:min(int(os.environ.get('EARLY', 3000)), len(p) - 1)].sum().item()
    return p[:-1].sum().item(), p[-1].item(), p[:-1].max().item(), early
for name, m in models.items():
    rows = [hazards(m, ids, plen) for ids, plen, r in items]
    print(f"{name:20s} sum P(</think>) before true end: {[round(a, 3) for a, b, c, d in rows]}  mean {sum(a for a,_,_,_ in rows)/len(rows):.3f} | P at true end mean {sum(b for _,b,_,_ in rows)/len(rows):.3f} | max early P {max(c for _,_,c,_ in rows):.3f} | EARLY(<3k tokens) sum P mean {sum(d for _,_,_,d in rows)/len(rows):.4f}", flush=True)
