"""Ablation: restore teacher weights for one group (kind.band) into the quantized student; KL drop = that group's contribution."""
import sys, json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import build
from .kl import kl_eval
from .model import group_of
M, S = sys.argv[1:3]
tok = AutoTokenizer.from_pretrained(M); seqs, _ = build("kl", tok, 6144, 40)
T = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()
St = AutoModelForCausalLM.from_pretrained(S, dtype=torch.bfloat16, device_map="cuda").eval()
tp = dict(T.named_parameters()); sp = dict(St.named_parameters())
def ck(n): return n.replace("model.layers.", "model.language_model.layers.")
groups = {}
for n in sp:
    try: g = group_of(ck(n))
    except Exception: continue
    groups.setdefault(g, []).append(n)
print("student KL", kl_eval(T, St, seqs)[0], flush=True)
for g, ns in sorted(groups.items()):
    old = {n: sp[n].data.clone() for n in ns}
    for n in ns: sp[n].data.copy_(tp[n].data)
    kl = kl_eval(T, St, seqs)[0]
    nparam = sum(sp[n].numel() for n in ns)
    print(f"restore {g:12s} ({nparam/1e6:6.1f}M params) KL {kl:.5f}", flush=True)
    for n in ns: sp[n].data.copy_(old[n])
