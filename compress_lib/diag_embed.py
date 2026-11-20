"""Diagnostic: which embedding tier carries the KL? Restores teacher rows for a chosen tier into the student."""
import sys, json, numpy as np, torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import build
from .kl import kl_eval
from .model import _unpack2
M, S, COMP = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(M)
seqs, _ = build("kl", tok, 6144, 40)
with safe_open(COMP + "/model.safetensors", "pt") as f: tier = _unpack2(f.get_tensor("embed.tier").numpy(), 248320)
T = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()
St = AutoModelForCausalLM.from_pretrained(S, dtype=torch.bfloat16, device_map="cuda").eval()
Ws = St.model.embed_tokens.weight.data.clone(); Wt = T.model.embed_tokens.weight.data
print("tier sizes", [(tier == i).sum() for i in range(3)])
for name, tiers in [("none restored", []), ("T1 restored", [0]), ("T2 restored", [1]), ("T3 restored", [2]), ("T2+T3 restored", [1, 2])]:
    St.model.embed_tokens.weight.data.copy_(Ws)
    for t in tiers:
        rows = torch.from_numpy(np.where(tier == t)[0]).cuda()
        St.model.embed_tokens.weight.data[rows] = Wt[rows]
    kl, n, _ = kl_eval(T, St, seqs); print(f"{name:16s} KL {kl:.5f} ({n} tokens)", flush=True)
