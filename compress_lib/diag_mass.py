import sys, numpy as np, torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import build
from .model import _unpack2
M, S, COMP = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(M); seqs, _ = build("kl", tok, 4096, 6)
with safe_open(COMP + "/model.safetensors", "pt") as f: tier = torch.from_numpy(_unpack2(f.get_tensor("embed.tier").numpy(), 248320).astype(np.int64)).cuda()
T = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()
St = AutoModelForCausalLM.from_pretrained(S, dtype=torch.bfloat16, device_map="cuda").eval()
acc = {k: torch.zeros(3).cuda() for k in "ts"}; n = 0; norms = {}
with torch.no_grad():
    for ids, plen in seqs:
        ids = ids.cuda()[None]
        for k, m in (("t", T), ("s", St)):
            h = m.model(input_ids=ids).last_hidden_state[0, plen:]
            p = torch.softmax((h @ m.lm_head.weight.T).float(), -1)
            acc[k] += torch.stack([p[:, tier == i].sum(-1).sum() for i in range(3)])
        n += ids.shape[1] - plen
for k in "ts": print("mean prob mass on tiers T1,T2,T3 (%s):" % ("teacher" if k == "t" else "student"), (acc[k] / n).tolist())
for i in range(3):
    print("tier", i, "row norm teacher", T.model.embed_tokens.weight[tier == i].float().norm(dim=1).mean().item(), "student", St.model.embed_tokens.weight[tier == i].float().norm(dim=1).mean().item())
