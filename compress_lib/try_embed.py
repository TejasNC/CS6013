"""Embedding-tier variants evaluated by KL on the frozen slice, with the (fixed) A1 body. Reports embedding bits too."""
import sys, json, numpy as np, torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import build
from .kl import kl_eval
from .model import compress_embedding, restore_embedding
M, S, COUNTS = sys.argv[1:4]
variants = json.loads(sys.argv[4])
tok = AutoTokenizer.from_pretrained(M); seqs, _ = build("kl", tok, 6144, 40)
T = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()
St = AutoModelForCausalLM.from_pretrained(S, dtype=torch.bfloat16, device_map="cuda").eval()
E0 = T.model.embed_tokens.weight.data.cpu(); counts = np.load(COUNTS)
print("reference (teacher embedding)", flush=True); St.model.embed_tokens.weight.data.copy_(E0.cuda()); print(" KL %.5f" % kl_eval(T, St, seqs)[0], flush=True)
for v in variants:
    out, st = compress_embedding(E0, counts, v["n1"], v["n2"], v["b1"], v["b2"], v["K"], "cuda", list(range(248044, 248070)), v.get("rank", "count"))
    bits = sum(t.numel() * t.element_size() * 8 for t in out.values())
    E = restore_embedding(out, 248320, 2560, "cuda")
    St.model.embed_tokens.weight.data.copy_(E.cuda())
    print(json.dumps(v), "embed bits %.3fe9" % (bits / 1e9), "KL %.5f" % kl_eval(T, St, seqs)[0], flush=True)
