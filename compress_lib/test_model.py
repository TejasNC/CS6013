import torch, numpy as np, time, sys
from pathlib import Path
from safetensors import safe_open
from compress_lib.model import _quant_matrix, _dequant_matrix
M = Path(open('../scratch_model_path.txt').read().strip())
dev = "cuda"
def get(name):
    for f in sorted(M.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            if name in sf.keys(): return sf.get_tensor(name)
for name in ["model.language_model.layers.0.mlp.gate_proj.weight", "model.language_model.layers.3.self_attn.q_proj.weight",
             "model.language_model.layers.1.linear_attn.out_proj.weight"]:
    W = get(name)
    for bits in (3.67, 1.76):
        t = time.time(); q, st = _quant_matrix(W, name, bits, None, dev); 
        Wr = _dequant_matrix(q, name, dev).cpu()
        print(name.split("layers.")[1], bits, {k: round(v, 4) for k, v in st.items()}, "restored rel err", ((Wr - W.float()).norm() / W.float().norm()).item(), f"{time.time()-t:.1f}s")
