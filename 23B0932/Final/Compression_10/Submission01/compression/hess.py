"""Collect per-input-group Hessians X^T X / n (float32) from the shipped calibration token ids (offline)."""
import time

import torch
from transformers import AutoModelForCausalLM


def collect_hessians(model_dir: str, ids_path: str, dev: str = "cuda", log=print) -> dict:
    seqs = torch.load(ids_path)
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map=dev).eval()
    H, cnt = {}, {}

    def hook(key):
        def f(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            if key not in H:
                H[key] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
            H[key].addmm_(x.T, x)
            cnt[key] = cnt.get(key, 0) + x.shape[0]
        return f

    hs = []
    for i, layer in enumerate(m.model.layers):
        if hasattr(layer, "linear_attn"):
            hs.append(layer.linear_attn.in_proj_qkv.register_forward_pre_hook(hook(f"L{i}.attn_in")))
            hs.append(layer.linear_attn.out_proj.register_forward_pre_hook(hook(f"L{i}.o_in")))
        else:
            hs.append(layer.self_attn.q_proj.register_forward_pre_hook(hook(f"L{i}.attn_in")))
            hs.append(layer.self_attn.o_proj.register_forward_pre_hook(hook(f"L{i}.o_in")))
        hs.append(layer.mlp.gate_proj.register_forward_pre_hook(hook(f"L{i}.mlp_in")))
        hs.append(layer.mlp.down_proj.register_forward_pre_hook(hook(f"L{i}.down_in")))
    t = time.time()
    with torch.no_grad():
        for j, ids in enumerate(seqs):
            m.model(input_ids=ids.long()[None].to(m.device))
            if j % 8 == 0:
                log(f"calibration sequence {j}/{len(seqs)} {time.time() - t:.0f}s")
    out = {k: (H[k] / cnt[k]).cpu() for k in H}
    del m
    return out
