"""Collect per-input-group Hessians X^T X (float32) over calibration sequences (never the KL slice)."""
import sys, time, torch, argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--nseq", type=int, default=64); ap.add_argument("--maxlen", type=int, default=4096)
    a = ap.parse_args()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=dev).eval()
    seqs, recs = build("cal", tok, a.maxlen, a.nseq)
    print("calibration sequences", len(seqs), "tokens", sum(len(s[0]) for s in seqs), flush=True)
    H, cnt = {}, {}
    def hook(key):
        def f(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            if key not in H: H[key] = torch.zeros(x.shape[1], x.shape[1], device=dev)
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
        for j, (ids, _) in enumerate(seqs):
            m.model(input_ids=ids[None].to(dev))
            if j % 8 == 0: print(j, f"{time.time()-t:.0f}s", flush=True)
    out = {k: (H[k] / cnt[k]).cpu() for k in H}
    torch.save(out, a.out)
    print("saved", a.out, len(out))

main()
