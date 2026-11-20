"""Distillation of a restored (dequantized) student from the bf16 teacher, training LoRA adapters only.

  python -m compress_lib.kd --teacher <bf16 dir> --student <restored dir> --out <dir> --rank 16 --hours 6

Data: calibration-split teacher traces only (compress_lib.data.build("cal")): the 30 local HMMT-2025 problems; the KL slice and
hmmt_feb_50 / AIME are never touched. Per problem we keep correct traces and prefer the shortest (overthinking). Loss: token-level
KL(teacher || student) over response positions, x`--w-end` on the position that predicts </think> and on the last `--tail`
tokens (final answer). Adapters: out_proj/in_proj_qkv/in_proj_z/q,k,v,o/gate,up,down (every body linear). Base weights frozen.

Outputs adapters.pt (fp32 A,B per original weight name) every --save-every seconds and at the end.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.checkpoint import checkpoint

from .data import build

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        return y + (((x.float() @ self.A.T) @ self.B.T) * self.scale).to(y.dtype)


def add_lora(model, r, alpha):
    names = {}
    for n, m in list(model.named_modules()):
        for cn, c in list(m.named_children()):
            if cn in TARGETS and isinstance(c, nn.Linear) and ".layers." in f"{n}.{cn}":
                l = LoRALinear(c, r, alpha)
                setattr(m, cn, l)
                names[f"{n}.{cn}"] = l
    return names


def hf_to_ckpt_name(n: str) -> str:  # model.layers.3.mlp.up_proj -> model.language_model.layers.3.mlp.up_proj.weight
    return n.replace("model.layers.", "model.language_model.layers.") + ".weight"


def chunked_kl(h_t, h_s, W_t, W_s, weights, chunk=512):
    """sum_i w_i * KL(softmax(h_t W_t^T) || softmax(h_s W_s^T)), logits never materialised in full."""
    tot = h_s.new_zeros((), dtype=torch.float32)
    for i in range(0, h_s.shape[0], chunk):
        def f(hs, ht, w):
            ls = F.log_softmax((hs @ W_s.T).float(), -1)
            with torch.no_grad():
                lt = F.log_softmax((ht @ W_t.T).float(), -1)
            return ((lt.exp() * (lt - ls)).sum(-1) * w).sum()
        tot = tot + checkpoint(f, h_s[i:i + chunk], h_t[i:i + chunk], weights[i:i + chunk], use_reentrant=False)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True); ap.add_argument("--student", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--maxlen", type=int, default=8192); ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--w-end", type=float, default=5.0); ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=3600); ap.add_argument("--shortest-per-problem", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-short", type=float, default=2.5); ap.add_argument("--val-problems", type=int, default=3)
    ap.add_argument("--val-every", type=int, default=1800)
    a = ap.parse_args()
    dev = "cuda"; random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.teacher)
    # All TA-prompt calibration traces (correct, incorrect, capped), no frozen-KL-slice members. Whole problems are held out for val KL.
    seqs, recs = build("cal", tok, a.maxlen, None, include_unfinished=True)
    keep = [i for i, r in enumerate(recs) if not r["src"].startswith("teacher_old_")]  # KD uses TA-prompt traces only
    seqs, recs = [seqs[i] for i in keep], [recs[i] for i in keep]
    probs = sorted({(r["src"], r["id"]) for r in recs})
    rng = random.Random(1234); rng.shuffle(probs)
    val_probs = set(probs[:a.val_problems])
    # short correct traces (the shortest few per problem) get extra weight
    best = {}
    for i, r in enumerate(recs):
        if r["correct"] and r["finish"] == "stop": best.setdefault((r["src"], r["id"]), []).append((r["ntok"], i))
    boosted = {i for v in best.values() for _, i in sorted(v)[:a.shortest_per_problem]}
    data, val = [], []
    for i, ((ids, plen), r) in enumerate(zip(seqs, recs)):
        item = (ids, plen, a.w_short if i in boosted else 1.0, r["finish"] == "stop" and len(ids) < a.maxlen)  # end/answer-tail weights only if the trace was not cut by --maxlen
        (val if (r["src"], r["id"]) in val_probs else data).append(item)
    print(f"KD data: {len(data)} train seqs ({sum(1 for d in data if d[2] > 1)} boosted x{a.w_short}) from {len(probs) - len(val_probs)} problems, "
          f"{sum(len(d[0]) for d in data)} tokens; val {len(val)} seqs from {len(val_probs)} problems", flush=True)
    end_id = tok.convert_tokens_to_ids("</think>")

    T = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.bfloat16, device_map=dev).eval()
    for p in T.parameters(): p.requires_grad_(False)
    S = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16, device_map=dev)
    for p in S.parameters(): p.requires_grad_(False)
    S.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # frozen embeddings: reentrant ckpt would give LoRA no grads
    S.config.use_cache = False
    lora = add_lora(S, a.rank, a.alpha)
    params = [p for l in lora.values() for p in (l.A, l.B)]
    print(f"LoRA r={a.rank} on {len(lora)} matrices, {sum(p.numel() for p in params)/1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    W_t, W_s = T.lm_head.weight, S.lm_head.weight
    t0 = time.time(); last_save = t0; step = 0; micro = 0; run = []
    last_val = t0; best_val = float("inf"); worse = 0

    def snapshot():
        return {hf_to_ckpt_name(n): {"A": l.A.detach().cpu(), "B": l.B.detach().cpu(), "scale": l.scale} for n, l in lora.items()}

    @torch.no_grad()
    def val_kl():
        S.eval(); tot = 0.0; n = 0
        for ids, plen, _, _ in val:
            ids = ids.to(dev)[None]
            h_t = T.model(input_ids=ids).last_hidden_state[0]; h_s = S.model(input_ids=ids).last_hidden_state[0]
            lo, hi = plen - 1, ids.shape[1] - 1
            tot += chunked_kl(h_t[lo:hi], h_s[lo:hi], W_t, W_s, torch.ones(hi - lo, device=dev)).item(); n += hi - lo
        S.train(); return tot / max(n, 1)

    v0 = val_kl(); print(f"val KL (unweighted, held-out problems) at start: {v0:.5f}", flush=True); best_val = v0
    torch.save(snapshot(), out / "best_adapters.pt")
    S.train()
    while time.time() - t0 < a.hours * 3600:
        random.shuffle(data)
        for ids, plen, wseq, finished in data:
            ids = ids.to(dev)[None]
            with torch.no_grad():
                h_t = T.model(input_ids=ids).last_hidden_state[0]
            h_s = S.model(input_ids=ids).last_hidden_state[0]
            lo, hi = plen - 1, ids.shape[1] - 1  # positions predicting response tokens
            w = torch.ones(hi - lo, device=dev)
            tgt = ids[0, plen:]  # token predicted at position lo + j is tgt[j]
            w[tgt == end_id] = a.w_end
            if finished: w[-a.tail:] = a.w_end  # answer tail only exists in traces that ended
            loss = wseq * chunked_kl(h_t[lo:hi], h_s[lo:hi], W_t, W_s, w) / w.sum()
            (loss / a.accum).backward(); run.append(loss.item()); micro += 1
            if micro % a.accum == 0:
                frac = (time.time() - t0) / (a.hours * 3600)
                for g in opt.param_groups: g["lr"] = a.lr * min(1.0, step / 10 + 0.1) * (0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 5 == 0:
                    print(f"step {step} micro {micro} loss {sum(run[-20:])/len(run[-20:]):.5f} lr {opt.param_groups[0]['lr']:.2e} {time.time()-t0:.0f}s", flush=True)
            if time.time() - last_save > a.save_every or time.time() - t0 > a.hours * 3600:
                last_save = time.time()
                torch.save(snapshot(), out / "adapters.pt")
                print("saved adapters", flush=True)
            if time.time() - last_val > a.val_every and micro % a.accum == 0:
                last_val = time.time(); v = val_kl()
                print(f"VAL step {step} elapsed {time.time()-t0:.0f}s val KL {v:.5f} (best {best_val:.5f})", flush=True)
                if v < best_val:
                    best_val = v; worse = 0; torch.save(snapshot(), out / "best_adapters.pt"); torch.save(snapshot(), out / f"adapters_t{int((time.time()-t0)/60)}min.pt")
                else:
                    worse += 1
                    if worse >= 2: print("early stop: val KL rose twice in a row", flush=True); a.hours = 0
            if time.time() - t0 > a.hours * 3600: break
    torch.save(snapshot(), out / "adapters.pt")
    print("done", step, "steps", flush=True)


if __name__ == "__main__":
    main()
