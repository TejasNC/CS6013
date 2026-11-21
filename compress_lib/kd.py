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
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.checkpoint import checkpoint

from .data import build, build_rollouts

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


def chunked_kl(h_t, h_s, W_t, W_s, weights, chunk=512, rev=0.0, tgt=None, ce=None):
    """sum_i w_i * [(1-rev) KL(t||s) + rev KL(s||t)]; at positions where ce[i] is True the loss is instead the
    cross-entropy of the FORCED token tgt[i] (used to teach emitting the wrap phrase / </think>, which the teacher itself would not emit).
    Logits are never materialised in full."""
    tot = h_s.new_zeros((), dtype=torch.float32)
    for i in range(0, h_s.shape[0], chunk):
        def f(hs, ht, w, tg, cm):
            ls = F.log_softmax((hs @ W_s.T).float(), -1)
            with torch.no_grad():
                lt = F.log_softmax((ht @ W_t.T).float(), -1)
            k = (lt.exp() * (lt - ls)).sum(-1)
            if rev > 0:
                k = (1 - rev) * k + rev * (ls.exp() * (ls - lt)).sum(-1)
            if cm is not None:
                k = torch.where(cm, -ls.gather(-1, tg.unsqueeze(-1)).squeeze(-1), k)
            return (k * w).sum()
        sl = slice(i, i + chunk)
        tot = tot + checkpoint(f, h_s[sl], h_t[sl], weights[sl], tgt[sl] if tgt is not None else None, ce[sl] if ce is not None else None, use_reentrant=False)
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
    ap.add_argument("--val-maxlen", type=int, default=6144, help="val windows use this length regardless of --maxlen (keeps val KL comparable across arms)")
    ap.add_argument("--complete-maxlen", type=int, default=0, help="if >0: whole (unwindowed) sequences up to this length for finished correct traces, end weights x5, shortest per problem boosted")
    ap.add_argument("--w-complete", type=float, default=1.5)
    ap.add_argument("--rollouts", default=None, help="student rollouts jsonl (gen_eval --keep-text on the student); added as on-policy KD data")
    ap.add_argument("--w-roll", type=float, default=1.0)
    ap.add_argument("--roll-maxlen", type=int, default=0, help="max length for rollout sequences (default: --maxlen)")
    ap.add_argument("--rev", type=float, default=0.0, help="weight of reverse KL (student||teacher) in the loss")
    ap.add_argument("--no-early-stop", action="store_true", help="always snapshot at each val check; never stop on val KL (on-policy KD selects checkpoints by other metrics)")
    ap.add_argument("--budget-data", default=None, help="budget_gen.py output: thinking prefix + wrap phrase + </think> + teacher answer")
    ap.add_argument("--w-budget", type=float, default=1.0); ap.add_argument("--w-budget-end", type=float, default=5.0)
    ap.add_argument("--budget-window", type=int, default=6000, help="for cuts >= 12288 keep only the last N prefix tokens (window ends at wrap + answer)")
    ap.add_argument("--init-adapters", default=None, help="warm start from adapters.pt/best_adapters.pt (same rank)")
    ap.add_argument("--end-windows", action="store_true", help="add prompt+last-K-token windows ending at </think> and the answer for finished correct traces")
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
    if a.rollouts:
        n_r = 0
        for ids, plen, r in build_rollouts(a.rollouts, tok, a.roll_maxlen or a.maxlen):
            if (r["src"], r["id"]) in val_probs: continue
            data.append((ids, plen, a.w_roll, r["finish"] == "stop" and len(ids) < (a.roll_maxlen or a.maxlen))); n_r += 1
        print(f"on-policy student rollouts added: {n_r}", flush=True)
    if a.budget_data:
        from .budget_gen import WRAP
        from .data import problems as _problems, _instr
        pb = _problems(); wrap_ids = tok(WRAP, add_special_tokens=False).input_ids
        bitems = []; per_cut = {}
        for l in open(a.budget_data):
            r = json.loads(l)
            if not r["correct"] or r["finish"] != "stop" or (r["src"], r["id"]) in val_probs: continue
            msg = [{"role": "user", "content": f"{_instr()}\n\nProblem:\n{pb[(r['src'], r['id'])].strip()}"}]
            pid = tok(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
            pre = tok(r["prefix"] + "\n\n", add_special_tokens=False).input_ids
            if r["cut"] >= 12288: pre = pre[-a.budget_window:]
            ans = tok(r["answer"], add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
            ids = pid + pre + wrap_ids + ans
            nresp_pre, n_wrap = len(pre), len(wrap_ids)
            w = torch.ones(len(ids) - len(pid)); w[nresp_pre:] = a.w_budget_end
            ce = torch.zeros(len(ids) - len(pid), dtype=torch.bool); ce[nresp_pre:nresp_pre + n_wrap + 16] = True  # forced wrap phrase incl. </think> and the first answer tokens; the answer body keeps teacher KL
            if r["cut"] >= 12288: ce[nresp_pre + n_wrap:] = True  # windowed item: the teacher on this truncated context lacks the dropped reasoning, so train the verified full-context answer with CE, not KL
            bitems.append((r["cut"], (torch.tensor(ids), len(pid), a.w_budget, False, (w, ce))))
        rr = random.Random(7); short = [b for b in bitems if b[0] <= 6144]; long_ = [b for b in bitems if b[0] > 6144]
        max_short = int(0.3 / 0.7 * len(long_))  # 4k+6k cuts capped at ~30% of the budget examples: the forced stop acts like a hazard rising with length
        rr.shuffle(short); short = short[:max_short]
        for cut, item in short + long_:
            data.append(item); per_cut[cut] = per_cut.get(cut, 0) + 1
        n_b = len(short) + len(long_)
        print(f"budget-forced sequences: {n_b} per cut {dict(sorted(per_cut.items()))} (4k+6k capped at 30%: kept {len(short)} of {len(bitems) - len(long_)})", flush=True)
    if a.complete_maxlen:
        cseqs, crecs = build("cal", tok, a.complete_maxlen + 600, None, include_unfinished=False)  # finished traces only, whole sequence
        ck = [i for i, r in enumerate(crecs) if not r["src"].startswith("teacher_old_") and r["correct"] and r["finish"] == "stop"
              and (r["src"], r["id"]) not in val_probs and len(cseqs[i][0]) < a.complete_maxlen + 600]
        byp = {}
        for i in ck: byp.setdefault((crecs[i]["src"], crecs[i]["id"]), []).append((crecs[i]["ntok"], i))
        n_short = 0
        for v in byp.values():
            for rank, (_, i) in enumerate(sorted(v)):
                ids, plen = cseqs[i]
                short = rank < a.shortest_per_problem
                data.append((ids, plen, a.w_short if short else a.w_complete, True)); n_short += short
        print(f"complete (unwindowed) short traces: {len(ck)} from {len(byp)} problems ({n_short} boosted x{a.w_short}), mean len {sum(len(cseqs[i][0]) for i in ck)/max(len(ck),1):.0f} tokens", flush=True)
    if a.val_maxlen != a.maxlen:  # re-tokenize the held-out problems at the fixed validation length
        vs, vr = build("cal", tok, a.val_maxlen, None, include_unfinished=True)
        val = [(ids, plen, 1.0, False) for (ids, plen), r in zip(vs, vr) if (r["src"], r["id"]) in val_probs and not r["src"].startswith("teacher_old_")]
    if a.end_windows:  # arm B: windows that land on the real </think> and answer (prompt + last tokens), end weights apply there
        tseqs, trecs = build("cal", tok, a.maxlen, None, include_unfinished=True, tail=True)
        tk = [i for i, r in enumerate(trecs) if not r["src"].startswith("teacher_old_")]
        tseqs, trecs = [tseqs[i] for i in tk], [trecs[i] for i in tk]
        n_end = 0
        for (ids, plen), r in zip(tseqs, trecs):
            if r["finish"] == "stop" and r["correct"] and (r["src"], r["id"]) not in val_probs:
                data.append((ids, plen, a.w_short if r["ntok"] <= 20000 else 1.0, True)); n_end += 1
        print(f"added {n_end} end-window sequences (prompt + last tokens, maxlen {a.maxlen})", flush=True)
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
    if a.init_adapters:
        ad = torch.load(a.init_adapters, map_location="cpu")
        for n, l in lora.items():
            d = ad[hf_to_ckpt_name(n)]; l.A.data.copy_(d["A"]); l.B.data.copy_(d["B"])
        print("warm-started from", a.init_adapters, flush=True)
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
        for ids, plen, *_ in val:
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
        for item in data:
            ids, plen, wseq, finished = item[:4]
            ids = ids.to(dev)[None]
            with torch.no_grad():
                h_t = T.model(input_ids=ids).last_hidden_state[0]
            h_s = S.model(input_ids=ids).last_hidden_state[0]
            lo, hi = plen - 1, ids.shape[1] - 1  # positions predicting response tokens
            tgt = ids[0, plen:]  # token predicted at position lo + j is tgt[j]
            if len(item) > 4:  # budget-forced item: explicit per-token weights and CE mask
                w, ce = item[4][0].to(dev), item[4][1].to(dev)
                loss = wseq * chunked_kl(h_t[lo:hi], h_s[lo:hi], W_t, W_s, w, rev=a.rev, tgt=tgt, ce=ce) / w.sum()
            else:
                w = torch.ones(hi - lo, device=dev)
                w[tgt == end_id] = a.w_end
                if finished: w[-a.tail:] = a.w_end  # answer tail only exists in traces that ended
                loss = wseq * chunked_kl(h_t[lo:hi], h_s[lo:hi], W_t, W_s, w, rev=a.rev) / w.sum()
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
                if a.no_early_stop:
                    torch.save(snapshot(), out / f"adapters_t{int((time.time()-t0)/60)}min.pt"); best_val = min(best_val, v)
                elif v < best_val:
                    best_val = v; worse = 0; torch.save(snapshot(), out / "best_adapters.pt"); torch.save(snapshot(), out / f"adapters_t{int((time.time()-t0)/60)}min.pt")
                else:
                    worse += 1
                    if worse >= 2: print("early stop: val KL rose twice in a row", flush=True); a.hours = 0
            if time.time() - t0 > a.hours * 3600: break
    torch.save(snapshot(), out / "adapters.pt")
    print("done", step, "steps", flush=True)


if __name__ == "__main__":
    main()
