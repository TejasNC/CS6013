"""Teacher-forced KL(teacher || student) on the held-out teacher-sample slice (response positions only)."""
import argparse, json, time, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import build


@torch.no_grad()
def logits_chunks(model, ids, chunk=1024):
    h = model.model(input_ids=ids[None]).last_hidden_state[0]
    W = model.lm_head.weight
    for i in range(0, h.shape[0], chunk):
        yield i, (h[i:i + chunk] @ W.T).float()


@torch.no_grad()
def kl_eval(teacher, student, seqs, dev="cuda"):
    tot, n = 0.0, 0; per = []
    for ids, plen in seqs:
        ids = ids.to(dev); s = 0.0; c = 0
        for (i, tl), (_, sl) in zip(logits_chunks(teacher, ids), logits_chunks(student, ids)):
            lo = max(i, plen - 1); hi = min(i + tl.shape[0], len(ids) - 1)  # positions predicting response tokens
            if hi <= lo: continue
            tl, sl = tl[lo - i:hi - i], sl[lo - i:hi - i]
            kl = (F.softmax(tl, -1) * (F.log_softmax(tl, -1) - F.log_softmax(sl, -1))).sum(-1)
            s += kl.sum().item(); c += kl.numel()
        tot += s; n += c; per.append(s / max(c, 1))
    return tot / n, n, per


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True); ap.add_argument("--student", required=True)
    ap.add_argument("--maxlen", type=int, default=6144); ap.add_argument("--nseq", type=int, default=48)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(); dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.teacher)
    seqs, recs = build("kl", tok, a.maxlen, a.nseq)
    T = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.bfloat16, device_map=dev).eval()
    S = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16, device_map=dev).eval()
    t = time.time(); kl, n, per = kl_eval(T, S, seqs)
    print(f"KL {kl:.5f} over {n} tokens, {len(seqs)} seqs ({time.time()-t:.0f}s)")
    if a.json: json.dump(dict(kl=kl, tokens=n, per_seq=per), open(a.json, "w"))
