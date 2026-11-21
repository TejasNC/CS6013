"""Arm E: stop-logit bias. The embedding is tied (lm_head = embed_tokens), so adding delta = beta * mu / (mu . mu) to the </think> row raises
its logit by ~beta nats at an average state, where mu = mean final-norm hidden state at sentence boundaries of teacher traces (computed with the
SAME merged student whose weights will ship). The new row is stored exactly in bf16 as an override (41 kbit; counted by the size rule).

  python -m compress_lib.stopbias mu   --student RESTORED_DIR --out mu.pt          (GPU: forwards ~10 calibration traces)
  python -m compress_lib.stopbias apply --comp COMP_DIR --restored RESTORED_DIR --mu mu.pt --beta 0.5 --out NEW_COMP_DIR
Restore applies the override (model.py restore_embedding). Note: the same row is also the input embedding of </think>, which is only fed
after the token has been emitted.
"""
import argparse, shutil, json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def cmd_mu(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .data import build
    M = open(Path(__file__).resolve().parents[2] / "scratch_model_path.txt").read().strip()
    tok = AutoTokenizer.from_pretrained(M)
    seqs, recs = build("cal", tok, 4096, None)
    val = {(r["src"], r["id"]) for r in recs}; import random; allp = sorted(val); random.Random(1234).shuffle(allp); vp = set(allp[:3])
    keep = [i for i, r in enumerate(recs) if (r["src"], r["id"]) not in vp and not r["src"].startswith("teacher_old_")][:a.n]
    seqs = [seqs[i] for i in keep]  # calibration traces only: the 3 validation problems are excluded from mu
    S = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16, device_map=a.dev).eval()
    tot = torch.zeros(2560, device=a.dev, dtype=torch.float32); n = 0
    with torch.no_grad():
        for ids, plen in seqs:
            x = ids.to(a.dev)[None]; h = S.model(input_ids=x).last_hidden_state[0].float()
            for i in range(plen, len(ids) - 1):  # boundary = position whose token ends a sentence (". ", "?", "!", or newline)
                t = tok.decode(ids[i:i + 1])
                if t.rstrip().endswith((".", "?", "!")) or "\n" in t:
                    tot += h[i]; n += 1
    mu = tot / max(n, 1)
    torch.save(dict(mu=mu.cpu(), n=n), a.out); print("mu from", n, "boundary positions, |mu|^2 =", float(mu @ mu))


def cmd_shift(a):
    """Realized </think> logit shift h . delta on HELD-OUT states (3 validation problems' traces), p10/p50/p90, boundary vs mid-sentence."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .data import build
    M = open(Path(__file__).resolve().parents[2] / "scratch_model_path.txt").read().strip()
    tok = AutoTokenizer.from_pretrained(M)
    seqs, recs = build("cal", tok, 4096, None)
    val = {(r["src"], r["id"]) for r in recs}
    import random
    allp = sorted(val); random.Random(1234).shuffle(allp); vp = set(allp[:3])
    held = [(ids, plen) for (ids, plen), r in zip(seqs, recs) if (r["src"], r["id"]) in vp and not r["src"].startswith("teacher_old_")][:8]
    mu = torch.load(a.mu)["mu"].float().to(a.dev)
    S = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16, device_map=a.dev).eval()
    bnd, mid = [], []
    with torch.no_grad():
        for ids, plen in held:
            h = S.model(input_ids=ids.to(a.dev)[None]).last_hidden_state[0].float()
            for i in range(plen, len(ids) - 1):
                t = tok.decode(ids[i:i + 1]); is_b = t.rstrip().endswith((".", "?", "!")) or "\n" in t
                (bnd if is_b else mid).append(float(h[i] @ mu) / float(mu @ mu))
    import numpy as np
    for name, v in (("boundary", bnd), ("mid-sentence", mid)):
        q = np.percentile(v, [10, 50, 90]); print(f"{name:13s} n={len(v)} shift per unit beta p10/p50/p90 = {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}  -> beta=0.5: {0.5*q[0]:.2f}/{0.5*q[1]:.2f}/{0.5*q[2]:.2f}  beta=1.0: {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}")


def cmd_apply(a):
    from .model import read_all
    comp, out = Path(a.comp), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mu = torch.load(a.mu)["mu"].float()
    from transformers import AutoTokenizer
    M = open(Path(__file__).resolve().parents[2] / "scratch_model_path.txt").read().strip()
    tid = AutoTokenizer.from_pretrained(M).convert_tokens_to_ids("</think>")
    with safe_open(str(Path(a.restored) / "model.safetensors"), "pt") as f:
        row = f.get_tensor("model.language_model.embed_tokens.weight")[tid].float()
    new = (row + a.beta * mu / (mu @ mu)).bfloat16()
    tensors = dict(read_all(comp))
    tensors["embed.override.idx"] = torch.tensor([tid], dtype=torch.int32)
    tensors["embed.override.rows"] = new[None].contiguous()
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "model.safetensors"), metadata={"format": "pt"})
    for fp in comp.iterdir():
        if fp.name != "model.safetensors" and fp.is_file(): shutil.copy(fp, out / fp.name)
    print("override row for token", tid, "delta norm", float((new.float() - row).norm()), "row norm", float(row.norm()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mu"); m.add_argument("--student", required=True); m.add_argument("--dev", default="cuda"); m.add_argument("--n", type=int, default=40); m.add_argument("--out", required=True)
    sh = sub.add_parser("shift"); sh.add_argument("--student", required=True); sh.add_argument("--mu", required=True); sh.add_argument("--dev", default="cuda")
    p = sub.add_parser("apply"); p.add_argument("--comp", required=True); p.add_argument("--restored", required=True)
    p.add_argument("--mu", required=True); p.add_argument("--beta", type=float, required=True); p.add_argument("--out", required=True)
    a = ap.parse_args(); {"mu": cmd_mu, "shift": cmd_shift, "apply": cmd_apply}[a.cmd](a)
