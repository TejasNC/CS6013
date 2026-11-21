"""Stop-token check: probability of </think> and <|im_end|> at their true positions on finished teacher traces (tail windows)."""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import build
M, *S = sys.argv[1:]
dev = sys.argv[-1] if sys.argv[-1] in ("cpu", "cuda") else "cpu"; S = [s for s in S if s not in ("cpu", "cuda")]
tok = AutoTokenizer.from_pretrained(M)
seqs, recs = build("cal", tok, 2600, None, tail=True)
import os, random
allp = sorted({(r["src"], r["id"]) for r in recs if not r["src"].startswith("teacher_old_")}); random.Random(1234).shuffle(allp)
val_p = set(allp[:3]); train_p = set(allp[3:])
which = os.environ.get("WHICH", "all")  # 'val' = the 3 problems held out from KD, 'train' = the rest
items = [(ids, plen, r) for (ids, plen), r in zip(seqs, recs) if r["finish"] == "stop" and r["correct"] and not r["src"].startswith("teacher_old_")
         and (which == "all" or (which == "val") == ((r["src"], r["id"]) in val_p))][:8]
print("items", len(items), "problems", sorted({r["id"] for _, _, r in items}), which, flush=True)
end_think = tok.convert_tokens_to_ids("</think>"); im_end = tok.convert_tokens_to_ids("<|im_end|>")
import os
models = {} if os.environ.get("SKIP_TEACHER") else {"teacher": AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=dev).eval()}
for s in S: models[s.split("/")[-1]] = AutoModelForCausalLM.from_pretrained(s, dtype=torch.bfloat16, device_map=dev).eval()
if os.environ.get("LORA"):  # diagnostic: apply LoRA adapters (rank 8, alpha 8) live to the student models
    from .kd import add_lora, hf_to_ckpt_name
    ad = torch.load(os.environ["LORA"], map_location="cpu")
    for name, m in models.items():
        lora = add_lora(m, 8, 8.0)
        for n, l in lora.items():
            d = ad[hf_to_ckpt_name(n)]; l.A.data.copy_(d["A"]); l.B.data.copy_(d["B"])
if os.environ.get("PIN"):  # diagnostic: exact (teacher) embedding rows for the special tokens in every student
    T_ = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=dev).eval()
    lo_, hi_ = int(os.environ["PIN"].split(":")[0]), int(os.environ["PIN"].split(":")[1])
    for name, m in models.items():
        m.model.embed_tokens.weight.data[lo_:hi_] = T_.model.embed_tokens.weight.data[lo_:hi_]
    del T_
with torch.no_grad():
    for name, m in models.items():
        pt, pe, rk = [], [], []
        for ids, plen, r in items:
            x = ids[None].to(dev); lg = m(input_ids=x).logits[0].float(); p = torch.softmax(lg, -1)
            it = (ids == end_think).nonzero()[-1].item()      # position of </think>: predicted at it-1
            pt.append(p[it - 1, end_think].item())
            pe.append(p[len(ids) - 2, im_end].item())          # last token is <|im_end|>, predicted at len-2
            rk.append((lg[len(ids) - 2] > lg[len(ids) - 2, im_end]).sum().item())
        print(f"{name:22s} P(</think>) {sum(pt)/len(pt):.4f}  P(<|im_end|> after answer) {sum(pe)/len(pe):.4f}  rank of <|im_end|> {rk}", flush=True)
