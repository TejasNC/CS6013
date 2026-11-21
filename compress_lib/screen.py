"""Free-running screen: does a student loop / fail to close its thinking? (teacher-forced KL does not catch this).

  python -m compress_lib.screen --teacher <bf16 dir> --student <restored dir> [--lora adapters.pt --alpha .. ] \
        [--restore-teacher dnet,attn,ffn_gu,ffn_down,embed,...] --n 30 --tokens 3000 --out result.json

30 TA-prompt HMMT problems (hmmt_feb_50, eval problems are fine here: no training signal is produced) x --tokens new tokens, harness
sampling (temp 1.0, top_p .95, top_k 20, presence penalty 1.5), HF generate (no vLLM restart needed for variants).
Reports: loop fraction (share of samples whose last 1500 generated tokens are >50% repeated 8-grams), share that emitted </think>,
share that emitted <|im_end|>, mean generated length, and for each sample a one-line tail.
"""
import argparse, json, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from gen_eval import INSTR, load  # noqa: E402
from .kd import add_lora, hf_to_ckpt_name  # noqa: E402
from .model import group_of  # noqa: E402


class PresencePenalty(LogitsProcessor):
    def __init__(self, penalty, plen):
        self.p, self.plen, self.seen = penalty, plen, None

    def __call__(self, input_ids, scores):
        if self.seen is None:
            self.seen = torch.zeros_like(scores, dtype=torch.bool)
        self.seen.scatter_(1, input_ids[:, -1:].clamp(max=scores.shape[1] - 1), True) if input_ids.shape[1] > self.plen else None
        return scores - self.p * self.seen.to(scores.dtype)


def ngram_loop_fraction(ids, n=8, last=1500):
    t = ids[-last:]
    if len(t) < n + 10:
        return 0.0
    grams = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True); ap.add_argument("--student", required=True)
    ap.add_argument("--lora", default=None); ap.add_argument("--rank", type=int, default=8); ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--restore-teacher", default="", help="comma list of groups/prefixes to copy from the teacher: dnet,attn,ffn_gu,ffn_down,embed,band_a,band_b,band_c")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--tokens", type=int, default=3000); ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.teacher); tok.padding_side = "left"
    S = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16, device_map=dev).eval()
    if a.lora:
        ad = torch.load(a.lora, map_location="cpu")
        lora = add_lora(S, a.rank, a.alpha)
        for n, l in lora.items():
            d = ad[hf_to_ckpt_name(n)]; l.A.data.copy_(d["A"]); l.B.data.copy_(d["B"])
    if a.restore_teacher:
        T = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.bfloat16, device_map="cpu").eval()
        tp = dict(T.named_parameters()); wanted = a.restore_teacher.split(",")
        n_copied = 0
        for n, p in S.named_parameters():
            if n.endswith(".A") or n.endswith(".B"): continue
            nn_ = n.replace(".base.weight", ".weight")
            ck = nn_.replace("model.layers.", "model.language_model.layers.")
            hit = False
            if "embed" in wanted and "embed_tokens" in n: hit = True
            else:
                try: g = group_of(ck)
                except Exception: g = None
                if g and any(w == g.split(".")[0] or w == g or (w.startswith("band_") and g.endswith("." + w[-1])) for w in wanted): hit = True
            if hit and nn_ in tp:
                p.data.copy_(tp[nn_].data.to(p.device)); n_copied += 1
        print(f"restored {n_copied} tensors from teacher: {a.restore_teacher}", flush=True)
        del T
    rows = load("hmmt_feb_50")[:a.n]
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{INSTR}\n\nProblem:\n{r['problem'].strip()}"}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) for r in rows]
    torch.manual_seed(a.seed)
    res = []
    t0 = time.time()
    for i in range(0, len(prompts), a.batch):
        enc = tok(prompts[i:i + a.batch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        plen = enc.input_ids.shape[1]
        with torch.no_grad():
            out = S.generate(**enc, max_new_tokens=a.tokens, do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
                             logits_processor=LogitsProcessorList([PresencePenalty(1.5, plen)]), pad_token_id=tok.pad_token_id)
        for j in range(out.shape[0]):
            g = out[j, plen:].tolist()
            while g and g[-1] == tok.pad_token_id: g.pop()
            res.append(g)
        print(f"generated {len(res)}/{len(prompts)} ({time.time()-t0:.0f}s)", flush=True)
    end_think, im_end = tok.convert_tokens_to_ids("</think>"), tok.convert_tokens_to_ids("<|im_end|>")
    loops = [ngram_loop_fraction(g) for g in res]
    summ = dict(n=len(res), loop_frac_mean=sum(loops) / len(loops), loop_share_gt50=sum(x > 0.5 for x in loops) / len(loops),
                emitted_end_think=sum(end_think in g for g in res) / len(res), emitted_im_end=sum(im_end in g for g in res) / len(res),
                mean_len=sum(len(g) for g in res) / len(res))
    print(json.dumps(summ), flush=True)
    for g, l in list(zip(res, loops))[:6]:
        print(f"--- len {len(g)} loop {l:.2f} tail: {tok.decode(g[-160:])!r}", flush=True)
    if a.out:
        json.dump(dict(summary=summ, loops=loops, tails=[tok.decode(g[-300:]) for g in res]), open(a.out, "w"))


if __name__ == "__main__":
    main()
