# CS6013 — Efficient AI — Project Source of Truth

**Student:** Tejas Chaudhari · Enrollment `23D0383` · IIT Bombay
**Course:** CS6013, Efficient AI, Fall 2026
**Instructor:** Aditya Desai · **TAs:** Lavinia Nongbri, Hasmita Kurre
**Course site:** https://apd10.github.io/CS6013_Fall26/
**TA evaluation repo (primary source for grading mechanics):** https://github.com/lavinianongbri/cs6013
**Course project PDF ("Autumn 2026–2027" instructions, primary source for rules):** `EfficientAI_Project_instructions.pdf`
**Deprecated — do not use:** `https://github.com/skylight-org/efficient_ai_CS6013_2026Fall` (student starter template; contains a stale base model, stale file-naming convention, and a benchmark mix that doesn't match the real grader — kept off this doc entirely except where noted)

This document consolidates the course PDFs, the live course site, and the TA's evaluation repo into one working reference. Facts below are stated directly; where something is still genuinely unconfirmed by the TAs, it's marked ❓ in §11.

---

## 1. Base Model & Domain

- **Base model:** `Qwen/Qwen3.5-4B` — confirmed by the literal `model_name` field in the TA eval repo's committed `evaluation/configs/config.yaml`, and independently corroborated by the course site and the project PDF.
- No separate "Instruct" checkpoint exists for this model. `Qwen/Qwen3.5-4B` is itself the post-trained, chat-ready checkpoint (thinks by default, ships its own chat template). `Qwen/Qwen3.5-4B-Base` is the separate pretraining-only variant — don't confuse either with the unrelated `Qwen3-4B-Instruct-2507` naming from the older Qwen3 line.
- **It's multimodal — prune the vision tower.** `Qwen/Qwen3.5-4B` is `Image-Text-to-Text`: a 4B-parameter language backbone plus a vision encoder (ViT), bundled as one ~5B-parameter checkpoint. For a text-only math-domain project, the vision encoder is dead weight. Pruning it outright is close to free compression, and it's officially supported — Qwen's own vLLM serving docs include a `--language-model-only` flag to skip the vision encoder. This should be the first step of your `compress.py`, before quantization/pruning of what's left.
- **Language-backbone architecture** (for sizing your pruning/quantization budget): 4B params, hidden dim 2560, 32 layers, hybrid layout `8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`, Gated DeltaNet (32 V-heads / 16 QK-heads, head dim 128), Gated Attention (16 Q-heads / 4 KV-heads, head dim 256, RoPE dim 64), FFN intermediate dim 9216, native context 262,144 tokens.
- **Embedding/LM-head are tied** — same 248,320 × 2560 matrix used for both input embedding and output projection. Pruning vocab rows saves parameters once, not twice.
- **Target domain:** Math, evaluated on a hidden dataset. The TA eval repo's currently-committed benchmark (GPQA Diamond) is a placeholder, not the real math-domain benchmark — don't read GPQA-specific performance as predictive of your leaderboard score.

## 2. Tracks & Compression Targets

| Track | What's evaluated | Notes |
|---|---|---|
| **Track 1** | Checkpoint size alone. | Pure compression; no CUDA-speedup requirement. |
| **Track 2** | Checkpoint size, but the technique must be one that *could* speed up CUDA inference in principle (no need to implement the speedup). | |
| **Track 2+ (bonus)** | Real CUDA kernels for specific submodules, with demonstrated speedups at scale. | Open only to Checkpoint-1 top performers; A100/H100 access; up to **+5%** on the total course grade. |

Every student submits for both Track 1 and Track 2; Track 2+ is optional. Compression targets are **10%, 20%, 40% of original size** (compress model *to* X%, not *by* X%), for each track.

**Technique eligibility:**
- **Lossless compression** (gzip/zstd/etc. on top of a Track 1 submission) does **not** count toward Track 2 unless it comes with a real CUDA-speedup story — fine for Track 1, not Track-2-eligible on its own.
- **Knowledge distillation is ineligible** as a compression technique for this project.

## 3. Compression % — Working Formula

The TAs' own "Check Compression" script is not yet published (their repo's README says "Will update soon," confirmed by pulling the full repo — no size-checking code exists anywhere in it). Until they publish one, use this as the working formula:

> **Compression % = size(compressed state dict) / size(original state dict) × 100**, where `size(state dict) = Σ over all tensors of numel(tensor) × itemsize(tensor.dtype)` — raw parameter byte-count at each tensor's actual stored dtype (not a serialized file size, no headers/metadata, no gzip).

```python
def state_dict_size_bytes(state_dict) -> int:
    return sum(t.numel() * t.element_size() for t in state_dict.values())

target_bytes = 0.10 * state_dict_size_bytes(original_state_dict)   # for a "10%" submission
```

- **Baseline:** compute `original_state_dict` from the **full `Qwen/Qwen3.5-4B` checkpoint as downloaded, including the vision tower** — not a pre-pruned, text-only version. This means ViT pruning genuinely counts toward your 10/20/40% budget rather than being done "for free" before the baseline is drawn.
- **Blind spot:** magnitude pruning (zeroing weights, same shape/dtype) doesn't move this number at all — `numel` and `dtype` are unchanged.
- **GPTQ/packed formats:** sum the actual stored tensors (`qweight`, `qzeros`, `scales`, `g_idx`, ...) rather than naively computing "4 bits × original numel" — quantization metadata adds bytes back.

This is our own working assumption for planning and self-evaluation, not TA-confirmed — revisit if/when they publish their own method.

## 4. Timeline & Weightage

- **Checkpoint 1 (mid-term):** September 15, 2026. **Checkpoint 2 (end-term):** November 15, 2026.
- Weekly submission window: Saturday 12:00 AM through Friday 11:59 PM; leaderboard updates weekly.
- Top-k performers at Checkpoint 1 present their ideas in class; code stays hidden until after that presentation, then a "snapshot" of everyone's code becomes public — cite anything reused later.
- Project = **60% of the course grade**. Track 2+ adds up to **+5%** on top of the total course grade.

| | Eval 1 (mid-term) | Eval 2 (end-term) |
|---|---|---|
| Weight | 50% | 50% |
| — Leaderboard position | 60% | 60% |
| — Report | 40% | 40% |

**Report:** PDF, max 3 pages of text (figures/tables/references may go after page 3, referenced from the text). Over 3 pages of text → not evaluated.

## 5. AI Policy & Honor Code

- **Code/ideation:** Any-AI policy — use whatever AI tooling you want; interesting AI usage is explicitly rewarded if reported.
- **Report writing:** strict NO-AI policy, including grammar/spell-check. Violation → 0 for the entire project.
- **Honor code** (signed at every submission): *"I hereby declare that this submission is my own work and I have not shared my code or artifacts with any other student. I understand that any violation of the honor code will result in a grade of 0 for the project."*
- Repos must not be shared with any other student; add TAs + instructor as collaborators.

## 6. Repository Requirements (GitHub)

**Required files:** `compress.py`, `decompress.py`, `pyproject.toml`, with modular `compression/` and `decompression/` packages.
```
python compress.py   --model_name <name> --checkpoint_path <path> --output_path <path>
python decompress.py --model_name <name> --checkpoint_path <path> --output_path <path>
```
(`code.py` with `convert_from_hf_checkpoint`/`convert_to_hf_checkpoint` is **not** required — that convention belongs to the deprecated starter repo.)

**Restore/output dtype:** not pinned to one value. The TA eval repo serves the restored checkpoint via `vllm ... --dtype auto`, which reads dtype straight from the checkpoint's own config/tensors — `decompress.py` can write out whatever dtype `save_pretrained()` produces (bf16/fp16/fp32), as long as the checkpoint's declared dtype matches what's actually stored.

**pyproject.toml:** mandatory, must list all external dependencies, and must be CUDA-12.6-compatible.

**README.md:** required — document setup/usage/repro instructions (not the compression technique itself), and the base→compressed dtype (e.g. `torch.float16 → int4`).

**Folder structure:**
```
CS6013/
  <roll number>/
    Week01/
      Compression_<compression target>/
        Submission01/
          compress.py
          decompress.py
          pyproject.toml
          compression/
          decompression/
          README.md
```
No "Track" segment in the path — the compression-target folder is what varies. **Wrong folder structure or naming → 0 grade for that submission.**

## 7. Hugging Face Checkpoint

- Push the compressed checkpoint to a HF repo, and **keep it public.**
- URL format: `huggingface.co/<user>/<EnrollmentNo>-Week<week>-Compression<target>-Submission<num>`
- Checkpoint must be self-contained — TAs must not need to run your compression code to use it.
- Must **not** include README, source code, notebooks, logs, or anything beyond what's needed to load/decompress/evaluate the model.
- **No modification after submission** — treat this as a hard rule.

## 8. Submission Mechanics

- **Cadence:** max 1 submission per week **per compression target** — i.e. up to 3 submissions/week/track (one per 10/20/40% target).
- Latest submission in a window wins if you submit more than once.
- No duplicate submissions: once a model is successfully evaluated (on the leaderboard), you can't resubmit it identically. A submission scored 0 for a spec violation may be fixed and resubmitted.
- Submit via a Google Form (link/details from TAs) with links to the GitHub repo and HF checkpoint.
- Leaderboard is anonymized — you only see your own score.
- Wrong base model or naming violations → 0 for that submission.

**Evaluation parameters** (confirmed exactly from the TA eval repo's committed `config.yaml`):
```yaml
max_new_tokens: 32000
temperature: 1.0
top_p: 0.95
top_k: 20
min_p: 0.0
presence_penalty: 1.5
repetition_penalty: 1.0
enable_thinking: true
```
These are verbatim Qwen's own recommended "thinking mode for general tasks" sampling settings for Qwen3.5 — the TAs are using Qwen's stock config, not a custom one. (`request_timeout`, `max_concurrency`, `limit` in the committed file are smoke-test settings, not part of the graded config.)

**Evaluation mechanics:** the model is served via vLLM's OpenAI-compatible API (`vllm ... --dtype auto --max-model-len 33000`), then queried with `AsyncOpenAI` chat completions — not a direct HF `transformers` load. Fixed generation `seed=42`.

**Environment reference** — pinned dependency versions from the TA eval repo's `pyproject.toml`, useful for matching the grading environment: `torch==2.12.1`, `transformers==5.16.1`, `gptqmodel==7.3.2`, `optimum==2.3.0`, `accelerate==1.14.0`, `datasets==5.0.1`, `huggingface-hub==1.29.0`, `peft==0.18.0`.

## 9. Known Technical Issues

vLLM does not currently register a **text-only** `Qwen3_5ForCausalLM` class — only the multimodal `Qwen3_5ForConditionalGeneration`. Loading a text-only checkpoint with the multimodal class causes a weight-prefix mismatch. This is directly relevant here since the base model is confirmed Qwen3.5-4B and the TA eval repo does serve it via vLLM. If your `decompress.py` prunes the vision tower entirely (§1) and produces a genuinely text-only checkpoint, loading it through vLLM may hit this exact registration gap — worth testing empirically once you have a pruned checkpoint, rather than assuming either way (vLLM's `--language-model-only` flag is documented for the *original* multimodal checkpoint, not necessarily for an already-pruned one).

## 10. Resources

- Compute: Kaggle, Google Colab, Molab.
- Course site: https://apd10.github.io/CS6013_Fall26/
- TA evaluation repo: https://github.com/lavinianongbri/cs6013 (still being built out — compression-checking script not yet published)
- Leaderboard: https://apd10.github.io/CS6013_Fall26/leaderboard/index.html

---

## 11. Open Questions

1. ❓ **Compression-% / size-metric formula.** Not yet published by the TAs — §3's formula is our own working assumption, not TA-confirmed.
2. ❓ **Whether a pruned, text-only checkpoint hits the vLLM `Qwen3_5ForCausalLM` registration gap** (§9) — test empirically once you have a pruned checkpoint.
