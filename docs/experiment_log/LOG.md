# Results log (implementer appends; experiment reviewer keeps a summary at top)

See **REPORT_LOG.md** for the structured experiment + decision log (hypotheses, evidence, incidents, reviewer verdicts) written for the report.

## Reviewer summary (updated Fri 20 Nov 20:50 IST)
- Incumbents: none frozen yet. 20% candidate = A2 (new embedding tiers, ~19.65%); 10% candidate = B2 (~9.8%). B1_v0 is over the size margin (9.90% > 9.85%) and is out.
- bf16 baseline HMMT 34.0 +- 6.3 (k=4), length 27.9k, 32k-cap 66.5%. Cap-rate gate for compressed models: <= 71.5%.
- Embedding tiers drove almost all of A1's KL (body-only 0.0065 vs total 0.089). Use T1 16k@5b + T2 82k@1.25b for 20%, T1 16k@4b + T2 82k@1.0b for 10% unless body KL says otherwise.
- 20% body is already near the KL floor, so spend remaining effort on 10%.
- Overthinking is the biggest accuracy lever: nearly all correct answers come from the 33% of samples that finish. KD on short correct traces is the priority.
- Fri 21:30 update: A2 (19.65%, KL 0.0251) frozen as 20% incumbent. B2 KL 0.558, out; B3 (reverse water-filling) building. New hard stop Sat 22:00 IST; KD start by Sat 06:00. LoRA r=16 costs ~0.36 pt of size: reserve it. Traces sufficient (60, 1.45M tok); GPU 0 goes to the TA-prompt baseline.
- Sat 00:50 update: TA-prompt baseline 44.0 +- 6.8, 52% cap (matches leaderboard); cap gate <= 57%. 10% candidate B3c (base 9.56%, KL 0.303) + LoRA r=8 -> ~9.75%; KD running on GPU 3 with thin data (27 seqs) -> recommended restart with all 60 traces + val holdout. Plan: evaluate 4 h KD checkpoint on GPU 1 as fallback; final 10% decision after final KD eval (~Sat 10:00).
- Sat 05:05 update: 20% FINAL = A2 (HMMT k=2 42.0 +- 6.8 vs 44.0, cap 52%, 19.65%). 10% scalar models COLLAPSED in free running (phase 1: 100% capped, 1/30 correct despite KL 0.156). Direction: free-run screen (30 prompts x 3k tokens, loop fraction) for all 10% candidates, B4 vector quantizer gated by a one-layer test by Sat 08:00, KD redone on it Sat 12:00-16:00. Fallback: best scalar 10% model (phase 1/2).
- Evals: HMMT k=2 for finalists only (gate on cap rate/length; 2-SE rule is unreachable at k=2, so ties go to lower KL).

Columns: run | KL (held-out teacher-forced) | HMMT mean/SE (k) | AIME (k) | mean len (tok) | 32k-cap rate | size (GiB, frac of 8.0585) | notes

| run | KL | HMMT | AIME | mean len | cap rate | size | notes |
|---|---|---|---|---|---|---|---|
| bf16 baseline (S3) | - | 34.0 ± 6.3 (k=4, 50 problems, hmmt_feb_50, our prompt) | pending (k=2 running) | 27940 | 66.5% | 8.0585 GiB = 100% | 200 gens via pipeline/gen_eval.py (per-sample seeds). Heavy overthinking: two thirds hit the cap. Leaderboard baseline 43.5 used the TA prompt/single sample. |
| A1_v0 (20%, count-tiered embed) | 0.0893 (older slice) / body-only 0.0065 | - | - | - | - | 1.5839 GiB = 19.66% | Hadamard+GPTQ+rANS, FFN 3.33 / attn 5 / dnet 4 bits. Embedding T3 (1024 centroids) causes ~all KL. |
| B1_v0 (10%) | not yet | - | - | - | - | 0.7980 GiB = 9.90% (over 9.85% margin) | FFN 1.48 / attn 3 / dnet 2 bits; retune pending |
| bf16 baseline, TA prompt (S3b) | - | 44.0 ± 6.8 (k=2, 50 problems) | - | 25988 | 52.0% | 100% | 100 gens. Matches leaderboard 43.5. Cap-rate gate for candidates: <= 57%. Paired vs first-prompt run: +10.0 pts (2.3 SE), so the "concise" instruction matters. |
| A2 (20% incumbent) | 0.0251 | - | - | - | - | 1.5834 GiB = 19.65% | FFN 3.30 / attn 5 / dnet 4; embed T1 16k@5b + T2 82k@1.25b; restore + vLLM 33k load OK; body-only floor 0.0071 |
| B2 (10%) | 0.558 | - | - | - | - | 0.7893 GiB = 9.80% | uniform bits; out |
| B3 (10%) | 0.374 | - | - | - | - | 0.7900 GiB = 9.80% | bit allocation from ablation (reverse water-filling) |
| B3b (10%) | 2.066 | - | - | - | - | 0.7730 GiB = 9.59% | second water-filling round overshot (dnet.a 0.85, down.a 1.15 bits): early layers collapse below ~1.5 bits. Discarded. |
| B3c (10% base, hess_v1) | 0.303 | - | - | - | - | 0.7706 GiB = 9.56% | B3 with small cuts to band-a groups; better Hessians (48 seqs). With LoRA r=8 int8 (+~0.19%) expect ~9.75%. KD running. |
| A2 + smoke LoRA r=16 (34 steps) | 0.0238 | - | - | - | - | 20.03% (LoRA r=16 costs +0.38%) | merge + int8 restore path verified; r=16 too big for 20% target, not used |
| KD phase 1 (B3c + LoRA r=8, GPU 3 shared) @ 01:07 | val KL 0.324 → 0.256 after 9 steps (34 min); val = 3 held-out problems, 12 seqs | - | - | - | - | est. ~9.75% with LoRA | ~16 steps/hour (≈57 s per 6144-token sequence on contended GPU 3). Still running. |
| KD phase 1 final (B3c + LoRA r=8, 40 steps / 2.7 h on shared GPU 3) | val KL 0.324 → 0.186 (-43%); frozen-slice KL pending | pending | - | - | - | packed 0.7869 GiB = 9.76% | adapters: emir_data/phase1_best_adapters_final.pt. Packed dir ckpt/B3c_p1; restore + eval pending. |
| KD phase 2 (end-windows + plain 4096, warm start) | val KL start 0.186; gate <= 0.198 | - | - | - | - | same | running on GPU 3 since 03:35 |
| A2 final gate (20%) | 0.0251 | 42.0 ± 6.8 (k=2; paired vs bf16 -2.0 pts, -0.57 SE) | - | 25404 | 52.0% | 1.5834 GiB = 19.65% | restore OK, vLLM 33k load OK. Passes: size, cap <= 57%, score within noise of 44.0. FINAL 20% unless something better lands. |
| B3c + phase-1 LoRA (10%) | 0.1564 (B3c 0.3033) | pending (k=2 running) | - | - | - | 0.7869 GiB = 9.76% | KD 40 steps; restore OK; vLLM load OK |
| 10% diagnosis (Sat 05:10) | phase1 HMMT first 33 gens: 100% capped, 2.5% | - | - | 32000 | 100% | - | Stop-token gate (teacher-forced, 8 finished correct traces, tail windows): teacher P(</think>)=0.12/P(<\|im_end\|>)=0.90 rank 0; A2 0.11/0.84 rank 0; B3c plain 0.022/0.0036 rank 3-16; phase1 0.035/0.0094 rank 2-4. Exact special-token rows do not help (0.0052). |
| phase-2 adapters on B3c, held-out problems only (hmmt_feb_2025-5, -11, -12; never in KD train/end-window data) | - | - | - | - | - | 9.76% | P(</think>)=0.0998 (teacher 0.148), P(<\|im_end\|>)=0.8215 (teacher 0.966), rank 0 on 8/8. The 3 problems are `val_probs` in kd.py (random.Random(1234) shuffle of problems, first 3); train data and end-windows both skip them. |
| RVQ one-layer test (layers.10 up/down) | - | - | - | - | - | - | rotated MSE, 2-stage dim-8 RVQ vs entropy-coded scalar (RTN) at equal bits: 1.0 b/w +27%, 1.5 b/w +10%, 2.0 b/w -6% (scalar better). Below the 15% gate at 1.5 b; only the ~1-bit groups benefit. |

Rebuild recipe (fallbacks, after any NAS cleanup): base B3c = `python -m compress_lib.build --model <Qwen3.5-4B dir> --out B3c_v0 --plan plans/B3c.json --hess hess_v1.pt --counts tok_counts_v0.npy` (hess_v1: `python -m compress_lib.hess --model <dir> --out hess_v1.pt --nseq 48 --maxlen 8192`; counts: `python -m compress_lib.counts`). Adapters: emir_data/phase1_best_adapters_final.pt (phase 1), emir_data/phase2_best_adapters.pt (phase 2, KD: rank 8 alpha 8 lr 1e-4 maxlen 4096 --end-windows init from phase 1). Package: `python -m compress_lib.lora_io --comp B3c_v0 --adapters <adapters.pt> --out B3c_p2`; restore: `python -m compress_lib.restore --comp B3c_p2 --out B3c_p2_restored`.
| 10% phase 2 final HMMT (B3c + phase-2 LoRA r=8) | frozen-slice KL ~0.15; val KL 0.179 | 9.0 ± 4.0 (k=2) | - | 30122 | 82% | 0.7869 GiB = 9.76% | FAILS cap gate (<=57%). 18/100 finished (9 correct). Frozen as 10% FALLBACK. Restore: python -m compress_lib.lora_io --comp B3c_v0 --adapters phase2_best_adapters.pt --out B3c_p2 ; python -m compress_lib.restore --comp B3c_p2 --out B3c_p2_restored. Base B3c_v0 = compress_lib.build --plan plans/B3c.json (hess_v1, counts v0). |
| hazard / rollout diagnostics (phase 2) | per-position KL on own rollouts 0.087/0.082/0.089 (0-1k/1-4k/4-8k) | - | - | 8186 (8192 cap) | 99% capped at 8192 | - | teacher-forced stop hazard on teacher traces is NOT low (sum early P(</think>) 0.41 vs teacher 0.27); free-run capping is exposure bias. 'Wait' 2.8/1k tokens, repeated 8-gram share 0.174. |
