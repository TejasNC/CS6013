# efficient_ai_CS6013_2026Fall

## Evaluate a submission

```bash
python evaluation/eval_submission.py \
  --submission sample_submission/code.py \
  --model Qwen/Qwen3-4B-Instruct-2507
```

This compresses the model with the submission, restores it to HF format, then
runs GPQA Diamond and MMLU-Pro (default: 10 samples each). Use `--limit 1` for a
smoke test.
