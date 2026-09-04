# CS6013 — Efficient AI — Compression Pipeline

Reorganized off the old `skylight-org/efficient_ai_CS6013_2026Fall` starter
template (now confirmed stale — see the project source-of-truth doc). This
repo now tracks the **actual** grading harness
([`lavinianongbri/cs6013`](https://github.com/lavinianongbri/cs6013)) and
the Autumn PDF's submission conventions instead.

## Layout

```
.
├── 23B0932/                      # actual submissions — folder structure/naming is graded, don't improvise
│   └── Week01/
│       └── Compression_<target>/
│           └── Submission01/
├── templates/submission_template/  # copy this to start a new submission
├── pipeline/                     # local dev tooling (not part of any submission)
│   ├── common.py, run_eval.py    #   copied from the TA eval repo — this is the real grading logic
│   ├── setup_dataset.py          #   downloads/caches GPQA Diamond locally
│   ├── size_check.py             #   our working compression-% formula (TAs haven't published theirs yet)
│   └── run_pipeline.py           #   orchestrates: compress -> size check -> decompress -> vLLM serve -> eval
└── configs/eval_config.yaml      # local eval config, mirrors the TA's config.yaml exactly on sampling params
```

## Quickstart

```bash
pip install -e .
# vLLM separately (CUDA-version-specific wheel):
pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

# One-time: cache GPQA Diamond locally (gated dataset — accept terms on HF first)
python pipeline/setup_dataset.py

# New submission: copy the template
cp -r templates/submission_template 23B0932/Week02/Compression_20/Submission01

# Run the full local pipeline for a submission
python pipeline/run_pipeline.py \
  --submission 23B0932/Week02/Compression_20/Submission01 \
  --limit 20
```

`run_pipeline.py` compresses, reports the compression % against our working
formula, decompresses, serves the result via vLLM, and evaluates it with the
*same* `pipeline/run_eval.py` logic as the TA harness — so local numbers are
as close as this repo can get to what the leaderboard will actually compute,
short of the TAs publishing their own compression-checking script.

## Reference

See the project source-of-truth doc for the full rules (base model, tracks,
folder/naming conventions, eval parameters, open questions).
