# CLAUDE.md — configs/

## `eval_config.yaml`

Local eval config consumed by `pipeline/run_eval.py`. The sampling/decoding
parameters (`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
`repetition_penalty`, `max_new_tokens`, `enable_thinking`) are copied
**verbatim** from the TA eval repo's committed `config.yaml` — these are
confirmed-exact values the leaderboard grader uses (see
`docs/PROJECT_SOURCE_OF_TRUTH.md` §8.1), not something to tune. If you need
different sampling for an experiment, copy this file rather than editing it
in place, so the canonical one stays matched to the TAs'.

`model_name`, `vllm_base_url`, `output`, and `limit` are the fields meant to
be overridden per-run — `pipeline/run_pipeline.py` does this automatically
when orchestrating a submission; edit them by hand only for one-off manual
`run_eval.py` invocations.

`dataset_path` points at whatever `pipeline/setup_dataset.py` produced
locally. If you re-run dataset setup with a different `--out-dir`, update
this to match.
