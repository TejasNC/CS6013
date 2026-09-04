# CLAUDE.md — pipeline/ (local dev tooling)

Not part of any graded submission — this is our own infrastructure for
developing and self-testing submissions before they go under `23B0932/`.

## Files and where they came from

| File | Origin | Notes |
|---|---|---|
| `common.py` | Copied verbatim from `lavinianongbri/cs6013` (the TA's actual grading repo), `evaluation/common.py` | This is the real grading logic: GPQA Diamond loading, prompt building, `\boxed{}` answer extraction/matching, vLLM chat-completion calls. **Prefer not editing this** — if the TA repo updates, re-diff and re-copy rather than hand-patching, so it's obvious this still matches upstream. |
| `run_eval.py` | Copied verbatim from the same TA repo, `evaluation/run_eval.py` | Entry point: `python run_eval.py --config <yaml>`. Same "don't casually edit" note as `common.py`. |
| `setup_dataset.py` | Ours | Downloads GPQA Diamond (`Idavidrein/gpqa`, config `gpqa_diamond` — gated, needs `huggingface-cli login`) and writes it as `train.parquet` so `common.py`'s `load_dataset(dataset_path, split="train")` can read it back. The exact on-disk format the TAs expect isn't published; this script self-verifies by round-tripping the load, but flag it if eval results look suspicious. |
| `size_check.py` | Ours | Implements the compression-% working formula from `docs/PROJECT_SOURCE_OF_TRUTH.md` §3: `Σ numel(t) * itemsize(t)` over the state dict, compressed vs. the full `Qwen/Qwen3.5-4B` checkpoint (vision tower included) as baseline. Not TA-confirmed. |
| `run_pipeline.py` | Ours | Orchestrates a submission end-to-end: shells out to that submission's `compress.py` → `size_check.py` → `decompress.py` → starts a vLLM server on the restored checkpoint → runs `run_eval.py` against it. Talks to submissions only through their CLI contract (`--model_name`/`--checkpoint_path`/`--output_path`), never by importing their internals — this should keep working for submissions with completely different internal compression code. |

## If the TA eval repo changes

`common.py` / `run_eval.py` here are a snapshot. If
`https://github.com/lavinianongbri/cs6013` gets updated (e.g. they finally
publish "Check Compression," or change eval params), re-pull those two
files and diff against what's here — don't assume this snapshot stays
correct indefinitely. `docs/PROJECT_SOURCE_OF_TRUTH.md` §11 tracks what's
still unconfirmed/likely to change.
