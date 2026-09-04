# CLAUDE.md — templates/submission_template/

The skeleton to copy for every new submission. Copy the whole directory,
don't reference it in place — each submission needs its own
`compress.py`/`decompress.py`/`pyproject.toml` per the course's repo
requirements (`docs/PROJECT_SOURCE_OF_TRUTH.md` §6).

```bash
cp -r templates/submission_template 23B0932/Week<N>/Compression_<target>/Submission01
```

## What's a placeholder vs. what's real

- `compress.py`, `decompress.py` — **real**, working CLI entry points
  matching the required interface (`--model_name --checkpoint_path
  --output_path`). Load a model, call into `compression/technique.py` /
  `decompression/technique.py`, save the result. Shouldn't need much
  editing beyond wiring in a real technique.
- `compression/technique.py`, `decompression/technique.py` — **placeholder
  only**. Currently a no-op pass-through (`compress_state_dict` just
  detaches/contiguous-copies; `decompress_state_dict` just merges the dicts
  back together). This is where the actual compression recipe for that
  week goes — quantization, structured pruning, low-rank factorization,
  whatever the week's plan calls for. The pass-through exists so the
  pipeline runs end-to-end before you've written the real technique, not
  as a real submission.
- `compression/vision_prune.py` — a real, working utility, but its use is
  **opt-in and experimental** (`compress.py --prune-vision`, default off).
  It heuristically finds vision-tower tensor names by substring match
  (`"vision"`, `"visual"`, `"vit"`, etc.) since the exact parameter-naming
  convention for `Qwen/Qwen3.5-4B`'s vision tower wasn't independently
  confirmed. Always eyeball its printed key list before trusting it on a
  new checkpoint. Whether a vision-pruned checkpoint still loads through
  vLLM's serving class is unconfirmed — see root `CLAUDE.md` rule 6.
- `README.md` — has a `## Track` field. Fill it in; it's the only place
  Track (1 vs 2) is recorded, since the folder path doesn't carry it (see
  `23B0932/CLAUDE.md`).
- `pyproject.toml` — pinned to the TA grading environment's versions where
  known. Add technique-specific deps (e.g. `gptqmodel`) as needed.

## Testing a new submission

Use `pipeline/run_pipeline.py` from the repo root rather than invoking
`compress.py`/`decompress.py` by hand — it also runs the size check and
full eval, not just the compress/decompress step.
