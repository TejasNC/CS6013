# CLAUDE.md — CS6013 Efficient AI Compression Pipeline

Orientation file for Claude Code. Full rules live in
`docs/PROJECT_SOURCE_OF_TRUTH.md` — read that before making decisions that
touch grading (folder names, base model, eval params, submission mechanics).
This file is the map; that file is the law.

## What this repo is

A model-compression project for IIT Bombay's CS6013 course. Each week we
submit a compressed version of `Qwen/Qwen3.5-4B` (a multimodal 4B-param
text backbone + vision tower) under a graded folder/naming convention, to
two tracks (size-only, and CUDA-speedup-eligible), at three compression
targets (10%/20%/40% of original size).

## Critical rules — read before touching anything

1. **Base model is `Qwen/Qwen3.5-4B`, not `Qwen/Qwen3-4B-Instruct-2507`.**
   The old starter template (now removed from this repo) defaulted to the
   latter — it's wrong. If you see that string anywhere, it's a leftover
   bug, not intentional.
2. **Folder structure under `23B0932/` is graded.** Wrong naming there is an
   automatic zero for that submission, per the course's Autumn PDF. Don't
   restructure it without checking `docs/PROJECT_SOURCE_OF_TRUTH.md` §6
   first. The convention is `23B0932/Week<N>/Compression_<target>/Submission<NN>/`
   — no "Track" segment in the path (see `23B0932/CLAUDE.md` for how we
   handle that gap).
3. **Never modify a submission folder that's already been submitted via the
   Google Form.** The course rules treat post-submission edits as a hard
   violation. `23B0932/Week01/Track1_10/Submission01/` is left in its
   original (pre-reorg) state for exactly this reason — its submission
   status was unconfirmed as of the last reorg. Check with Tejas before
   renaming, editing, or deleting anything already under `23B0932/`.
4. **`pipeline/common.py` and `pipeline/run_eval.py` are copied verbatim
   from the TA's actual grading repo** (`lavinianongbri/cs6013`). Treat
   them as ground truth, not as this project's code to refactor — if they
   need a fix, prefer patching around them (or re-diffing against upstream)
   over rewriting their internals, so it stays obvious what came from the
   TAs vs. what's ours.
5. **The compression-% formula in `pipeline/size_check.py` is our own
   working assumption**, not TA-confirmed (their own "Check Compression"
   script is unpublished as of the last check — see source-of-truth doc
   §3). Don't treat its output as gospel; treat it as the best local
   estimate we have.
6. **Vision-tower pruning (`compression/vision_prune.py`,
   `compress.py --prune-vision`) is experimental and off by default.**
   Whether a vision-pruned checkpoint still loads through vLLM's
   multimodal serving class is an open question (source-of-truth doc §11).
   Don't flip it on for a real submission without testing the vLLM load
   path first.

## Directory map

| Path | Purpose |
|---|---|
| `23B0932/` | Actual graded submissions. See `23B0932/CLAUDE.md`. |
| `templates/submission_template/` | Copy this to start a new week's submission. See its `CLAUDE.md`. |
| `pipeline/` | Local dev tooling: dataset setup, size check, eval, orchestration. See `pipeline/CLAUDE.md`. |
| `configs/` | Eval configs (sampling params, dataset path). See `configs/CLAUDE.md`. |
| `docs/PROJECT_SOURCE_OF_TRUTH.md` | Full project rules, with source citations and open questions. Read this for anything grading-related. |

## Common workflows

**Start a new submission:**
```bash
cp -r templates/submission_template 23B0932/Week<N>/Compression_<target>/Submission01
# edit compression/technique.py and decompression/technique.py with the real recipe
```

**Run the full local pipeline for a submission:**
```bash
python pipeline/run_pipeline.py --submission 23B0932/Week<N>/Compression_<target>/Submission01 --limit 20
```
This compresses, checks size %, decompresses, serves via vLLM, and evaluates
using the same logic as the TA harness.

**One-time dataset setup:**
```bash
python pipeline/setup_dataset.py
```

## Current known gaps (don't silently resolve these — flag them)

- Whether `23B0932/Week01/Track1_10/Submission01` has been submitted (blocks migrating it to the new folder convention).
- The compression-% formula (§3 of the source-of-truth doc) is unconfirmed by the TAs.
- Whether the Track distinction needs to be encoded somewhere machine-readable beyond each submission's own README (see `23B0932/CLAUDE.md`).
