# CLAUDE.md — 23B0932/ (graded submissions)

Everything under here is what actually gets graded. Treat structure and
naming as load-bearing, not stylistic.

## Convention

```
23B0932/
  Week<N>/
    Compression_<target>/      # target is 10, 20, or 40 (percent of original size)
      Submission<NN>/          # NN is a 2-digit counter, e.g. Submission01
        compress.py
        decompress.py
        compression/
        decompression/
        pyproject.toml
        README.md
```

This is the Autumn PDF's convention (see `docs/PROJECT_SOURCE_OF_TRUTH.md`
§6), which superseded an earlier "Guidelines PDF" convention that used a
`Track<n>_<target>/` folder name instead of `Compression_<target>/`. If you
see a `Track1_10`-style folder anywhere under here, it's a leftover from
before that correction — see the note on `Week01/Track1_10/` below.

## The Track gap

The Autumn PDF's own path convention has no segment for Track (1 vs 2),
even though every student submits to both tracks. We couldn't resolve this
from any source, so the working fix is: **each submission's own `README.md`
states its track explicitly** (see the `## Track` section in
`templates/submission_template/README.md`). If two tracks land on the same
`Week<N>/Compression_<target>/`, they currently need distinct `Submission<NN>`
numbers to avoid colliding on disk — there is no TA-confirmed rule for
exactly how that should look. Flag this rather than inventing a fix if it
comes up.

## `Week01/Track1_10/Submission01/`

Pre-reorg artifact, deliberately left untouched. It:
- uses the old `Track1_10` folder-naming convention (now superseded)
- uses `code.py` with `convert_from_hf_checkpoint`/`convert_to_hf_checkpoint`
  (the deprecated starter-repo convention, superseded by
  `compress.py`/`decompress.py`)
- hardcodes `Qwen/Qwen3-4B-Instruct-2507` as the base model (wrong — should
  be `Qwen/Qwen3.5-4B`)

**Do not rename, restructure, or edit this folder without first confirming
whether it was already submitted via the course's Google Form.** If it
wasn't submitted yet, it should be migrated to
`Week01/Compression_10/Submission01/` using the current template and the
correct base model. If it was submitted, it stays exactly as-is — instead,
create a fresh, correctly-structured submission for the next iteration.
