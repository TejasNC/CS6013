#!/usr/bin/env python3
"""Download HMMT Feb 2025 + AIME 2025 and cache them as one local parquet file.

Writes ``<out-dir>/train.parquet`` with columns: example_id, source, problem, answer.

Sources (MathArena, public on the Hub; column names are matched tolerantly):
  - MathArena/hmmt_feb_2025   (30 problems)
  - MathArena/aime_2025       (30 problems; I + II combined)

Usage
-----
    python pipeline/setup_math_dataset.py                       # -> datasets/math_eval
    python pipeline/setup_math_dataset.py --out-dir /some/path
"""

from __future__ import annotations

import argparse
from pathlib import Path

SOURCES = {
    "hmmt_feb_2025": "MathArena/hmmt_feb_2025",
    "aime_2025": "MathArena/aime_2025",
}


def _pick(row: dict, names: tuple[str, ...]) -> str:
    for n in names:
        if n in row and row[n] is not None:
            return str(row[n])
    raise KeyError(f"none of {names} in columns {sorted(row)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("datasets/math_eval"))
    args = ap.parse_args()

    from datasets import Dataset, load_dataset

    rows = []
    for tag, repo in SOURCES.items():
        ds = load_dataset(repo, split="train")
        print(f"{repo}: {len(ds)} rows, columns {ds.column_names}")
        for i, r in enumerate(ds):
            idx = next((r[k] for k in ("problem_idx", "id", "idx") if k in r), i)
            rows.append(
                {
                    "example_id": f"{tag}-{idx}",
                    "source": tag,
                    "problem": _pick(r, ("problem", "question")),
                    "answer": _pick(r, ("answer", "solution_answer")),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(out))
    print(f"Wrote {len(rows)} problems to {out}")


if __name__ == "__main__":
    main()
