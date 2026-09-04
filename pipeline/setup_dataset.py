#!/usr/bin/env python3
"""Download and locally cache the GPQA Diamond dataset for local eval runs.

The TA eval repo's ``run_eval.py`` calls ``datasets.load_dataset(dataset_path,
split="train")`` (not ``load_from_disk``), so this script writes a single
``train.parquet`` file into the target directory rather than using
``Dataset.save_to_disk()`` — the generic ``datasets`` loader auto-detects
parquet files and infers the split from the filename ("train"), so
``load_dataset("datasets/gpqa_diamond", split="train")`` reads it back
correctly. This matches the TA repo's own commented-out example path
(``submission_evaluation/sample_evaluation/datasets/gpqa_diamond/``), but the
exact on-disk format they use internally isn't published, so this is a
best-effort reconstruction — verified by round-tripping (see main()).

GPQA is a gated dataset on the Hub. Before running this script:
  1. Accept the terms at https://huggingface.co/datasets/Idavidrein/gpqa
  2. huggingface-cli login   (or set HF_TOKEN in your environment)

Usage
-----
    python pipeline/setup_dataset.py                 # gpqa_diamond -> ./datasets/gpqa_diamond
    python pipeline/setup_dataset.py --out-dir /path  # custom location
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="Idavidrein/gpqa",
        help="HF Hub dataset id (default: the real GPQA source, gated)",
    )
    parser.add_argument(
        "--config",
        default="gpqa_diamond",
        help="Dataset config/subset name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to fetch (GPQA only ships a 'train' split)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("datasets/gpqa_diamond"),
        help="Local directory to write train.parquet into",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Missing dependency: pip install datasets", file=sys.stderr)
        raise SystemExit(1)

    print(f"Downloading {args.dataset} [{args.config}] split={args.split} ...")
    try:
        dataset = load_dataset(args.dataset, args.config, split=args.split)
    except Exception as exc:  # noqa: BLE001 - want a clear message for gated-access errors
        print(
            "Failed to download the dataset. If this is a gated-access error, "
            f"accept the terms at https://huggingface.co/datasets/{args.dataset} "
            "and run `huggingface-cli login` (or set HF_TOKEN), then retry.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Loaded {len(dataset)} rows, columns: {dataset.column_names}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / f"{args.split}.parquet"
    dataset.to_parquet(out_file)
    print(f"Wrote {out_file}")

    # Round-trip check: this is exactly how pipeline/run_eval.py will load it.
    reloaded = load_dataset(str(args.out_dir), split=args.split)
    assert len(reloaded) == len(dataset), (
        f"Round-trip row count mismatch: wrote {len(dataset)}, "
        f"reloaded {len(reloaded)} — the load_dataset(dir, split=...) "
        "convention may not be what the TA harness actually expects. "
        "Check with the TAs before relying on this for grading-adjacent work."
    )
    print(
        f"Verified: load_dataset('{args.out_dir}', split='{args.split}') "
        f"round-trips to {len(reloaded)} rows. Set dataset_path in your "
        f"eval config to: {args.out_dir}"
    )


if __name__ == "__main__":
    main()
