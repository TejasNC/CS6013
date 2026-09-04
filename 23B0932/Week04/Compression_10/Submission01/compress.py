#!/usr/bin/env python3
"""Compress an HF checkpoint.

Required CLI per the project source-of-truth doc (§6):
    python compress.py --model_name <name> [--checkpoint_path <local path>] --output_path <path>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from compression.technique import compress_state_dict
from compression.vision_prune import strip_vision_tower

# Week04, Track 1, Compression_10 -- compress to <=10% of the original
# checkpoint's byte size. See compression/technique.py for the technique
# (budget-aware mixed-precision bit-packed quantization).
TARGET_PCT = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-4B", help="HF model id (or local dir)")
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Optional local checkpoint dir to load instead of downloading --model_name",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--prune-vision",
        action="store_true",
        help=(
            "EXPERIMENTAL: strip vision-tower tensors before compressing "
            "(see compression/vision_prune.py). Not yet verified to reload "
            "through vLLM — off by default."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.checkpoint_path or args.model_name

    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    state_dict = model.state_dict()
    del model

    if args.prune_vision:
        state_dict = strip_vision_tower(state_dict)

    compressed = compress_state_dict(state_dict, target_pct=TARGET_PCT)

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": args.model_name, "state_dict": compressed}, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
