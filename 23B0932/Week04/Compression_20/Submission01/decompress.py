#!/usr/bin/env python3
"""Restore a compressed checkpoint to a full HF model directory.

Required CLI per the project source-of-truth doc (§6):
    python decompress.py --model_name <name> --checkpoint_path <compressed artifact> --output_path <restored dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from decompression.technique import decompress_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-4B", help="HF model id used for original shapes")
    parser.add_argument("--checkpoint_path", required=True, help="Path to the compressed artifact from compress.py")
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    compressed = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    original_state_dict = model.state_dict()

    restored = decompress_state_dict(compressed, original_state_dict)
    # strict=False: tolerates a pruned vision tower or other intentionally-dropped keys.
    model.load_state_dict(restored, strict=False)

    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
