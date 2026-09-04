#!/usr/bin/env python3
"""Self-evaluate a compressed checkpoint's size against the working formula.

The TAs have not yet published their official "Check Compression" script
(their eval repo's README still says "Will update soon" as of this check).
Until they do, this implements our own working assumption, documented in
the project source-of-truth doc:

    compression % = size(compressed state dict) / size(original state dict) * 100
    size(state dict) = sum over all tensors of numel(tensor) * itemsize(tensor.dtype)

i.e. raw parameter byte-count at each tensor's actual stored dtype — not a
serialized file size (no safetensors headers/metadata, no gzip). The
baseline is the FULL Qwen/Qwen3.5-4B checkpoint as downloaded (vision tower
included), so pruning the vision tower counts toward your compression
budget rather than being done "for free" before the baseline is drawn.

Usage
-----
    # Against the live default baseline (downloads Qwen/Qwen3.5-4B once):
    python pipeline/size_check.py --compressed path/to/compressed.pt

    # Against an HF checkpoint directory instead of a torch.save .pt file:
    python pipeline/size_check.py --compressed path/to/restored_hf_dir

    # Cache the baseline size so repeated runs don't re-download the model:
    python pipeline/size_check.py --compressed compressed.pt --baseline-cache baseline_bytes.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import torch


def state_dict_size_bytes(state_dict: Mapping[str, torch.Tensor]) -> int:
    """Sum of numel * itemsize over every tensor in the state dict."""
    total = 0
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor):
            total += tensor.numel() * tensor.element_size()
    return total


def size_of_path(path: Path) -> int:
    """Compute state-dict size for either a torch.save .pt file or an HF checkpoint dir."""
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        if not isinstance(state_dict, dict):
            raise ValueError(f"Could not find a state_dict in {path}")
        return state_dict_size_bytes(state_dict)

    if path.is_dir():
        # HF checkpoint directory: load via transformers so dtype casts,
        # tied weights, and safetensors sharding are all handled correctly
        # rather than re-implemented here.
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        size = state_dict_size_bytes(model.state_dict())
        del model
        return size

    raise FileNotFoundError(path)


def baseline_size_bytes(model_name: str, cache_path: Path | None) -> int:
    if cache_path is not None and cache_path.is_file():
        cached = int(cache_path.read_text().strip())
        print(f"[baseline] using cached size from {cache_path}: {cached:,} bytes")
        return cached

    from transformers import AutoModelForCausalLM

    print(f"[baseline] loading {model_name} to compute original size (one-time)...")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    size = state_dict_size_bytes(model.state_dict())
    del model

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(str(size))
        print(f"[baseline] cached to {cache_path}")

    return size


def fmt_bytes(n: int) -> str:
    return f"{n:,} bytes ({n / 1024**3:.4f} GiB)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compressed",
        type=Path,
        required=True,
        help="Path to the compressed checkpoint (.pt state_dict, or an HF checkpoint dir)",
    )
    parser.add_argument(
        "--baseline-model",
        default="Qwen/Qwen3.5-4B",
        help="Original model id used as the 100%% baseline",
    )
    parser.add_argument(
        "--baseline-cache",
        type=Path,
        default=None,
        help="Optional file to cache the baseline byte count in, to skip re-downloading",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    original_bytes = baseline_size_bytes(args.baseline_model, args.baseline_cache)
    compressed_bytes = size_of_path(args.compressed)

    pct = 100.0 * compressed_bytes / original_bytes

    print()
    print(f"Original  ({args.baseline_model}): {fmt_bytes(original_bytes)}")
    print(f"Compressed ({args.compressed}):    {fmt_bytes(compressed_bytes)}")
    print(f"Compression %: {pct:.2f}%  (target buckets: 10% / 20% / 40%)")


if __name__ == "__main__":
    main()
