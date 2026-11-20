#!/usr/bin/env python3
"""Measure a compressed checkpoint the way the TAs' MidSem script does.

Source: `CS6013Fall_ProjectEvalMidSemScript/measure_checkpoint_bits.py` and
`eval_common.sh` in https://github.com/lavinianongbri/cs6013 (commit 0a7f1840,
read 2026-11-20). Not yet confirmed that the FINAL evaluation uses it unchanged.

    bits(tensor)   = numel * element_size * 8        (stored dtype, e.g. uint8 blobs = 8 bits)
    size_text_GiB  = sum of bits over all tensors NOT in the vision tower / 8 / 2**30
    size_frac      = size_text_GiB / 8.0585          (hard-coded original text-tower GiB)

A tensor is "visual" if its name contains "visual." or has a path segment in
{visual, vision, vision_tower, vision_model}; visual tensors are NOT counted
(the TA harness zeroes them before decompress). Tokenizer/config files do not
count. Tied embeddings / lm_head count as text.

This reads safetensors headers only (pure Python, no torch needed), so it can
run on the HF-repo folder exactly as it would be submitted.

Targets: 20% -> <= 1.6117 GiB, 10% -> <= 0.8059 GiB. Keep a safety margin.

Usage
-----
    python pipeline/size_check.py --compressed path/to/hf_checkpoint_dir
    python pipeline/size_check.py --compressed path/to/dir --json
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

ORIGINAL_TEXT_GIB = 8.0585  # from eval_common.sh (ORIGINAL_TEXT_GB)

# safetensors dtype -> bits
DTYPE_BITS = {
    "F64": 64, "I64": 64, "U64": 64,
    "F32": 32, "I32": 32, "U32": 32,
    "F16": 16, "BF16": 16, "I16": 16, "U16": 16,
    "F8_E4M3": 8, "F8_E5M2": 8, "I8": 8, "U8": 8, "BOOL": 8,
}
VISUAL_SEGMENTS = {"visual", "vision", "vision_tower", "vision_model"}


def is_visual(name: str) -> bool:
    return "visual." in name or any(seg in VISUAL_SEGMENTS for seg in name.split("."))


def tensor_bits(path: Path):
    """Yield (name, bits) for every tensor in one .safetensors file."""
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        numel = 1
        for d in meta["shape"]:
            numel *= d
        yield name, numel * DTYPE_BITS[meta["dtype"]]


def measure(ckpt_dir: Path) -> dict:
    files = sorted(ckpt_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {ckpt_dir}")
    text_bits = visual_bits = 0
    for fp in files:
        for name, bits in tensor_bits(fp):
            if is_visual(name):
                visual_bits += bits
            else:
                text_bits += bits
    text_gib = text_bits / 8 / 2**30
    return {
        "text_gib": text_gib,
        "visual_gib": visual_bits / 8 / 2**30,
        "size_frac": text_gib / ORIGINAL_TEXT_GIB,
        "avg_bits_per_original_weight": 16 * text_gib / ORIGINAL_TEXT_GIB,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compressed", type=Path, required=True, help="HF checkpoint dir with *.safetensors")
    ap.add_argument("--json", action="store_true")
    # Accepted but ignored, so older callers (run_pipeline.py) keep working.
    ap.add_argument("--baseline-model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--baseline-cache", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    r = measure(args.compressed)
    if args.json:
        print(json.dumps(r))
        return
    print(f"Text tower:   {r['text_gib']:.4f} GiB (counted)")
    print(f"Vision tower: {r['visual_gib']:.4f} GiB (not counted)")
    print(f"size_frac = {r['text_gib']:.4f} / {ORIGINAL_TEXT_GIB} = {100 * r['size_frac']:.2f}%")
    print("Budgets: 20% <= 1.6117 GiB, 10% <= 0.8059 GiB (keep a margin)")


if __name__ == "__main__":
    main()
