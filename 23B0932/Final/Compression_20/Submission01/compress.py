#!/usr/bin/env python3
"""Compress an original Qwen3.5-4B HF checkpoint directory into the compressed checkpoint format."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from compression import attach, compress_model
from compression.hess import collect_hessians

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3.5-4B", help="unused placeholder; weights are read from --checkpoint_path")
    ap.add_argument("--checkpoint_path", required=True, help="original (uncompressed) HF checkpoint directory")
    ap.add_argument("--output_path", required=True, help="output directory for the compressed checkpoint")
    ap.add_argument("--hessians", default=None, help="optional precomputed Hessians (.pt); default: computed from the shipped calibration token ids")
    ap.add_argument("--adapters", default=None, help="unused for this target")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    plan = json.load(open(HERE / "compression/data/plan.json"))
    counts = np.load(HERE / "compression/data/tok_counts.npy")
    hess = torch.load(a.hessians) if a.hessians else collect_hessians(
        a.checkpoint_path, str(HERE / "compression/data/calib_ids.pt"), dev=a.device)
    out = Path(a.output_path)
    if a.adapters:
        tmp = Path(tempfile.mkdtemp(dir=out.parent if out.parent.exists() else None))
        compress_model(a.checkpoint_path, str(tmp), plan, hess, counts, dev=a.device)
        attach(str(tmp), a.adapters, str(out))
        shutil.rmtree(tmp)
    else:
        compress_model(a.checkpoint_path, str(out), plan, hess, counts, dev=a.device)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
