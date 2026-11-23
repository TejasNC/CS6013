#!/usr/bin/env python3
"""Restore a compressed checkpoint directory to a full bf16 HF checkpoint (offline; reads only --checkpoint_path)."""
import argparse
import torch
from decompression import restore_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3.5-4B", help="unused placeholder (no network access is made)")
    ap.add_argument("--checkpoint_path", required=True, help="compressed checkpoint directory")
    ap.add_argument("--output_path", required=True, help="output directory for the restored bf16 checkpoint")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if >6 GB free, else cpu)")
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 6e9 else "cpu")
    restore_model(a.checkpoint_path, a.output_path, dev=dev)
    print(f"Wrote {a.output_path}")


if __name__ == "__main__":
    main()
