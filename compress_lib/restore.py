"""usage: python -m compress_lib.restore --comp DIR --out DIR [--dev cpu]  (no model_name, no network)"""
import argparse, torch
from .model import restore_model
ap = argparse.ArgumentParser(); ap.add_argument("--comp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--dev", default=None)
a = ap.parse_args()
dev = a.dev or ("cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 6e9 else "cpu")
restore_model(a.comp, a.out, dev=dev)
