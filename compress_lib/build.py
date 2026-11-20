"""usage: python -m compress_lib.build --model DIR --out DIR --plan plans/A1.json [--hess H.pt] [--counts tok_counts.npy]"""
import argparse, json, numpy as np, torch
from .model import compress_model
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--plan", required=True)
ap.add_argument("--dev", default="cuda"); ap.add_argument("--hess", default=None); ap.add_argument("--counts", default=None)
a = ap.parse_args()
plan = json.load(open(a.plan))
H = torch.load(a.hess) if a.hess else None
C = np.load(a.counts) if a.counts else None
compress_model(a.model, a.out, plan, H, C, dev=a.dev)
