"""Token frequency over calibration (non-KL-slice) HMMT teacher traces -> npy used for embedding tiers."""
import argparse, numpy as np
from transformers import AutoTokenizer
from .data import build
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
seqs, _ = build("cal", tok, 10**9)
c = np.zeros(248320, dtype=np.int64)
for ids, _ in seqs: c += np.bincount(ids.numpy(), minlength=248320)[:248320]
np.save(a.out, c); print("sequences", len(seqs), "tokens", c.sum(), "distinct", (c > 0).sum())
