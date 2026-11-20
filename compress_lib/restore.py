"""usage: python -m compress_lib.restore --comp DIR --out DIR   (no model_name, no network)"""
import argparse
from .model import restore_model
ap = argparse.ArgumentParser(); ap.add_argument("--comp", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); restore_model(a.comp, a.out)
