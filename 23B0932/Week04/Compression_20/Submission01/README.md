# Submission — Week 4, Compression 20%, Track 1

Per the project source-of-truth doc: this README documents setup/usage/repro,
not the compression technique itself (that goes in the report).

## Track

Track: 1 (checkpoint size only — no CUDA-speedup eligibility claimed)

## Technique (summary — see the report for the full writeup)

Budget-aware mixed-precision, bit-packed affine quantization. Every weight
tensor with `ndim >= 2` (i.e. essentially the whole checkpoint — 1-D tensors
like LayerNorm weight/bias are left untouched, bf16) gets row-wise
(per-output-channel) asymmetric affine quantization; a greedy "waterfill"
(`compression/technique.py::allocate_bits`) assigns each tensor a bit-width
from `{1,2,3,4,6,8}` so the whole checkpoint fits under the byte budget for
this target, spending extra bits on the smallest tensors first. For a 20%
target this lands close to a uniform ~3 bits/weight. `decompress.py`
dequantizes back to `torch.bfloat16` for serving — this is a fake-quant
scheme (no packed GPU kernel), matching Track 1's "no CUDA-speedup required"
scope.

## Setup

```bash
pip install -e .
```

## Usage

```bash
python compress.py --model_name Qwen/Qwen3.5-4B --output_path compressed.pt
python decompress.py --model_name Qwen/Qwen3.5-4B --checkpoint_path compressed.pt --output_path restored_hf/
```

Or, from the repo root, run the full local pipeline (compress → size check →
decompress → vLLM serve → eval):

```bash
python pipeline/run_pipeline.py --submission 23B0932/Week04/Compression_20/Submission01 --limit 20
```

## dtype

`torch.bfloat16` → mixed-precision (~3 bit, per-tensor, bit-packed); dequantized to `torch.bfloat16` on decompress.

## Known caveats

- The compression-% figure is measured against this repo's own working
  formula (`pipeline/size_check.py`), not a TA-published one — see
  `docs/PROJECT_SOURCE_OF_TRUTH.md` §3.
- Vision-tower pruning (`compress.py --prune-vision`) is **not** used here —
  the vision tower is only ~7% of this checkpoint's total bytes (measured
  directly from the safetensors headers), so it isn't worth the unresolved
  vLLM-loading risk (source-of-truth doc §9) for this little budget.
