"""Inverse of compression/technique.py.

Dequantizes every row-wise-quantized tensor back to its original dtype
(bf16) at full shape, and passes the untouched 1-D tensors straight
through. The result is loaded into the original architecture via
`model.load_state_dict(..., strict=False)` in decompress.py -- `strict=False`
only matters here if compress.py was run with `--prune-vision`, in which
case vision-tower keys are simply absent from `compressed` and the base
model's own (uncompressed) vision-tower weights are kept, per this repo's
`compress.py --prune-vision` convention.
"""

from __future__ import annotations

import torch

from compression.technique import dequantize_rows


def decompress_state_dict(
    compressed: dict,
    original_state_dict: dict[str, torch.Tensor],
    **kwargs,
) -> dict[str, torch.Tensor]:
    quant = compressed["quant"]
    plain = compressed["plain"]

    restored: dict[str, torch.Tensor] = {}
    for key, payload in quant.items():
        restored[key] = dequantize_rows(payload)
    for key, tensor in plain.items():
        restored[key] = tensor

    # Anything not present in `compressed` at all (e.g. a vision tower
    # pruned out by compress.py --prune-vision) falls back to the
    # original, uncompressed value so load_state_dict(strict=False) still
    # gets a complete, working model.
    for key, tensor in original_state_dict.items():
        restored.setdefault(key, tensor)

    return restored
