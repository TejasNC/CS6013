"""Budget-aware mixed-precision affine quantization.

Track 1 has no CUDA-speedup requirement, so this is a plain "fake-quant"
scheme: weights are packed to arbitrary bit-widths for the *compressed
artifact* (what size_check.py measures), and decompression dequantizes back
to bf16 for serving (see decompression/technique.py) -- there's no packed
GPU kernel involved, matching the project source-of-truth doc's note that
restore dtype isn't pinned to one value.

Technique
---------
Every tensor with ndim >= 2 (i.e. every Linear/Embedding weight matrix; this
is essentially the whole checkpoint -- 1-D tensors like LayerNorm
weights/biases are <0.02% of this model's total bytes) gets row-wise
(per-output-channel) asymmetric affine quantization:

    scale = (max - min) / (levels - 1)     # per output row
    zero  = min                            # per output row
    q     = round((w - zero) / scale)      # in [0, levels-1]
    w'    = q * scale + zero               # dequantized reconstruction

`q` is then bit-packed (MSB-first) to exactly `bits` bits/element -- not
rounded up to a byte -- via `pack_bits`/`unpack_bits`, so odd bit-widths
(3, 6, ...) genuinely save bytes rather than silently costing a full byte
each.

Rather than one uniform bit-width, bits are allocated per-tensor by
`allocate_bits`: a greedy "waterfill" that starts every tensor at the
minimum bit-width and repeatedly upgrades the *smallest* remaining tensors
by one tier first, spending the byte budget on the tensors where an extra
bit is cheapest. This lets a single budget-driven policy hit any target
percentage (not just budgets that land on a whole bit-width), and biases
precision toward smaller tensors within the fixed budget rather than an
arbitrary/uniform cutoff.

1-D tensors are left at their original dtype untouched (negligible size,
and quantizing LayerNorm scale/shift is known to be disproportionately
damaging for very little byte savings).
"""

from __future__ import annotations

import numpy as np
import torch

_BIT_TIERS = (1, 2, 3, 4, 6, 8)
_SAFETY_MARGIN = 0.97  # target slightly under budget to absorb scale/zero + packing overhead


def pack_bits(q: np.ndarray, bits: int) -> np.ndarray:
    """Pack a 1-D array of uint32 values in [0, 2**bits - 1] to `bits` bits/element (MSB-first)."""
    if bits == 8:
        return q.astype(np.uint8)
    bit_positions = np.arange(bits - 1, -1, -1, dtype=np.uint32)
    bitmat = ((q[:, None] >> bit_positions) & 1).astype(np.uint8)
    return np.packbits(bitmat.reshape(-1))


def unpack_bits(packed: np.ndarray, bits: int, n: int) -> np.ndarray:
    """Inverse of pack_bits: recover `n` uint32 values from a `bits`-bit-packed buffer."""
    if bits == 8:
        return packed[:n].astype(np.uint32)
    bitmat = np.unpackbits(packed)[: n * bits].reshape(n, bits)
    weights = (1 << np.arange(bits - 1, -1, -1, dtype=np.uint32))
    return (bitmat.astype(np.uint32) * weights).sum(axis=1)


def allocate_bits(numels: list[int], budget_bits: float, tiers: tuple[int, ...] = _BIT_TIERS) -> list[int]:
    """Greedily assign a bit-width per tensor (from `tiers`) to fit under `budget_bits` total.

    Every tensor starts at tiers[0] (the floor); remaining budget is spent
    upgrading the smallest tensors first, one tier at a time, round-robin,
    until nothing more fits.
    """
    n = len(numels)
    tier_idx = [0] * n
    bits = [tiers[0]] * n
    used = float(sum(numels)) * tiers[0]
    if used > budget_bits:
        raise ValueError(
            f"Budget too small even at the minimum bit-width ({tiers[0]}-bit): "
            f"need {used:,.0f} bits, have {budget_bits:,.0f}. Target percentage is infeasible "
            "with this quantizer -- would need structural pruning, not just lower precision."
        )
    order = sorted(range(n), key=lambda i: numels[i])  # smallest tensors first
    changed = True
    while changed:
        changed = False
        for i in order:
            if tier_idx[i] + 1 >= len(tiers):
                continue
            next_bits = tiers[tier_idx[i] + 1]
            cost = numels[i] * (next_bits - bits[i])
            if used + cost <= budget_bits:
                used += cost
                bits[i] = next_bits
                tier_idx[i] += 1
                changed = True
    return bits


def _quantize_rows(w: torch.Tensor, bits: int) -> dict:
    orig_shape = tuple(w.shape)
    rows = orig_shape[0]
    flat = w.detach().to(torch.float32).reshape(rows, -1).numpy()

    levels = 2 ** bits
    w_min = flat.min(axis=1)
    w_max = flat.max(axis=1)
    scale = (w_max - w_min) / (levels - 1)
    scale = np.where(scale <= 0, 1.0, scale)  # constant rows: avoid div-by-zero, q will be all-zero

    q = np.round((flat - w_min[:, None]) / scale[:, None]).clip(0, levels - 1).astype(np.uint32)
    packed = pack_bits(q.reshape(-1), bits)

    return {
        "bits": bits,
        "shape": orig_shape,
        "scale": torch.from_numpy(scale.astype(np.float16)),
        "zero": torch.from_numpy(w_min.astype(np.float16)),
        "packed": torch.from_numpy(packed),
        "orig_dtype": str(w.dtype).removeprefix("torch."),
    }


def dequantize_rows(payload: dict) -> torch.Tensor:
    shape = payload["shape"]
    bits = payload["bits"]
    rows = shape[0]
    cols = 1
    for d in shape[1:]:
        cols *= d

    packed = payload["packed"].numpy()
    q = unpack_bits(packed, bits, rows * cols).reshape(rows, cols).astype(np.float32)
    scale = payload["scale"].numpy().astype(np.float32)[:, None]
    zero = payload["zero"].numpy().astype(np.float32)[:, None]
    flat = q * scale + zero

    orig_dtype = getattr(torch, payload["orig_dtype"])
    return torch.from_numpy(flat).to(orig_dtype).reshape(shape)


def compress_state_dict(state_dict: dict[str, torch.Tensor], *, target_pct: float, **kwargs) -> dict:
    """Quantize `state_dict` to hit `target_pct` (e.g. 10.0/20.0/40.0) of its own byte size.

    `target_pct` is evaluated against THIS state_dict's own size (numel *
    itemsize, matching pipeline/size_check.py's formula) -- compress.py
    passes in the full original-checkpoint state_dict (optionally with the
    vision tower already stripped), so that's the correct 100% baseline.
    """
    baseline_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    budget_bytes = _SAFETY_MARGIN * (target_pct / 100.0) * baseline_bytes

    quant_keys = [k for k, v in state_dict.items() if v.dim() >= 2]
    plain_keys = [k for k, v in state_dict.items() if v.dim() < 2]

    plain_bytes = sum(state_dict[k].numel() * state_dict[k].element_size() for k in plain_keys)
    # ~4 bytes/row (fp16 scale + fp16 zero) of metadata overhead per quantized tensor.
    overhead_bytes = sum(state_dict[k].shape[0] * 4 for k in quant_keys)
    quant_budget_bytes = budget_bytes - plain_bytes - overhead_bytes

    numels = [state_dict[k].numel() for k in quant_keys]
    bits_per_tensor = allocate_bits(numels, quant_budget_bytes * 8)

    quant: dict[str, dict] = {}
    for key, bits in zip(quant_keys, bits_per_tensor):
        quant[key] = _quantize_rows(state_dict[key], bits)

    plain = {k: state_dict[k].detach().cpu().contiguous() for k in plain_keys}

    return {"quant": quant, "plain": plain, "target_pct": target_pct}
