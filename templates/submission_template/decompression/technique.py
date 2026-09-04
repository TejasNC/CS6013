"""Inverse of compression/technique.py — replace alongside it.

``decompress_state_dict`` receives the compressed state_dict and the
ORIGINAL model's state_dict (for shapes/reference) and must return a
state_dict loadable via ``model.load_state_dict(...)`` on the original
architecture. Keys missing from the compressed dict (e.g. a pruned vision
tower) are left out of the update, so the base model's own weights for
those keys are kept unless you delete them from ``original_state_dict``
yourself before calling ``model.load_state_dict(..., strict=False)``.
"""

from __future__ import annotations

import torch


def decompress_state_dict(
    compressed: dict[str, torch.Tensor],
    original_state_dict: dict[str, torch.Tensor],
    **kwargs,
) -> dict[str, torch.Tensor]:
    # TODO: replace with the inverse of your actual compression technique.
    restored = dict(original_state_dict)
    restored.update(compressed)
    return restored
