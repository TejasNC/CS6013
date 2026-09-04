"""Your compression technique goes here.

``compress_state_dict`` receives the model's state_dict (already with the
vision tower optionally removed by compress.py's --prune-vision) and must
return a new dict to be torch.save'd. This placeholder is a no-op
pass-through only so the pipeline runs end-to-end out of the box — replace
it with your actual recipe (quantization, structured pruning, low-rank
factorization, ...) before this counts as a real submission.
"""

from __future__ import annotations

import torch


def compress_state_dict(state_dict: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
    # TODO: replace with your actual compression technique.
    return {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
