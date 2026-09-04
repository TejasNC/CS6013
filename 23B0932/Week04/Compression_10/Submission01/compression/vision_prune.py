"""Heuristic vision-tower detection/removal for Qwen/Qwen3.5-4B.

Qwen/Qwen3.5-4B is multimodal (~5B params: a 4B text backbone + a ViT). For
this text-only, math-domain project, pruning the vision tower is near-free
compression (see the project source-of-truth doc, §1) — the baseline used
for the compression-% formula includes the vision tower, so removing it
counts toward your 10/20/40% budget.

This does NOT hardcode a specific parameter-name prefix, because that
wasn't independently confirmed against the actual checkpoint — it detects
likely vision-tower keys heuristically by name and prints them so you can
manually confirm the list looks right before trusting it.

Known open question (source-of-truth doc §11): whether a checkpoint with the
vision tower fully removed still loads through vLLM's multimodal
`Qwen3_5ForConditionalGeneration` class — there's no registered text-only
class as of writing. Test empirically before relying on this for a real
submission; that's why compress.py's --prune-vision defaults to off.
"""

from __future__ import annotations

_VISION_NAME_HINTS = ("vision", "visual", "vit", "image_", "mm_projector", "multi_modal")


def find_vision_keys(state_dict: dict) -> list[str]:
    """Return state_dict keys that look like they belong to the vision tower."""
    return [k for k in state_dict if any(hint in k.lower() for hint in _VISION_NAME_HINTS)]


def strip_vision_tower(state_dict: dict, *, verbose: bool = True) -> dict:
    keys = find_vision_keys(state_dict)
    if verbose:
        print(f"[vision_prune] detected {len(keys)} likely vision-tower tensors")
        for k in keys[:10]:
            print(f"    {k}")
        if len(keys) > 10:
            print(f"    ... and {len(keys) - 10} more")
        print(
            "[vision_prune] confirm this list looks right (e.g. print "
            "model.named_modules() and cross-check) before trusting it — "
            "name-based detection is a heuristic, not a confirmed mapping."
        )
    return {k: v for k, v in state_dict.items() if k not in keys}
