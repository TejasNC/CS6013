"""Track 1 submission (10% target): drop 10% of each weight's last dimension.

This is the file the evaluator (eval_submission.py) actually imports. It must
expose convert_from_hf_checkpoint and convert_to_hf_checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

SPARSITY = 0.10  # 10% compression target


def _chop_last_dim(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Keep the first (1 - sparsity) fraction of tensor along its last dim."""
    if tensor.ndim == 0 or tensor.shape[-1] < 2:
        return tensor.detach().cpu().contiguous()
    keep = round(tensor.shape[-1] * (1.0 - sparsity))
    keep = max(0, min(keep, tensor.shape[-1]))
    return tensor[..., :keep].detach().cpu().contiguous()


def _pad_last_dim(tensor: torch.Tensor, target_last_dim: int) -> torch.Tensor:
    """Pad zeros on the last dimension so shape[-1] == target_last_dim."""
    if tensor.shape[-1] == target_last_dim:
        return tensor
    if tensor.shape[-1] > target_last_dim:
        raise ValueError(
            f"Compressed last dim {tensor.shape[-1]} is larger than "
            f"target {target_last_dim}"
        )
    pad = target_last_dim - tensor.shape[-1]
    return torch.nn.functional.pad(tensor, (0, pad))


def convert_from_hf_checkpoint(
    model_name: str,
    output_path: str,
    sparsity: float = SPARSITY,
) -> None:
    """Load an HF model, drop a fraction of each weight's last dim, and save.

    Parameters
    ----------
    model_name:
        Hugging Face model id or local path (e.g. Qwen/Qwen3-4B-Instruct-2507).
    output_path:
        Where to write the compressed state_dict (torch.save file).
    sparsity:
        Fraction of last-dimension columns to discard.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    state_dict = model.state_dict()

    compressed: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        compressed[name] = _chop_last_dim(tensor, sparsity)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "sparsity": sparsity,
            "state_dict": compressed,
        },
        out,
    )

    del model, state_dict, compressed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def convert_to_hf_checkpoint(
    model_name: str,
    checkpoint_path: str,
    output_path: str,
) -> None:
    """Restore a compressed checkpoint into a full HF model directory.

    Loads model_name for the original parameter shapes, reads the compressed
    checkpoint produced by convert_from_hf_checkpoint, pads the missing
    portion of each last dimension with zeros, and writes a standard HF
    checkpoint under output_path via save_pretrained.
    """
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        compressed = payload["state_dict"]
    else:
        compressed = payload

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    original = model.state_dict()

    restored: dict[str, torch.Tensor] = {}
    for name, orig_tensor in original.items():
        if name not in compressed:
            restored[name] = torch.zeros_like(orig_tensor)
            continue
        chunk = compressed[name].to(dtype=orig_tensor.dtype)
        if chunk.shape == orig_tensor.shape:
            restored[name] = chunk
        else:
            if chunk.shape[:-1] != orig_tensor.shape[:-1]:
                raise ValueError(
                    f"Shape mismatch for '{name}': compressed {tuple(chunk.shape)} "
                    f"vs original {tuple(orig_tensor.shape)}"
                )
            restored[name] = _pad_last_dim(chunk, orig_tensor.shape[-1])

    model.load_state_dict(restored, strict=True)

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)

    del model, original, restored, compressed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Quick manual run: produces compressed.pt in the current directory.
    convert_from_hf_checkpoint(
        "Qwen/Qwen3-4B-Instruct-2507", "compressed.pt", sparsity=SPARSITY
    )
    print("Wrote compressed.pt")
