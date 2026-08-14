"""GPTQ submission: int8 GPTQ compress / bf16 restore for HF checkpoints.

Follows the Transformers GPTQ flow:
https://huggingface.co/docs/transformers/en/quantization/gptq
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig


def convert_from_hf_checkpoint(
    model_name: str,
    output_path: str,
    sparsity: float = 0.5,
) -> None:
    """GPTQ-quantize ``model_name`` to int8 and write a compressed HF directory.

    Parameters
    ----------
    model_name:
        Hugging Face model id or local path.
    output_path:
        Destination directory for the GPTQ checkpoint (``save_pretrained``).
        Eval may pass a ``.pt`` path; this still writes a directory at that path.
    sparsity:
        Accepted for API compatibility with ``eval_submission.py``. Unused by
        GPTQ (column-sparsity is handled by the sample submission instead).
    """
    _ = sparsity  # API compatibility with eval_submission --sparsity

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    gptq_config = GPTQConfig(bits=8, dataset="c4", tokenizer=tokenizer)

    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16,
        quantization_config=gptq_config,
        trust_remote_code=True,
    )

    out = Path(output_path)
    if out.exists():
        if out.is_file():
            out.unlink()
        elif out.is_dir() and not any(out.iterdir()):
            pass
        elif out.is_dir():
            # Reuse empty-able path; save_pretrained overwrites files.
            pass
    out.mkdir(parents=True, exist_ok=True)

    # HF docs: move fully to one device before save when quantized with device_map.
    quantized_model.to("cpu")
    quantized_model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    del quantized_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _dequantize_gptq_to_bf16(model: torch.nn.Module) -> torch.nn.Module:
    """Replace GPTQ TorchLinear modules with bf16 ``nn.Linear`` weights."""
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    named = dict(model.named_modules())
    for name, module in list(model.named_modules()):
        if isinstance(module, BaseQuantLinear) and not isinstance(module, TorchLinear):
            raise ValueError(
                "Only models loaded with backend=BACKEND.GPTQ_TORCH are supported "
                "for dequantization."
            )
        if not isinstance(module, TorchLinear):
            continue

        weight = module.dequantize_weight().T.detach().to(device="cpu", dtype=torch.bfloat16)
        new_module = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
        new_module.weight = nn.Parameter(weight)
        if module.bias is not None:
            new_module.bias = nn.Parameter(module.bias.detach().to(device="cpu", dtype=torch.bfloat16))

        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            setattr(named[parent_name], child_name, new_module)
        else:
            setattr(model, name, new_module)
        named[name] = new_module

    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        delattr(model.config, "quantization_config")
    return model


def convert_to_hf_checkpoint(
    model_name: str,
    checkpoint_path: str,
    output_path: str,
) -> None:
    """Load a GPTQ int8 checkpoint, dequantize to bf16, and save an HF model.

    Parameters
    ----------
    model_name:
        Original HF model id (unused for shapes; kept for API compatibility).
    checkpoint_path:
        Directory written by :func:`convert_from_hf_checkpoint`.
    output_path:
        Directory for the restored bf16 HF checkpoint.
    """
    _ = model_name

    from gptqmodel import GPTQModel
    from gptqmodel.utils.backend import BACKEND

    gptq_model = GPTQModel.load(
        str(checkpoint_path),
        backend=BACKEND.GPTQ_TORCH,
        trust_remote_code=True,
    )
    restored = _dequantize_gptq_to_bf16(gptq_model.model)
    restored = restored.to(dtype=torch.bfloat16)

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    restored.save_pretrained(out)

    tokenizer = getattr(gptq_model, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(out)

    del gptq_model, restored
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
