# CS6013 23B0932 Compression_20 Submission01

Base model: Qwen/Qwen3.5-4B. The compressed checkpoint is hosted at
https://huggingface.co/tejaszzz/23B0932-Final-Compression20-Submission01

## Setup

Python >= 3.12, CUDA 12.6 compatible (torch 2.12.1 from the cu126 index; see pyproject.toml).

    uv sync

## Decompress

    python decompress.py --model_name Qwen/Qwen3.5-4B --checkpoint_path <compressed_dir> --output_path <out_dir>

Reads only the compressed directory (no network, `--model_name` is an unused placeholder). Runs on GPU if one with
more than 6 GB is free, otherwise on CPU. Writes a complete checkpoint (model.safetensors plus config and tokenizer files).

## Compress (optional, reproduces the compressed directory)

    python compress.py --model_name Qwen/Qwen3.5-4B --checkpoint_path <original_hf_dir> --output_path <compressed_dir> [--hessians H.pt]

## Dtypes

- Original checkpoint: torch.bfloat16 (text tower; vision tower is not counted for size).
- Compressed checkpoint: uint8 entropy-coded weight streams, int32 symbol counts, float16 scales, int16/uint8 index tables, bfloat16 for the remaining tensors.
- Decompressed checkpoint: torch.bfloat16 (all tensors keep the original names and shapes).
