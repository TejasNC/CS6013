# Submission — Week <N>, Compression <target>%, Track <1/2>

Per the project source-of-truth doc: this README documents setup/usage/repro,
not the compression technique itself (that goes in the report).

## Track

<!-- The Autumn PDF's folder/HF naming convention doesn't encode Track in the
     path, so state it explicitly here for the grader. -->
Track: <1 or 2>

## Setup

```bash
pip install -e .
```

## Usage

```bash
python compress.py --model_name Qwen/Qwen3.5-4B --output_path compressed.pt
python decompress.py --model_name Qwen/Qwen3.5-4B --checkpoint_path compressed.pt --output_path restored_hf/
```

## dtype

`torch.bfloat16` → `<fill in your compressed dtype, e.g. int4>`
