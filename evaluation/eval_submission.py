#!/usr/bin/env python3
"""Evaluate a compression submission end-to-end on GPQA + MMLU-Pro (10 samples)."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Mapping

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_MODEL, configure_logging


def count_total_size(state_dict: Mapping[str, torch.Tensor]) -> dict[str, float | int]:
    """Aggregate parameter count and memory size for a checkpoint ``state_dict``.

    Size is derived from each tensor's number of elements and ``dtype`` item size
    (not from on-disk serialization overhead).

    Returns
    -------
    dict with:
      - ``num_tensors``: number of parameter tensors
      - ``num_parameters``: total scalar elements
      - ``total_bytes``: total size in bytes
      - ``total_mb`` / ``total_gb``: same size in MiB / GiB
    """
    num_tensors = 0
    num_parameters = 0
    total_bytes = 0
    for tensor in state_dict.values():
        if not isinstance(tensor, torch.Tensor):
            continue
        num_tensors += 1
        num_parameters += int(tensor.numel())
        total_bytes += int(tensor.numel()) * int(tensor.element_size())

    return {
        "num_tensors": num_tensors,
        "num_parameters": num_parameters,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024**2),
        "total_gb": total_bytes / (1024**3),
    }


def _format_size_report(label: str, stats: Mapping[str, float | int]) -> str:
    return (
        f"{label}: params={stats['num_parameters']:,} "
        f"({stats['num_tensors']} tensors), "
        f"size={stats['total_bytes']:,} bytes "
        f"({stats['total_mb']:.2f} MiB / {stats['total_gb']:.4f} GiB)"
    )


def load_submission(submission: str | Path) -> ModuleType:
    """Load a submission module exposing convert_from/to_hf_checkpoint."""
    path = Path(submission).resolve()
    if path.is_dir():
        candidates = [
            path / "code.py",
        ]
        for candidate in candidates:
            if candidate.is_file():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                f"No code.py found under {submission}"
            )
    if not path.is_file():
        raise FileNotFoundError(f"Submission file not found: {path}")

    spec = importlib.util.spec_from_file_location("submission_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load submission from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_module"] = module
    spec.loader.exec_module(module)

    for fn in ("convert_from_hf_checkpoint", "convert_to_hf_checkpoint"):
        if not hasattr(module, fn):
            raise AttributeError(f"Submission {path} is missing required function: {fn}")
    return module


def print_checkpoint_size(checkpoint_path: Path) -> None:
    """Print on-disk size and parameter/dtype aggregate size when possible."""
    if checkpoint_path.is_file():
        disk_bytes = checkpoint_path.stat().st_size
    elif checkpoint_path.is_dir():
        disk_bytes = sum(p.stat().st_size for p in checkpoint_path.rglob("*") if p.is_file())
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(
        f"[size] on-disk {checkpoint_path}: "
        f"{disk_bytes:,} bytes ({disk_bytes / (1024**2):.2f} MiB)"
    )

    if not checkpoint_path.is_file():
        return

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    if not isinstance(state_dict, dict):
        return
    stats = count_total_size(state_dict)
    print(_format_size_report("[size] compressed state_dict", stats))


def ensure_tokenizer(model_name: str, restored_dir: Path) -> None:
    """Copy tokenizer files into the restored HF directory for eval loading."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(restored_dir)


def _env_for_eval_subprocess() -> dict[str, str]:
    """Build env for ``run_eval.py`` without GPTQ/tokenicer cache redirects.

    Importing ``gptqmodel`` (via a GPTQ submission) rewrites ``HF_HOME`` /
    ``HF_HUB_CACHE`` / etc. to ``/tmp/tokenicer_hf_home``. A naive
    ``subprocess.run`` inherits that, so the child can't see the normal
    Hugging Face dataset cache and gated loads (GPQA) fail even when a
    fresh shell can load the same checkpoint fine.
    """
    env = os.environ.copy()
    for key in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_MODULES_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
    ):
        value = env.get(key, "")
        if "tokenicer_hf_home" in value:
            env.pop(key, None)
    return env


def run_eval_on_checkpoint(
    restored_dir: Path,
    *,
    limit: int,
    output_dir: Path,
    batch_size: int,
    max_new_tokens: int,
    dtype: str,
    device_map: str,
    verbose: bool,
) -> None:
    run_eval_script = Path(__file__).resolve().parent / "run_eval.py"
    cmd = [
        sys.executable,
        str(run_eval_script),
        "--model",
        str(restored_dir),
        "--benchmarks",
        "gpqa",
        "mmlu_pro",
        "--limit",
        str(limit),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(batch_size),
        "--max-new-tokens",
        str(max_new_tokens),
        "--dtype",
        dtype,
        "--device-map",
        device_map,
    ]
    if verbose:
        cmd.append("--verbose")

    print(f"[eval] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=_env_for_eval_subprocess())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        type=Path,
        required=True,
        help="Path to submission file (code.py) or directory containing it",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HF model id / path used as the source checkpoint",
    )
    parser.add_argument("--limit", type=int, default=10, help="Examples per benchmark")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results/submission"),
        help="Where run_eval.py writes JSON results",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory under /tmp for intermediate checkpoints (default: auto)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help=(
            "Fraction of last-dimension columns to drop; passed through to "
            "convert_from_hf_checkpoint (e.g. 0.5 keeps the first 50%%)"
        ),
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Do not delete /tmp checkpoints after evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    submission = load_submission(args.submission)
    print(f"[submission] Loaded from {Path(args.submission).resolve()}")

    work_dir = args.work_dir
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="eval_submission_", dir="/tmp"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    compressed_path = work_dir / "compressed.pt"
    restored_dir = work_dir / "restored_hf"

    if not 0.0 <= args.sparsity <= 1.0:
        raise SystemExit(f"--sparsity must be in [0, 1], got {args.sparsity}")

    print(
        f"[1/3] convert_from_hf_checkpoint -> {compressed_path} "
        f"(sparsity={args.sparsity})"
    )
    submission.convert_from_hf_checkpoint(
        args.model, str(compressed_path), sparsity=args.sparsity
    )
    print_checkpoint_size(compressed_path)

    print(f"[2/3] convert_to_hf_checkpoint -> {restored_dir}")
    submission.convert_to_hf_checkpoint(args.model, str(compressed_path), str(restored_dir))
    ensure_tokenizer(args.model, restored_dir)
    print_checkpoint_size(restored_dir)

    print(f"[3/3] run_eval.py on {args.limit} samples (gpqa + mmlu_pro)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_eval_on_checkpoint(
        restored_dir,
        limit=args.limit,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        device_map=args.device_map,
        verbose=args.verbose,
    )

    print(f"[done] Results in {args.output_dir.resolve()}")
    print(f"[done] Intermediate checkpoints in {work_dir}")
    if not args.keep_tmp:
        # Keep artifacts by default in /tmp; user can delete. Only note path.
        pass


if __name__ == "__main__":
    main()
