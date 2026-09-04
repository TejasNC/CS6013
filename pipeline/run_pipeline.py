#!/usr/bin/env python3
"""Run a submission's full local pipeline: compress -> size check -> decompress
-> serve via vLLM -> evaluate (pipeline/run_eval.py, mirroring the TA eval
repo's actual grading path).

This shells out to each submission's compress.py/decompress.py as
subprocesses (rather than importing their functions), so it works for any
submission folder that follows the required CLI contract
(--model_name / --checkpoint_path / --output_path) without needing to know
anything about that submission's internals.

Usage
-----
    python pipeline/run_pipeline.py \\
        --submission 23B0932/Week01/Compression_10/Submission01 \\
        --model-name Qwen/Qwen3.5-4B \\
        --limit 20

    # Skip the eval step if you only want the compressed size:
    python pipeline/run_pipeline.py --submission ... --skip-eval
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], **kwargs) -> None:
    print(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def compress(submission_dir: Path, model_name: str, work_dir: Path) -> Path:
    compressed_path = work_dir / "compressed.pt"
    run([
        sys.executable, str(submission_dir / "compress.py"),
        "--model_name", model_name,
        "--output_path", str(compressed_path),
    ])
    return compressed_path


def decompress(submission_dir: Path, model_name: str, compressed_path: Path, work_dir: Path) -> Path:
    restored_dir = work_dir / "restored_hf"
    run([
        sys.executable, str(submission_dir / "decompress.py"),
        "--model_name", model_name,
        "--checkpoint_path", str(compressed_path),
        "--output_path", str(restored_dir),
    ])
    return restored_dir


def check_size(compressed_path: Path, model_name: str, baseline_cache: Path) -> None:
    run([
        sys.executable, str(REPO_ROOT / "pipeline" / "size_check.py"),
        "--compressed", str(compressed_path),
        "--baseline-model", model_name,
        "--baseline-cache", str(baseline_cache),
    ])


def start_vllm_server(model_dir: Path, port: int) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_dir),
        "--served-model-name", str(model_dir),
        "--tensor-parallel-size", "1",
        "--dtype", "auto",
        "--max-model-len", "33000",
        "--port", str(port),
    ]
    print(f"$ {' '.join(cmd)}   (background)")
    return subprocess.Popen(cmd)


def wait_for_server(port: int, timeout_s: int = 1800) -> None:
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            print("[vllm] server is ready")
            return
        except Exception:
            time.sleep(10)
    raise TimeoutError(f"vLLM server did not become ready within {timeout_s}s")


def run_eval(base_config: Path, model_dir: Path, limit: int | None, output_path: Path, port: int) -> None:
    cfg = yaml.safe_load(base_config.read_text())
    cfg["model_name"] = str(model_dir)
    cfg["vllm_base_url"] = f"http://localhost:{port}/v1"
    cfg["output"] = str(output_path)
    if limit is not None:
        cfg["limit"] = limit

    tmp_config = Path(tempfile.mkstemp(suffix=".yaml")[1])
    tmp_config.write_text(yaml.safe_dump(cfg, sort_keys=False))

    run([sys.executable, str(REPO_ROOT / "pipeline" / "run_eval.py"), "--config", str(tmp_config)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True, help="Submission folder (has compress.py/decompress.py)")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "eval_config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Override the config's example limit")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--skip-eval", action="store_true", help="Stop after compress/decompress/size-check")
    parser.add_argument(
        "--baseline-cache", type=Path, default=REPO_ROOT / ".cache" / "baseline_bytes.txt",
        help="Cache file for the original model's byte count, to skip re-downloading it every run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="pipeline_", dir="/tmp"))
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"[work_dir] {work_dir}")

    print("\n[1/4] compress")
    compressed_path = compress(args.submission, args.model_name, work_dir)

    print("\n[2/4] size check")
    check_size(compressed_path, args.model_name, args.baseline_cache)

    print("\n[3/4] decompress")
    restored_dir = decompress(args.submission, args.model_name, compressed_path, work_dir)

    if args.skip_eval:
        print(f"\n[done] --skip-eval set; restored checkpoint at {restored_dir}")
        return

    print("\n[4/4] serve + evaluate")
    server = start_vllm_server(restored_dir, args.port)
    try:
        wait_for_server(args.port)
        output_path = work_dir / "results.json"
        run_eval(args.config, restored_dir, args.limit, output_path, args.port)
        print(f"\n[done] results at {output_path}")
    finally:
        print("[vllm] shutting down server")
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
