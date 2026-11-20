#!/usr/bin/env bash
# Serve a model dir with vLLM exactly like the TA phase2_eval.sh (language-model-only, 33000 ctx, bf16).
# usage: serve_vllm.sh MODEL_DIR GPU PORT [LOGFILE]
set -euo pipefail
source /mnt/home/tejaschaudhari/emir/env.sh
M=$1; GPU=$2; PORT=$3; LOG=${4:-/mnt/nas/tejaschaudhari/emir_data/runs/vllm_$PORT.log}
export VLLM_USE_FLASHINFER_SAMPLER=0; export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=$GPU setsid nohup /mnt/home/tejaschaudhari/emir/envs_vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$M" --served-model-name Qwen/Qwen3.5-4B --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 33000 --language-model-only --port "$PORT" \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" --max-num-seqs "${MAX_SEQS:-128}" > "$LOG" 2>&1 &
echo $! > "${LOG%.log}.pid"
timeout 900 bash -c "until curl -sf localhost:$PORT/v1/models >/dev/null; do sleep 5; done" && echo READY || { tail -30 "$LOG"; exit 1; }
