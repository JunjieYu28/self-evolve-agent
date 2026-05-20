#!/usr/bin/env bash
# SimpleVQA 专用 9B 多模态 Agent（GPU3 :8003，与 2Wiki :8001 分离）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export VLLM_PORT="${VLLM_PORT:-8003}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
export VLLM_LANGUAGE_MODEL_ONLY=0

LOG="${ROOT}/logs/vllm_simplevqa_agent.log"
PID_FILE="${ROOT}/logs/vllm_simplevqa_agent.pid"
mkdir -p "${ROOT}/logs"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "SimpleVQA Agent vLLM 已在运行 (PID=$(cat "${PID_FILE}"))，端口 ${VLLM_PORT}"
    exit 0
  fi
fi

MODEL_PATH="${MODEL_PATH:-${ROOT}/ckpt/Qwen3.5-9B/qwen/Qwen3.5-9B}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT}"
GPU="${CUDA_VISIBLE_DEVICES}"
SERVED_NAME="${SERVED_MODEL_NAME:-qwen-3.5}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
VLLM_BIN="${VLLM_BIN:-/opt/conda/envs/or_llm/bin/vllm}"
CACHE_ROOT="${VLLM_CACHE_ROOT:-${ROOT}/.vllm_cache_simplevqa}"

mkdir -p "${CACHE_ROOT}"
export VLLM_CACHE_ROOT="${CACHE_ROOT}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torch_compile"
export TMPDIR="${TMPDIR:-${ROOT}/tmp}"
mkdir -p "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TMPDIR}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "模型目录不存在: ${MODEL_PATH}"
  exit 1
fi

echo "启动 SimpleVQA Agent vLLM: GPU=${GPU} port=${PORT}"
export CUDA_VISIBLE_DEVICES="${GPU}"

nohup "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_NAME}" \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enforce-eager \
  --limit-mm-per-prompt '{"image": 1}' \
  >> "${LOG}" 2>&1 &

echo $! > "${PID_FILE}"
echo "已后台启动 PID=$(cat "${PID_FILE}")，日志 ${LOG}"
