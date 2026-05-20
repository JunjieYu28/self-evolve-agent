#!/usr/bin/env bash
# 32B Agent 专用 vLLM（带 tool calling），与 :8000 反思服务、:8001 9B 基座互不干扰。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${ROOT}/ckpt/Qwen3-32B}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_32B_AGENT_PORT:-8003}"
GPU="${CUDA_VISIBLE_DEVICES:-3}"
SERVED_NAME="${SERVED_MODEL_NAME:-Qwen3-32B}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"
VLLM_BIN="${VLLM_BIN:-/opt/conda/envs/or_llm/bin/vllm}"
CACHE_ROOT="${VLLM_CACHE_ROOT:-${ROOT}/.vllm_cache_32b}"
LOG="${ROOT}/logs/vllm_32b_agent.log"
PID_FILE="${ROOT}/logs/vllm_32b_agent.pid"

mkdir -p "${ROOT}/logs" "${CACHE_ROOT}"
export VLLM_CACHE_ROOT="${CACHE_ROOT}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torch_compile"
export TMPDIR="${TMPDIR:-${ROOT}/tmp}"
mkdir -p "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TMPDIR}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "模型目录不存在: ${MODEL_PATH}"
  exit 1
fi

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "32B Agent vLLM 已在运行 (PID=$(cat "${PID_FILE}"))，端口 ${PORT}"
  exit 0
fi

if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "端口 ${PORT} 已有 vLLM 服务，跳过启动"
  exit 0
fi

echo "启动 32B Agent vLLM: ${MODEL_PATH}"
echo "  GPU=${GPU}  port=${PORT}  served-model-name=${SERVED_NAME}"
echo "  日志: ${LOG}"

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
  >> "${LOG}" 2>&1 &

echo $! > "${PID_FILE}"
echo "已后台启动 PID=$(cat "${PID_FILE}")"
echo "就绪后: curl http://127.0.0.1:${PORT}/v1/models"
