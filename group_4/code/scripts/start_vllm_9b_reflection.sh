#!/usr/bin/env bash
# 专用 9B 反思 vLLM（与 Agent :8001/:8002 分离，默认 GPU2 :8004）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${ROOT}/ckpt/Qwen3.5-9B/qwen/Qwen3.5-9B}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_REFLECTION_PORT:-8004}"
GPU="${CUDA_VISIBLE_DEVICES:-2}"
SERVED_NAME="${SERVED_MODEL_NAME:-qwen-3.5}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
# 反思多为串行诊断，并发需求低于 Agent
MAX_NUM_SEQS="${VLLM_REFLECTION_MAX_NUM_SEQS:-4}"
VLLM_BIN="${VLLM_BIN:-/opt/conda/envs/or_llm/bin/vllm}"
LOG="${ROOT}/logs/vllm_9b_reflection.log"
PID_FILE="${ROOT}/logs/vllm_9b_reflection.pid"
CACHE_ROOT="${VLLM_CACHE_ROOT:-${ROOT}/.vllm_cache_reflection}"

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
  echo "9B 反思 vLLM 已在运行 (PID=$(cat "${PID_FILE}"))，端口 ${PORT}"
  exit 0
fi

echo "启动 9B 反思 vLLM: ${MODEL_PATH}"
echo "  GPU=${GPU}  port=${PORT}  served-model-name=${SERVED_NAME}"
echo "  模式: 纯文本（无 tool call，供 REFLECTION_LLM_BASE_URL 使用）"
echo "  日志: ${LOG}"

export CUDA_VISIBLE_DEVICES="${GPU}"

nohup "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_NAME}" \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enforce-eager \
  --language-model-only \
  >> "${LOG}" 2>&1 &

echo $! > "${PID_FILE}"
echo "已后台启动 PID=$(cat "${PID_FILE}")"
echo "测试: curl http://127.0.0.1:${PORT}/v1/models"
echo "配置: REFLECTION_LLM_BASE_URL=http://127.0.0.1:${PORT}/v1  REFLECTION_MODEL_NAME=${SERVED_NAME}"
