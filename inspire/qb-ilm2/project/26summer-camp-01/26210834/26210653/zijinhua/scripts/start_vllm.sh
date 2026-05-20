#!/usr/bin/env bash
# 本地 vLLM 部署 Qwen3.5-9B（OpenAI 兼容 API，供 zijinhua Agent 使用）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${ROOT}/ckpt/Qwen3.5-9B/qwen/Qwen3.5-9B}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8001}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
SERVED_NAME="${SERVED_MODEL_NAME:-qwen-3.5}"
# 多模态需加载视觉编码器，默认降低 max_len 以适配单卡显存
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
# 并发评测时 vLLM 同时处理的请求数（与 eval --workers 匹配）
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
# 设为 1 可退回纯文本模式（跳过视觉编码器）
VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-0}"
LOG="${ROOT}/logs/vllm_serve.log"
PID_FILE="${ROOT}/logs/vllm.pid"
VLLM_BIN="${VLLM_BIN:-/opt/conda/envs/or_llm/bin/vllm}"
CACHE_ROOT="${VLLM_CACHE_ROOT:-${ROOT}/.vllm_cache}"

mkdir -p "${ROOT}/logs" "${CACHE_ROOT}"

# 根分区 / 已满，缓存必须放到 /data
export VLLM_CACHE_ROOT="${CACHE_ROOT}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torch_compile"
export TMPDIR="${TMPDIR:-${ROOT}/tmp}"
mkdir -p "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TMPDIR}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "模型目录不存在: ${MODEL_PATH}"
  echo "请先运行: python scripts/download_qwen35_9b.py"
  exit 1
fi

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "vLLM 已在运行 (PID=$(cat "${PID_FILE}"))，端口 ${PORT}"
  exit 0
fi

echo "启动 vLLM: ${MODEL_PATH}"
echo "  GPU=${GPU}  port=${PORT}  served-model-name=${SERVED_NAME}"
echo "  日志: ${LOG}"

export CUDA_VISIBLE_DEVICES="${GPU}"

VLLM_ARGS=(
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_NAME}"
  --trust-remote-code
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_UTIL}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --enforce-eager
  --limit-mm-per-prompt '{"image": 1}'
)

if [[ "${VLLM_LANGUAGE_MODEL_ONLY}" == "1" ]]; then
  echo "  模式: 纯文本 (--language-model-only，无视觉)"
  VLLM_ARGS+=(--language-model-only)
else
  echo "  模式: 多模态（支持 image_url / base64 输入）"
fi

nohup "${VLLM_BIN}" serve "${MODEL_PATH}" \
  "${VLLM_ARGS[@]}" \
  >> "${LOG}" 2>&1 &

echo $! > "${PID_FILE}"
echo "已后台启动 PID=$(cat "${PID_FILE}")"
echo "就绪后测试: curl http://127.0.0.1:${PORT}/v1/models"
echo "Agent: LLM_BACKEND=vllm  LLM_BASE_URL=http://127.0.0.1:${PORT}/v1  MODEL_NAME=${SERVED_NAME}"
