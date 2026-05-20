#!/usr/bin/env bash
# 重启 Qwen3.5-9B vLLM（多模态模式）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${ROOT}/logs/vllm.pid"

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  if kill -0 "${PID}" 2>/dev/null; then
    echo "停止旧 vLLM PID=${PID} ..."
    kill "${PID}" || true
    sleep 3
    kill -9 "${PID}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
fi

export VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-0}"
exec bash "${ROOT}/scripts/start_vllm.sh"
