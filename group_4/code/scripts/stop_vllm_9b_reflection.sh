#!/usr/bin/env bash
# 停止 scripts/start_vllm_9b_reflection.sh 启动的反思 vLLM
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${ROOT}/logs/vllm_9b_reflection.pid"
if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 3
    kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}" 2>/dev/null || true
    echo "已停止 9B 反思 vLLM PID=${pid}"
  fi
  rm -f "${PID_FILE}"
else
  echo "未找到 PID 文件，尝试按端口 8004 查找..."
  pkill -f "vllm serve.*--port 8004" 2>/dev/null && echo "已发送停止信号" || echo "无匹配进程"
fi
