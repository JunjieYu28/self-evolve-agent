#!/usr/bin/env bash
# 本机启动 search-proxy（默认 127.0.0.1:8090）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 自动加载代理
if [[ -f "${ROOT}/scripts/set_proxy.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/set_proxy.sh"
fi
# 从 .env 读取 SERPER_API_KEY（若未 export）
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source <(grep -E '^(SERPER_API_KEY|JINA_API_KEY)=' "${ROOT}/.env" | sed 's/\r$//')
  set +a
fi
PROXY_DIR="${ROOT}/external/harness-sii/search-proxy"
PORT="${SEARCH_PROXY_PORT:-8090}"
HOST="${SEARCH_PROXY_HOST:-127.0.0.1}"
LOG="${ROOT}/logs/search_proxy.log"
PID_FILE="${ROOT}/logs/search_proxy.pid"

if [[ -z "${SERPER_API_KEY:-}" ]]; then
  echo "请先设置: export SERPER_API_KEY=你的密钥"
  exit 1
fi

mkdir -p "${ROOT}/logs"
cd "${PROXY_DIR}"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "search-proxy 已在运行 PID=$(cat "${PID_FILE}")"
  exit 0
fi

# 清理占用端口的旧进程（无 pid 文件时 start 脚本无法感知）
if command -v ss >/dev/null 2>&1; then
  old_pid="$(ss -tlnp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {print}' | grep -oP 'pid=\K[0-9]+' | head -1 || true)"
elif command -v lsof >/dev/null 2>&1; then
  old_pid="$(lsof -ti "tcp:${PORT}" -sTCP:LISTEN 2>/dev/null | head -1 || true)"
else
  old_pid=""
fi
if [[ -n "${old_pid:-}" ]]; then
  echo "停止占用 ${PORT} 的旧进程 PID=${old_pid}"
  kill "${old_pid}" 2>/dev/null || true
  sleep 1
fi

export HOST PORT
nohup env \
  http_proxy="${http_proxy:-}" \
  https_proxy="${https_proxy:-}" \
  all_proxy="${all_proxy:-}" \
  SERPER_API_KEY="${SERPER_API_KEY}" \
  JINA_API_KEY="${JINA_API_KEY:-}" \
  IMAGE_UPLOADER="${IMAGE_UPLOADER:-tmpfiles}" \
  bash ./run.sh >> "${LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "已启动 search-proxy → http://${HOST}:${PORT}"
echo "健康检查: curl http://${HOST}:${PORT}/health"
echo "日志: ${LOG}"
echo ""
echo "GPU 侧 .env 填写（经 VS Code/SSH 端口转发后）:"
echo "  SEARCH_PROXY_URL=http://127.0.0.1:${PORT}"
