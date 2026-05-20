#!/usr/bin/env bash
# 本机启动 browser-service 沙盒（默认 0.0.0.0:8080）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${ROOT}/scripts/set_proxy.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/set_proxy.sh"
fi
# shellcheck disable=SC1091
source "${ROOT}/scripts/playwright_env.sh"
BS_DIR="${ROOT}/external/browser-service"
PY="${PLAYWRIGHT_PYTHON}"
PORT="${SANDBOX_PORT:-8080}"
HOST="${SANDBOX_HOST:-0.0.0.0}"
LOG="${ROOT}/logs/browser_service.log"
PID_FILE="${ROOT}/logs/browser_service.pid"

mkdir -p "${ROOT}/logs"
cd "${BS_DIR}"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "browser-service 已在运行 PID=$(cat "${PID_FILE}")"
  exit 0
fi

export HOST PORT

nohup env HOST="$HOST" PORT="$PORT" \
  PLAYWRIGHT_BROWSERS_PATH="$PLAYWRIGHT_BROWSERS_PATH" \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  TMPDIR="$TMPDIR" \
  http_proxy="${http_proxy:-}" https_proxy="${https_proxy:-}" all_proxy="${all_proxy:-}" \
  "$PY" -m app.main >> "${LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "已启动 browser-service → http://${HOST}:${PORT}"
echo "健康检查: curl http://127.0.0.1:${PORT}/health"
echo "API 文档: http://127.0.0.1:${PORT}/docs"
echo "日志: ${LOG}"
echo ""
echo "GPU 侧 .env 填写（经端口转发后）:"
echo "  SANDBOX_BASE_URL=http://127.0.0.1:${PORT}"
