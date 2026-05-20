#!/usr/bin/env bash
# 本机一键启动：search-proxy + browser-service + 提示 vLLM
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "=== 1/3 加载代理（访问 Google/Serper 需要）==="
if [[ -f "${ROOT}/scripts/set_proxy.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/set_proxy.sh"
  echo "  http_proxy=${http_proxy:-未设置}"
else
  echo "  未找到 scripts/set_proxy.sh，若搜索失败请先配置代理"
fi

echo ""
echo "=== 2/3 启动 search-proxy (8090) ==="
bash scripts/start_search_proxy.sh
sleep 2
curl -sf http://127.0.0.1:8090/health && echo "" || echo "  [warn] search-proxy 未就绪"

echo ""
echo "=== 3/3 启动 browser-service (8080) ==="
bash scripts/start_browser_service.sh || echo "  [warn] browser 启动失败，见 logs/browser_service.log"

echo ""
echo "=== vLLM（若未启动）==="
if curl -sf http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
  echo "  vLLM 已在 http://127.0.0.1:8001"
else
  echo "  请另开终端: CUDA_VISIBLE_DEVICES=0 bash scripts/start_vllm.sh"
fi

echo ""
echo "=== 测试 ==="
echo "  python main.py test-search"
echo "  python main.py test-llm"
echo "  python main.py agent"
