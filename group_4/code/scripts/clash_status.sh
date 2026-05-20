#!/usr/bin/env bash
# 检查 Clash 与 Serper 连通性
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/set_proxy.sh" 2>/dev/null || true

echo "=== Clash ==="
curl -sf http://127.0.0.1:9090/version && echo "" || echo "API 不可用"
curl -s http://127.0.0.1:9090/proxies/%E8%80%81%E7%8C%AB%E4%BA%91 | python3 -c "import sys,json; d=json.load(sys.stdin); print('老猫云:', d.get('now'))" 2>/dev/null || true

echo ""
echo "=== 代理探测 ==="
for u in https://www.baidu.com https://google.serper.dev; do
  code=$(curl -x "${http_proxy:-}" -s -o /dev/null -w "%{http_code}" --max-time 15 "$u" 2>/dev/null || echo err)
  echo "$u -> $code"
done

echo ""
echo "=== search-proxy ==="
curl -sf http://127.0.0.1:8090/health && echo "" || echo "8090 未启动"

if [[ -f "${ROOT}/.env" ]]; then
  # shellcheck disable=SC1091
  source <(grep -E '^SERPER_API_KEY=' "${ROOT}/.env" | sed 's/\r$//')
  if [[ -n "${SERPER_API_KEY:-}" ]]; then
    echo ""
    echo "=== Serper via proxy ==="
    curl -s --max-time 30 -X POST http://127.0.0.1:8090/search/text \
      -H 'Content-Type: application/json' \
      -d '{"query":"Python","top_k":1,"fetch":false}' | python3 -m json.tool 2>/dev/null | head -12
  fi
fi
