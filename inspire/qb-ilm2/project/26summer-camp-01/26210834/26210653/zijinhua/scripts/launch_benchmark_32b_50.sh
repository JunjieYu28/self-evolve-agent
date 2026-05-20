#!/usr/bin/env bash
# 一键启动 32B Agent vLLM（GPU3:8003）+ 前 50 题消融 benchmark（后台）
# 不会停止 group_004 / 9B :8001 上正在跑的任务
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/zijinhua/bin/python}"
RUN_LOG="${ROOT}/logs/benchmark/runs/group_032_32b.log"
VLLM_LOG="${ROOT}/logs/vllm_32b_agent.log"
PORT="${VLLM_32B_AGENT_PORT:-8003}"

mkdir -p "${ROOT}/logs/benchmark/runs"

echo "[1/3] 检查/启动 32B Agent vLLM (port ${PORT}, GPU 3) ..."
bash "${ROOT}/scripts/start_vllm_32b_agent.sh"

echo "[2/3] 等待 vLLM 就绪 ..."
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "  vLLM 已就绪 (${i}s)"
    break
  fi
  if [[ "${i}" -eq 120 ]]; then
    echo "  超时：请查看 ${VLLM_LOG}"
    exit 1
  fi
  sleep 5
done

# 验证 tool calling
if ! curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"Qwen3-32B\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"search_text\",\"description\":\"x\",\"parameters\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}},\"required\":[\"query\"]}}}],\"tool_choice\":\"auto\"}" \
  >/dev/null 2>&1; then
  echo "  警告: tool calling 探测失败，benchmark 可能无法调用 search_text"
  echo "  详见 ${VLLM_LOG}"
fi

echo "[3/3] 后台启动 32B 消融 benchmark（前 50 题，baseline+search）..."
: > "${RUN_LOG}"
nohup "${PYTHON}" "${ROOT}/scripts/run_benchmark_32b_ablation.py" \
  --group 032 \
  --limit 50 \
  --mode baseline \
  --tools search \
  --workers 1 \
  --max-steps 6 \
  >> "${RUN_LOG}" 2>&1 &

echo "PID=$!"
echo "进度: tail -f ${RUN_LOG}"
echo "提交物: ${ROOT}/group_032.json (跑完后)"
