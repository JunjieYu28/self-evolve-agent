#!/usr/bin/env bash
# SimpleVQA：9B baseline（无反思/记忆），默认 :8002 与 full(:8003) 分流
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/opt/conda/envs/zijinhua/bin/python}"
RUNS_DIR="$ROOT/logs/eval/runs"
mkdir -p "$RUNS_DIR"
LOG="${LOG:-$RUNS_DIR/simplevqa_9b_baseline.log}"

JSONL="${JSONL:-simpleVQA/SimpleVQA.jsonl}"
RUN_NAME="${RUN_NAME:-simplevqa_9b_baseline}"
TOOLS="${TOOLS:-search}"
MAX_STEPS="${MAX_STEPS:-6}"
WORKERS="${WORKERS:-1}"
AGENT_URL="${AGENT_URL:-http://127.0.0.1:8002/v1}"
AGENT_MODEL="${AGENT_MODEL:-qwen-3.5}"

api="${AGENT_URL%/}"
[[ "$api" != */v1 ]] && api="${api}/v1"
if ! curl -sf "${api}/models" >/dev/null 2>&1; then
  echo "Agent ${api} 未就绪，尝试启动 GPU1 :8002 ..."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" VLLM_PORT=8002 bash "$ROOT/scripts/start_vllm.sh" || true
  for _ in $(seq 1 60); do
    curl -sf "${api}/models" >/dev/null 2>&1 && break
    sleep 5
  done
  if ! curl -sf "${api}/models" >/dev/null 2>&1; then
    echo "错误: Agent API ${api} 未就绪"
    exit 1
  fi
fi

EXTRA=("$@")
START_TS="$(date -Iseconds)"
{
  echo "================================================================"
  echo "SimpleVQA Baseline — 9B（无反思 / 无记忆）"
  echo "启动时间: ${START_TS}"
  echo "----------------------------------------------------------------"
  echo "数据:        ${JSONL}"
  echo "样本数:      $(wc -l < "${JSONL}" | tr -d ' ')"
  echo "run_name:    ${RUN_NAME}"
  echo "mode:        baseline"
  echo "tools:       ${TOOLS}"
  echo "Agent URL:   ${AGENT_URL}  (full 评测常用 :8003，互不杀进程)"
  echo "日志目录:    logs/eval/${RUN_NAME}_baseline_${TOOLS}/"
  echo "控制台:      ${LOG}"
  echo "================================================================"
} >"$LOG"

nohup env MEMORY_PRELOAD_PATH= "$PYTHON" scripts/run_simplevqa_9b_baseline.py \
  --jsonl "$JSONL" \
  --run-name "$RUN_NAME" \
  --tools "$TOOLS" \
  --max-steps "$MAX_STEPS" \
  --workers "$WORKERS" \
  "${EXTRA[@]}" \
  >>"$LOG" 2>&1 &

echo "已启动 PID=$!"
echo "tail -f $LOG"
