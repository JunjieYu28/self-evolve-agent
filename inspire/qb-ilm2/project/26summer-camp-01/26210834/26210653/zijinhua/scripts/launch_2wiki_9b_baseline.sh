#!/usr/bin/env bash
# 2Wiki 纯 ReAct Baseline（9B，无反思/记忆）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/opt/conda/envs/zijinhua/bin/python}"
RUNS_DIR="$ROOT/logs/eval/runs"
mkdir -p "$RUNS_DIR"
LOG="$RUNS_DIR/2wiki_9b_baseline.log"

JSONL="${JSONL:-data/2wiki.jsonl}"
RUN_NAME="${RUN_NAME:-2wiki_9b_baseline}"
TOOLS="${TOOLS:-search}"
MAX_STEPS="${MAX_STEPS:-6}"
WORKERS="${WORKERS:-1}"
AGENT_URL="${AGENT_URL:-http://127.0.0.1:8001/v1}"
AGENT_MODEL="${AGENT_MODEL:-qwen-3.5}"
LLM_EXTRACT="${EVAL_ANSWER_LLM_EXTRACT:-auto}"
LLM_THRESHOLD="${EVAL_ANSWER_LLM_THRESHOLD:-20}"

api="${AGENT_URL%/}"
[[ "$api" != */v1 ]] && api="${api}/v1"
if ! curl -sf "${api}/models" >/dev/null 2>&1; then
  echo "错误: ${api} 9B 未就绪"
  exit 1
fi

EXTRA=("$@")
START_TS="$(date -Iseconds)"
{
  echo "================================================================"
  echo "2Wiki Baseline — 9B ReAct（无反思 / 无记忆）"
  echo "启动时间: ${START_TS}"
  echo "----------------------------------------------------------------"
  echo "数据:              ${JSONL}"
  echo "样本数:            $(wc -l < "${JSONL}" | tr -d ' ')"
  echo "run_name:          ${RUN_NAME}"
  echo "mode:              baseline"
  echo "tools:             ${TOOLS}"
  echo "max_steps:         ${MAX_STEPS}"
  echo "workers:           ${WORKERS}"
  echo "Agent URL:         ${AGENT_URL}"
  echo "Agent model:       ${AGENT_MODEL}"
  echo "EVAL_ANSWER_LLM_EXTRACT:   ${LLM_EXTRACT}"
  echo "EVAL_ANSWER_LLM_THRESHOLD: ${LLM_THRESHOLD}"
  echo "说明: 每轮 tool 结果进入 messages 供下轮 9B 使用；无 memory-context"
  echo "日志目录:          logs/eval/${RUN_NAME}_baseline_${TOOLS}/"
  echo "================================================================"
} >"$LOG"

nohup "$PYTHON" scripts/run_2wiki_baseline.py \
  --jsonl "$JSONL" \
  --run-name "$RUN_NAME" \
  --tools "$TOOLS" \
  --max-steps "$MAX_STEPS" \
  --workers "$WORKERS" \
  "${EXTRA[@]}" \
  >>"$LOG" 2>&1 &

echo "已启动 PID=$!  日志: $LOG"
echo "tail -f $LOG"
