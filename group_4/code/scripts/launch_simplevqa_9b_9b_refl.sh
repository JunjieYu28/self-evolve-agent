#!/usr/bin/env bash
# SimpleVQA：9B Agent (:8003 GPU3) + 9B 反思 (:8004)，不占用 2Wiki :8001
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/opt/conda/envs/zijinhua/bin/python}"
RUNS_DIR="$ROOT/logs/eval/runs"
mkdir -p "$RUNS_DIR"
LOG="${LOG:-$RUNS_DIR/simplevqa_9b_9b_refl.log}"

JSONL="${JSONL:-simpleVQA/SimpleVQA.jsonl}"
RUN_NAME="${RUN_NAME:-simplevqa_9b_9b_refl}"
MODE="${MODE:-full}"
TOOLS="${TOOLS:-search}"
MAX_STEPS="${MAX_STEPS:-6}"
WORKERS="${WORKERS:-1}"
AGENT_URL="${AGENT_URL:-http://127.0.0.1:8003/v1}"
AGENT_MODEL="${AGENT_MODEL:-qwen-3.5}"
REFL_URL="${REFL_URL:-http://127.0.0.1:8004/v1}"
REFL_MODEL="${REFL_MODEL:-qwen-3.5}"

api="${AGENT_URL%/}"
[[ "$api" != */v1 ]] && api="${api}/v1"
if ! curl -sf "${api}/models" >/dev/null 2>&1; then
  echo "Agent :8003 未就绪，启动 GPU3 vLLM ..."
  bash "$ROOT/scripts/start_vllm_9b_simplevqa_agent.sh"
  for _ in $(seq 1 90); do
    curl -sf "${api}/models" >/dev/null 2>&1 && break
    sleep 5
  done
  if ! curl -sf "${api}/models" >/dev/null 2>&1; then
    echo "错误: Agent API ${api} 未就绪，见 logs/vllm_simplevqa_agent.log"
    exit 1
  fi
fi

refl_api="${REFL_URL%/}"
[[ "$refl_api" != */v1 ]] && refl_api="${refl_api}/v1"
if [[ "$MODE" == "full" ]]; then
  if ! curl -sf "${refl_api}/models" >/dev/null 2>&1; then
    echo "反思 API 未就绪，尝试 start_vllm_9b_reflection.sh ..."
    bash "$ROOT/scripts/start_vllm_9b_reflection.sh" || true
    for _ in $(seq 1 60); do
      curl -sf "${refl_api}/models" >/dev/null 2>&1 && break
      sleep 5
    done
    if ! curl -sf "${refl_api}/models" >/dev/null 2>&1; then
      echo "错误: 反思 API ${refl_api} 未就绪"
      exit 1
    fi
  fi
fi

EXTRA=("$@")
RESUME_FLAG=""
if [[ " ${EXTRA[*]:-} " == *" --resume "* ]]; then
  RESUME_FLAG="(resume)"
else
  echo "将清空 logs/eval/${RUN_NAME}_${MODE}_${TOOLS}/"
fi

START_TS="$(date -Iseconds)"
{
  echo "================================================================"
  echo "SimpleVQA 评测 — 9B Agent (GPU3 :8003) + 9B 反思 (:8004)"
  echo "启动时间: ${START_TS}"
  echo "----------------------------------------------------------------"
  echo "数据:        ${JSONL}"
  echo "样本数:      $(wc -l < "${JSONL}" | tr -d ' ') 行"
  echo "run_name:    ${RUN_NAME}"
  echo "mode:        ${MODE}"
  echo "tools:       ${TOOLS}"
  echo "max_steps:   ${MAX_STEPS}"
  echo "workers:     ${WORKERS}"
  echo "resume:      ${RESUME_FLAG:-false}"
  echo "Agent URL:   ${AGENT_URL}"
  echo "反思 URL:    ${REFL_URL}"
  echo "闭卷:        eval_benchmark 不向 Agent/反思传入 gold（标答仅事后打分）"
  echo "日志目录:    logs/eval/${RUN_NAME}_${MODE}_${TOOLS}/"
  echo "控制台日志:  ${LOG}"
  echo "注意: 2Wiki 仍在 :8001 运行，本任务独立占用 GPU3"
  echo "================================================================"
} >"$LOG"

nohup "$PYTHON" scripts/run_simplevqa_9b_9b_refl.py \
  --jsonl "$JSONL" \
  --run-name "$RUN_NAME" \
  --mode "$MODE" \
  --tools "$TOOLS" \
  --max-steps "$MAX_STEPS" \
  --workers "$WORKERS" \
  "${EXTRA[@]}" \
  >>"$LOG" 2>&1 &

echo "已启动 SimpleVQA 评测 PID=$!  ${RESUME_FLAG}"
echo "tail -f $LOG"
