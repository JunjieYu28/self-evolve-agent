#!/usr/bin/env bash
# 9B 推理 (:8001) + 9B 反思 (:8004 专用)，评测 data/2wiki.jsonl
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/opt/conda/envs/zijinhua/bin/python}"
RUNS_DIR="$ROOT/logs/eval/runs"
mkdir -p "$RUNS_DIR"
LOG="${LOG:-$RUNS_DIR/2wiki_9b_agent_32b_refl.log}"

JSONL="${JSONL:-data/2wiki.jsonl}"
RUN_NAME="${RUN_NAME:-2wiki_9b_agent_32b_refl}"
MODE="${MODE:-full}"
TOOLS="${TOOLS:-search}"
MAX_STEPS="${MAX_STEPS:-6}"
WORKERS="${WORKERS:-1}"
AGENT_URL="${AGENT_URL:-http://127.0.0.1:8001/v1}"
AGENT_MODEL="${AGENT_MODEL:-qwen-3.5}"
REFL_URL="${REFL_URL:-http://127.0.0.1:8004/v1}"
REFL_MODEL="${REFL_MODEL:-qwen-3.5}"
TEMP="${EVAL_TEMPERATURE:-0.3}"
MAX_TOKENS="${EVAL_MAX_TOKENS:-512}"
THINKING="${LLM_ENABLE_THINKING:-false}"
LLM_EXTRACT="${EVAL_ANSWER_LLM_EXTRACT:-auto}"
LLM_THRESHOLD="${EVAL_ANSWER_LLM_THRESHOLD:-20}"

api="${AGENT_URL%/}"
[[ "$api" != */v1 ]] && api="${api}/v1"
if ! curl -sf "${api}/models" >/dev/null 2>&1; then
  echo "错误: Agent API ${api} 未就绪"
  exit 1
fi
refl_api="${REFL_URL%/}"
[[ "$refl_api" != */v1 ]] && refl_api="${refl_api}/v1"
if [[ "$MODE" == "full" ]]; then
  if ! curl -sf "${refl_api}/models" >/dev/null 2>&1; then
    echo "反思 API ${refl_api} 未就绪，尝试 bash scripts/start_vllm_9b_reflection.sh ..."
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
  echo "2Wiki 评测 — 9B Agent + 9B 反思"
  echo "启动时间: ${START_TS}"
  echo "----------------------------------------------------------------"
  echo "数据:        ${JSONL}"
  echo "样本数:      $(wc -l < "${JSONL}" | tr -d ' ') 行"
  echo "run_name:    ${RUN_NAME}"
  echo "mode:        ${MODE}  (reflection/memory=$([ "${MODE}" = full ] && echo on || echo off))"
  echo "tools:       ${TOOLS}"
  echo "max_steps:   ${MAX_STEPS}"
  echo "workers:     ${WORKERS}"
  echo "resume:      ${RESUME_FLAG:-false}"
  echo "Agent URL:   ${AGENT_URL}"
  echo "Agent model: ${AGENT_MODEL}"
  echo "反思 URL:    ${REFL_URL}"
  echo "反思 model:  ${REFL_MODEL}"
  echo "temperature: ${TEMP}"
  echo "max_tokens:  ${MAX_TOKENS}"
  echo "thinking:    ${THINKING}"
  echo "EVAL_ANSWER_LLM_EXTRACT:   ${LLM_EXTRACT}"
  echo "EVAL_ANSWER_LLM_THRESHOLD: ${LLM_THRESHOLD}"
  echo "答案抽取:      finalize_answer（不使用 gold，gold 仅事后 EM）"
  echo "闭卷:        eval_benchmark 不向 Agent/反思传入 gold（标答仅事后 EM）"
  echo "日志目录:    logs/eval/${RUN_NAME}_${MODE}_${TOOLS}/"
  echo "控制台日志:  ${LOG}"
  echo "对照:        32B+32B=run_2wiki_32b_eval | 纯 9B+9B=run_2wiki_9b_ablation"
  echo "================================================================"
} >"$LOG"

nohup "$PYTHON" scripts/run_2wiki_9b_agent_32b_refl.py \
  --jsonl "$JSONL" \
  --run-name "$RUN_NAME" \
  --mode "$MODE" \
  --tools "$TOOLS" \
  --max-steps "$MAX_STEPS" \
  --workers "$WORKERS" \
  "${EXTRA[@]}" \
  >>"$LOG" 2>&1 &

echo "已启动 PID=$!  ${RESUME_FLAG}"
echo "tail -f $LOG"
