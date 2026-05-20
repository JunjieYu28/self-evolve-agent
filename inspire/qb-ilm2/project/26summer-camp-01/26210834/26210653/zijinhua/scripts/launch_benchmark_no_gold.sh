#!/usr/bin/env bash
# 打榜 benchmark.csv（nohup），答案抽取无 gold；Agent 走 GPU1 :8002，减轻与 2Wiki(:8001) 争抢。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/opt/conda/envs/zijinhua/bin/python}"
GROUP="${GROUP:-004_no_gold}"
MODE="${MODE:-full}"
TOOLS="${TOOLS:-search}"
WORKERS="${WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-6}"
AGENT_URL="${AGENT_URL:-http://127.0.0.1:8002/v1}"
AGENT_MODEL="${AGENT_MODEL:-qwen-3.5}"
REFL_URL="${REFL_URL:-http://127.0.0.1:8004/v1}"
REFL_MODEL="${REFL_MODEL:-qwen-3.5}"
LOG="${LOG:-$ROOT/logs/benchmark/runs/group_${GROUP}.log}"

mkdir -p "$ROOT/logs/benchmark/runs"

api="${AGENT_URL%/}"
[[ "$api" != */v1 ]] && api="${api}/v1"
if ! curl -sf "${api}/models" >/dev/null 2>&1; then
  echo "错误: Agent API ${api} 未就绪（建议 GPU1: CUDA_VISIBLE_DEVICES=1 VLLM_PORT=8002）"
  exit 1
fi
refl_api="${REFL_URL%/}"
[[ "$refl_api" != */v1 ]] && refl_api="${refl_api}/v1"
if [[ "$MODE" == "full" ]]; then
  if ! curl -sf "${refl_api}/models" >/dev/null 2>&1; then
    echo "反思 API ${refl_api} 未就绪，尝试启动专用 9B 反思 vLLM..."
    bash "$ROOT/scripts/start_vllm_9b_reflection.sh" || true
    for _ in $(seq 1 60); do
      curl -sf "${refl_api}/models" >/dev/null 2>&1 && break
      sleep 5
    done
    if ! curl -sf "${refl_api}/models" >/dev/null 2>&1; then
      echo "错误: 反思 API ${refl_api} 仍未就绪（请先 bash scripts/start_vllm_9b_reflection.sh）"
      exit 1
    fi
  fi
fi
if ! curl -sf http://127.0.0.1:8090/search/text -X POST -H 'Content-Type: application/json' \
  -d '{"query":"test","top_k":1,"fetch":false}' >/dev/null 2>&1; then
  echo "警告: search-proxy :8090 可能异常，尝试 bash scripts/start_search_proxy.sh"
fi

EXTRA=("$@")
START_TS="$(date -Iseconds)"
{
  echo "================================================================"
  echo "Benchmark 打榜 — benchmark.csv"
  echo "启动时间: ${START_TS}"
  echo "----------------------------------------------------------------"
  echo "组号:              ${GROUP}  -> group_${GROUP}.json/csv/zip"
  echo "数据:              benchmark.csv"
  echo "mode:              ${MODE}"
  echo "tools:             ${TOOLS}"
  echo "max_steps:         ${MAX_STEPS}"
  echo "workers:           ${WORKERS}"
  echo "Agent URL:         ${AGENT_URL}  (建议 GPU1 :8002，避开 2Wiki 占用的 :8001/GPU0)"
  echo "Agent model:       ${AGENT_MODEL}"
  echo "反思 URL:          ${REFL_URL}  (专用 9B，GPU2 :8004)"
  echo "反思 model:        ${REFL_MODEL}"
  echo "答案抽取:          finalize_answer（无 gold）"
  echo "EVAL_ANSWER_LLM_THRESHOLD: ${EVAL_ANSWER_LLM_THRESHOLD:-20}"
  echo "日志:              logs/benchmark/group_${GROUP}/"
  echo "本文件:            ${LOG}"
  echo "GPU 参考: Agent :8002(GPU1) | 反思 :8004(GPU2) | 2Wiki Agent :8001(GPU0)"
  echo "================================================================"
} >"$LOG"

export LLM_BACKEND=sglang
export LLM_BASE_URL="${AGENT_URL}"
export MODEL_NAME="${AGENT_MODEL}"
export REFLECTION_LLM_BASE_URL="${REFL_URL}"
export REFLECTION_MODEL_NAME="${REFL_MODEL}"
export LLM_ENABLE_THINKING=false
export REFLECTION_ENABLE_THINKING=false
export EVAL_ANSWER_LLM_EXTRACT="${EVAL_ANSWER_LLM_EXTRACT:-auto}"
export EVAL_ANSWER_LLM_THRESHOLD="${EVAL_ANSWER_LLM_THRESHOLD:-20}"

nohup "$PYTHON" scripts/run_benchmark.py \
  --group "$GROUP" \
  --mode "$MODE" \
  --tools "$TOOLS" \
  --max-steps "$MAX_STEPS" \
  --workers "$WORKERS" \
  "${EXTRA[@]}" \
  >>"$LOG" 2>&1 &

echo "已启动 PID=$!  GROUP=${GROUP}"
echo "tail -f $LOG"
