#!/usr/bin/env bash
# 后台启动 2Wiki 9B+9B 消融评测，日志写入 logs/eval/runs/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUNS_DIR="$ROOT/logs/eval/runs"
mkdir -p "$RUNS_DIR"
LOG="$RUNS_DIR/2wiki_train_100_9b_ablation.log"
PYTHON="${PYTHON:-/opt/conda/envs/zijinhua/bin/python}"

if ! curl -sf http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
  echo "错误: :8001 9B vLLM 未就绪，请先 bash scripts/start_vllm.sh"
  exit 1
fi

EXTRA_ARGS=("$@")
if [[ " ${EXTRA_ARGS[*]:-} " != *" --resume "* ]]; then
  echo "提示: 非 --resume 将清空 logs/eval/2wiki_train_100_9b_ablation_full_search/"
fi

nohup "$PYTHON" scripts/run_2wiki_9b_ablation.py \
  --mode full \
  --tools search \
  --workers 1 \
  --max-steps 6 \
  "${EXTRA_ARGS[@]}" \
  >"$LOG" 2>&1 &

echo "已启动 PID=$!  日志: $LOG"
echo "查看: tail -f $LOG"
