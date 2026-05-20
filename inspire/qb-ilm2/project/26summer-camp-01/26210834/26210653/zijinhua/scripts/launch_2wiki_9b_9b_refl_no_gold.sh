#!/usr/bin/env bash
# 2Wiki full+反思，闭卷（eval_benchmark 不向 Agent/反思传 gold）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export JSONL="${JSONL:-data/2wiki.jsonl}"
export RUN_NAME="${RUN_NAME:-2wiki_9b_9b_refl_no_gold}"
export LOG="${LOG:-$ROOT/logs/eval/runs/${RUN_NAME}.log}"
export MODE="${MODE:-full}"
export TOOLS="${TOOLS:-search}"
export AGENT_URL="${AGENT_URL:-http://127.0.0.1:8001/v1}"
export REFL_URL="${REFL_URL:-http://127.0.0.1:8004/v1}"

exec bash "$ROOT/scripts/launch_2wiki_9b_agent_32b_refl.sh" "$@"
