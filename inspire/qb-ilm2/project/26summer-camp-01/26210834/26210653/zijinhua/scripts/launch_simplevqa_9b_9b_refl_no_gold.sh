#!/usr/bin/env bash
# SimpleVQA full+反思，闭卷（eval_benchmark 不向 Agent/反思传 gold）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 图搜上传需代理（与 rerun 一致）
if [[ -f "${ROOT}/scripts/set_proxy.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/set_proxy.sh"
fi
export IMAGE_UPLOADER="${IMAGE_UPLOADER:-uguu}"

export RUN_NAME="${RUN_NAME:-simplevqa_9b_9b_refl_no_gold}"
export LOG="${LOG:-$ROOT/logs/eval/runs/${RUN_NAME}.log}"
export MODE="${MODE:-full}"
export TOOLS="${TOOLS:-search}"

exec bash "$ROOT/scripts/launch_simplevqa_9b_9b_refl.sh" "$@"
