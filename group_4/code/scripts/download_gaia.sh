#!/usr/bin/env bash
# 下载 GAIA：需代理 + HF_TOKEN
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "${ROOT}/scripts/set_proxy.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/set_proxy.sh"
fi

export HF_HOME="${HF_HOME:-${ROOT}/huggingface_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
unset HF_ENDPOINT

exec python3 "${ROOT}/scripts/download_gaia.py" "$@"
