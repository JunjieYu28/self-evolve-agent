# Conda 用户态 Playwright（无需 sudo apt install-deps）
# 用法: source scripts/playwright_env.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-${ROOT}/.playwright-browsers}"
export TMPDIR="${TMPDIR:-${ROOT}/tmp}"
export LD_LIBRARY_PATH="/opt/conda/envs/zijinhua/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# 与 Playwright 1.49.1 匹配的 Chromium revision（npmmirror 可下 chromium-linux.zip）
export PLAYWRIGHT_PYTHON="${PLAYWRIGHT_PYTHON:-/opt/conda/envs/zijinhua/bin/python}"

# headless_shell 镜像未同步时，用完整 chrome 做软链
_hs="${PLAYWRIGHT_BROWSERS_PATH}/chromium_headless_shell-1148/chrome-linux"
_chrome="${PLAYWRIGHT_BROWSERS_PATH}/chromium-1148/chrome-linux/chrome"
if [[ -x "${_chrome}" ]] && [[ ! -x "${_hs}/headless_shell" ]]; then
  mkdir -p "${_hs}"
  ln -sf ../../chromium-1148/chrome-linux/chrome "${_hs}/headless_shell"
fi

mkdir -p "${PLAYWRIGHT_BROWSERS_PATH}" "${TMPDIR}"
