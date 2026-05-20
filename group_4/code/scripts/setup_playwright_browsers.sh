#!/usr/bin/env bash
# 一次性：从 npmmirror 拉 Chromium 1148（与 playwright==1.49.1 匹配）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/playwright_env.sh"

ZIP_URL="https://registry.npmmirror.com/-/binary/playwright/builds/chromium/1148/chromium-linux.zip"
DEST="${PLAYWRIGHT_BROWSERS_PATH}/chromium-1148"
ZIP="${PLAYWRIGHT_BROWSERS_PATH}/dl/chromium-linux.zip"

if [[ -x "${DEST}/chrome-linux/chrome" ]]; then
  echo "Chromium 1148 已存在: ${DEST}/chrome-linux/chrome"
  exit 0
fi

mkdir -p "${PLAYWRIGHT_BROWSERS_PATH}/dl"
echo "下载 ${ZIP_URL} ..."
curl -L --fail -o "${ZIP}" "${ZIP_URL}"
rm -rf "${DEST}"
mkdir -p "${DEST}"
unzip -q "${ZIP}" -d "${DEST}"
chmod +x "${DEST}/chrome-linux/chrome" 2>/dev/null || true

_hs="${PLAYWRIGHT_BROWSERS_PATH}/chromium_headless_shell-1148/chrome-linux"
mkdir -p "${_hs}"
ln -sf ../../chromium-1148/chrome-linux/chrome "${_hs}/headless_shell"

echo "完成: ${DEST}/chrome-linux/chrome"
echo "安装 Python 包: ${PLAYWRIGHT_PYTHON} -m pip install 'playwright==1.49.1'"
echo "运行时库: conda install -n zijinhua -c conda-forge nss nspr atk at-spi2-atk at-spi2-core libcups libxkbcommon alsa-lib pango cairo xorg-libxcomposite xorg-libxdamage xorg-libxfixes xorg-libxrandr"
