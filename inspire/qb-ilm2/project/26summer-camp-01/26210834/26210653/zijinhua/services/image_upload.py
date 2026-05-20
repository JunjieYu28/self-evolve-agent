"""本地图片上传公网图床（供 search-proxy / 直连兜底共用）。"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

TMPFILES_API = "https://tmpfiles.org/api/v1/upload"
UGUU_API = "https://uguu.se/upload"
DEFAULT_UPLOADER = os.getenv("IMAGE_UPLOADER", "uguu").strip().lower()
DEFAULT_TIMEOUT = float(os.getenv("UPLOAD_TIMEOUT", "60"))
USER_AGENT = "zijinhua-agent/1.0"


def _requests_proxies() -> dict[str, str] | None:
    http_p = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
    https_p = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY") or http_p
    if http_p or https_p:
        return {"http": http_p or https_p, "https": https_p or http_p}
    return None


def tmpfiles_direct_url(page_url: str, filename: str) -> str:
    """将 tmpfiles 页面 URL 转为 Serper Lens 可用的图片直链。"""
    m = re.search(r"tmpfiles\.org/([^/]+)/", page_url.strip())
    if not m:
        return page_url
    fid = m.group(1)
    safe_name = Path(filename).name or "image.bin"
    return f"https://tmpfiles.org/dl/{fid}/{safe_name}"


def upload_bytes_tmpfiles(
    file_bytes: bytes,
    filename: str,
    *,
    timeout: float | None = None,
) -> str:
    mime, _ = mimetypes.guess_type(filename)
    mime = mime or "application/octet-stream"
    safe_name = Path(filename).name or "image.bin"
    t = timeout if timeout is not None else DEFAULT_TIMEOUT
    resp = requests.post(
        TMPFILES_API,
        files={"file": (safe_name, file_bytes, mime)},
        headers={"User-Agent": USER_AGENT},
        proxies=_requests_proxies(),
        timeout=t,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"tmpfiles upload failed: {payload!r}")
    page_url = (payload.get("data") or {}).get("url", "").strip()
    if not page_url.startswith("http"):
        raise RuntimeError(f"tmpfiles invalid url: {payload!r}")
    direct = tmpfiles_direct_url(page_url, safe_name)
    head = requests.head(
        direct,
        headers={"User-Agent": USER_AGENT},
        proxies=_requests_proxies(),
        timeout=min(t, 30),
        allow_redirects=True,
    )
    head.raise_for_status()
    logger.info("tmpfiles upload %s -> %s", safe_name, direct)
    return direct


def upload_bytes_0x0(
    file_bytes: bytes,
    filename: str,
    *,
    timeout: float | None = None,
) -> str:
    mime, _ = mimetypes.guess_type(filename)
    mime = mime or "application/octet-stream"
    t = timeout if timeout is not None else DEFAULT_TIMEOUT
    resp = requests.post(
        "https://0x0.st",
        files={"file": (Path(filename).name, file_bytes, mime)},
        headers={"User-Agent": USER_AGENT},
        proxies=_requests_proxies(),
        timeout=t,
    )
    resp.raise_for_status()
    url = resp.text.strip()
    if not url.startswith("http"):
        raise RuntimeError(f"0x0.st upload failed: {url!r}")
    return url


def upload_bytes_uguu(
    file_bytes: bytes,
    filename: str,
    *,
    timeout: float | None = None,
) -> str:
    mime, _ = mimetypes.guess_type(filename)
    mime = mime or "application/octet-stream"
    safe_name = Path(filename).name or "image.bin"
    t = timeout if timeout is not None else DEFAULT_TIMEOUT
    resp = requests.post(
        UGUU_API,
        files={"files[]": (safe_name, file_bytes, mime)},
        headers={"User-Agent": USER_AGENT},
        proxies=_requests_proxies(),
        timeout=t,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"uguu upload failed: {payload!r}")
    files = payload.get("files", [])
    if not files or not files[0].get("url"):
        raise RuntimeError(f"uguu returned no url: {payload!r}")
    url = files[0]["url"]
    logger.info("uguu upload %s -> %s", safe_name, url)
    return url


def upload_bytes(
    file_bytes: bytes,
    filename: str,
    *,
    backend: str | None = None,
    timeout: float | None = None,
) -> str:
    """上传字节到公网图床，返回 https 直链。"""
    name = (backend or DEFAULT_UPLOADER).strip().lower()
    if name == "tmpfiles":
        return upload_bytes_tmpfiles(file_bytes, filename, timeout=timeout)
    if name == "0x0":
        return upload_bytes_0x0(file_bytes, filename, timeout=timeout)
    if name == "uguu":
        return upload_bytes_uguu(file_bytes, filename, timeout=timeout)
    raise RuntimeError(
        f"Unsupported IMAGE_UPLOADER={name!r} (supported: tmpfiles, 0x0, uguu)"
    )


def upload_path(path: Path | str, *, backend: str | None = None) -> str:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(p)
    return upload_bytes(p.read_bytes(), p.name, backend=backend)
