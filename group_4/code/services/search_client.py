"""search-proxy HTTP 客户端（移植自 harness-sii/tools/search_tool.py 代理模式）。"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

SEARCH_PROXY_URL = os.getenv("SEARCH_PROXY_URL", "http://127.0.0.1:8090").rstrip(
    "/"
)
SEARCH_PROXY_TOKEN = os.getenv("SEARCH_PROXY_TOKEN", "") or os.getenv(
    "PROXY_API_TOKEN", ""
)
PROXY_HTTP_TIMEOUT = float(os.getenv("SEARCH_PROXY_TIMEOUT", "120"))
SEARCH_PROXY_VERIFY_SSL = os.getenv("SEARCH_PROXY_VERIFY_SSL", "true").lower() not in (
    "0",
    "false",
    "no",
)

try:
    SEARCH_PROXY_EXTRA_HEADERS: dict[str, str] = json.loads(
        os.getenv("SEARCH_PROXY_EXTRA_HEADERS", "") or "{}"
    )
except json.JSONDecodeError:
    SEARCH_PROXY_EXTRA_HEADERS = {}


def _headers(json_body: bool = True) -> dict[str, str]:
    h: dict[str, str] = {}
    if json_body:
        h["Content-Type"] = "application/json"
    if SEARCH_PROXY_TOKEN:
        h["Authorization"] = f"Bearer {SEARCH_PROXY_TOKEN}"
    h.update(SEARCH_PROXY_EXTRA_HEADERS)
    return h


def proxy_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{SEARCH_PROXY_URL}{path}"
    resp = requests.post(
        url,
        json=payload,
        headers=_headers(True),
        timeout=PROXY_HTTP_TIMEOUT,
        verify=SEARCH_PROXY_VERIFY_SSL,
    )
    resp.raise_for_status()
    return resp.json()


def proxy_search(path: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = proxy_post(path, payload)
    if not data.get("ok", False):
        err = data.get("error", "unknown proxy error")
        logger.warning("search-proxy %s failed: %s", path, err)
        return [{"rank": 1, "title": "", "url": "", "snippet": f"[proxy-error] {err}"}]

    out: list[dict[str, Any]] = []
    for hit in data.get("results", []) or []:
        entry: dict[str, Any] = {
            "rank": hit.get("rank", len(out) + 1),
            "title": hit.get("title", ""),
            "url": hit.get("url", ""),
            "snippet": hit.get("snippet", ""),
        }
        if hit.get("content") is not None:
            entry["content"] = hit["content"]
        out.append(entry)
    return out


def upload_image(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"
    with path.open("rb") as fh:
        resp = requests.post(
            f"{SEARCH_PROXY_URL}/upload_image",
            files={"file": (path.name, fh, mime)},
            headers=_headers(False),
            timeout=PROXY_HTTP_TIMEOUT,
            verify=SEARCH_PROXY_VERIFY_SSL,
        )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok", False):
        raise RuntimeError(data.get("error", "upload_image failed"))
    url = data.get("url", "")
    if not url:
        raise RuntimeError("upload_image returned empty url")
    return url


def fetch_page_content(url: str, max_chars: int = 500) -> str | None:
    """对单 URL 调用 search-proxy /fetch（精排后按需抓正文）。"""
    if not url or not url.strip():
        return None
    try:
        data = proxy_post(
            "/fetch",
            {"url": url.strip(), "max_chars": int(max_chars)},
        )
        if data.get("ok"):
            return data.get("content") or ""
    except Exception as exc:
        logger.debug("fetch_page_content failed for %s: %s", url[:80], exc)
    return None


def resolve_image(image: str) -> str:
    image = image.strip()
    if image.startswith(("http://", "https://")):
        return image
    p = Path(image).expanduser()
    if p.is_file():
        return upload_image(p)
    raise ValueError(f"invalid image: {image!r}")
