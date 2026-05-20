"""Outbound calls to Serper / Jina / image hosts (0x0.st, tmpfiles.org, …).

This module is the only place that touches the real internet. It is intended
to run on the *CPU host* (which has internet access). The GPU host calls
this service over a private SSH-forwarded port.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import requests

from .config import settings

logger = logging.getLogger("search-proxy.upstream")

SERPER_SEARCH_URL = "https://google.serper.dev/search"


def _requests_proxies() -> dict[str, str] | None:
    """读取环境变量中的 HTTP/HTTPS 代理（本机 clash 等）。"""
    http_p = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
    https_p = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY") or http_p
    if http_p or https_p:
        return {"http": http_p or https_p, "https": https_p or http_p}
    return None
SERPER_LENS_URL = "https://google.serper.dev/lens"
JINA_READER_BASE = "https://r.jina.ai/"


# ---------------------------------------------------------------------------
# Serper
# ---------------------------------------------------------------------------
def _serper_post(url: str, payload: dict) -> dict:
    if not settings.serper_api_key:
        raise RuntimeError("SERPER_API_KEY not set on the proxy host")
    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }
    resp = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=settings.serper_timeout,
        proxies=_requests_proxies(),
    )
    resp.raise_for_status()
    return resp.json()


def serper_search(query: str, top_k: int) -> list[dict]:
    data = _serper_post(SERPER_SEARCH_URL, {"q": query, "num": top_k})
    return list(data.get("organic", []) or [])


def serper_lens(image_url: str, top_k: int) -> list[dict]:
    data = _serper_post(SERPER_LENS_URL, {"url": image_url})
    items = data.get("organic") or data.get("visual_matches") or []
    return list(items[:top_k])


# ---------------------------------------------------------------------------
# Jina Reader
# ---------------------------------------------------------------------------
def jina_fetch(url: str, max_chars: int) -> tuple[str, bool]:
    """Return (content, truncated). On failure raises (caller decides format)."""
    if not url:
        return "", False
    reader_url = JINA_READER_BASE + url
    headers = {"Accept": "text/plain"}
    if settings.jina_api_key:
        headers["Authorization"] = f"Bearer {settings.jina_api_key}"
    resp = requests.get(
        reader_url,
        headers=headers,
        timeout=settings.jina_timeout,
        proxies=_requests_proxies(),
    )
    resp.raise_for_status()
    text = resp.text or ""
    truncated = False
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + f"\n\n...[truncated at {max_chars} chars]"
        truncated = True
    return text, truncated


# ---------------------------------------------------------------------------
# Image hosting (used by the upload endpoint when GPU side has only a local file)
# ---------------------------------------------------------------------------
def upload_image(file_bytes: bytes, filename: str) -> str:
    """委托项目内统一图床上传（默认 tmpfiles.org）。"""
    import sys

    root = Path(__file__).resolve().parents[4]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from services.image_upload import upload_bytes

    backend = (settings.image_uploader or "tmpfiles").strip().lower()
    url = upload_bytes(
        file_bytes,
        filename,
        backend=backend,
        timeout=settings.upload_timeout,
    )
    logger.info("Uploaded %s via %s -> %s", filename, backend, url)
    return url
