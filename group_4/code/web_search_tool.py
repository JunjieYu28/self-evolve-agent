"""在线搜索：search-proxy 代理模式 或 Serper+Jina 直连模式。"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests

from tools import BaseTool

logger = logging.getLogger(__name__)

SERPER_SEARCH_URL = "https://google.serper.dev/search"
SERPER_LENS_URL = "https://google.serper.dev/lens"
JINA_READER_BASE = "https://r.jina.ai/"


def _env_bool(name: str, default: bool = True) -> bool:
    return os.getenv(name, str(default)).lower() not in ("0", "false", "no")


class WebSearchTool(BaseTool):
    """
    联网搜索工具。

    优先使用 SEARCH_PROXY_URL（harness search-proxy，GPU 无外网场景）。
    未配置代理时回退 Serper + Jina 直连（需 SERPER_API_KEY）。
    """

    def __init__(
        self,
        search_proxy_url: str | None = None,
        search_proxy_token: str | None = None,
        serper_api_key: str | None = None,
        jina_api_key: str | None = None,
        timeout: float = 30.0,
        proxy_timeout: float = 120.0,
        jina_timeout: float = 45.0,
    ) -> None:
        self.search_proxy_url = (
            search_proxy_url or os.getenv("SEARCH_PROXY_URL", "")
        ).rstrip("/")
        self.search_proxy_token = search_proxy_token or os.getenv(
            "SEARCH_PROXY_TOKEN", ""
        ) or os.getenv("PROXY_API_TOKEN", "")
        self.proxy_verify_ssl = _env_bool("SEARCH_PROXY_VERIFY_SSL", True)
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY", "")
        self.jina_api_key = jina_api_key or os.getenv("JINA_API_KEY", "")
        self.timeout = timeout
        self.proxy_timeout = float(
            os.getenv("SEARCH_PROXY_TIMEOUT", str(proxy_timeout))
        )
        self.jina_timeout = jina_timeout

        try:
            extra = os.getenv("SEARCH_PROXY_EXTRA_HEADERS", "") or "{}"
            self.proxy_extra_headers: dict[str, str] = json.loads(extra)
            if not isinstance(self.proxy_extra_headers, dict):
                self.proxy_extra_headers = {}
        except json.JSONDecodeError:
            self.proxy_extra_headers = {}

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web. Text search; image search when image_url or "
            "image_path is set. Returns ranked results with optional content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords."},
                "image_url": {"type": "string", "description": "Public image URL."},
                "image_path": {"type": "string", "description": "Local image path."},
                "top_k": {"type": "integer", "default": 5},
                "fetch": {"type": "boolean", "default": True},
                "max_chars": {"type": "integer", "default": 3000},
            },
            "required": ["query"],
        }

    def _use_proxy(self) -> bool:
        return bool(self.search_proxy_url)

    def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "").strip()
        if not query:
            return "Error: `query` is required."
        if not self._use_proxy() and not self.serper_api_key:
            return (
                "Error: set SEARCH_PROXY_URL or SERPER_API_KEY in .env"
            )

        top_k = max(1, min(int(kwargs.get("top_k", 5)), 10))
        fetch = bool(kwargs.get("fetch", True))
        max_chars = int(kwargs.get("max_chars", 3000))
        image_url = kwargs.get("image_url")
        image_path = kwargs.get("image_path")

        try:
            if self._use_proxy():
                if image_url or image_path:
                    results = self._proxy_image_search(
                        query, image_url, image_path, top_k, fetch, max_chars
                    )
                else:
                    results = self._proxy_text_search(
                        query, top_k, fetch, max_chars
                    )
            elif image_url or image_path:
                lens_url = self._resolve_image_url_direct(image_url, image_path)
                hits = self._serper_lens(lens_url, top_k)
                results = self._normalize_hits_direct(hits, fetch, max_chars)
            else:
                hits = self._serper_text(query, top_k)
                results = self._normalize_hits_direct(hits, fetch, max_chars)

            return self._format_results(results)
        except Exception as exc:
            logger.exception("web_search failed")
            return f"Error: web_search failed — {exc}"

    # ------------------------------------------------------------------
    # Proxy mode (search-proxy)
    # ------------------------------------------------------------------
    def _proxy_headers(self, json_body: bool = True) -> dict[str, str]:
        headers: dict[str, str] = {}
        if json_body:
            headers["Content-Type"] = "application/json"
        if self.search_proxy_token:
            headers["Authorization"] = f"Bearer {self.search_proxy_token}"
        headers.update(self.proxy_extra_headers)
        return headers

    def _proxy_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.search_proxy_url}{path}"
        resp = requests.post(
            url,
            json=payload,
            headers=self._proxy_headers(json_body=True),
            timeout=self.proxy_timeout,
            verify=self.proxy_verify_ssl,
        )
        resp.raise_for_status()
        return resp.json()

    def _proxy_parse_results(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        if not data.get("ok", False):
            err = data.get("error", "unknown proxy error")
            return [
                {
                    "rank": 1,
                    "title": "",
                    "url": "",
                    "snippet": f"[proxy-error] {err}",
                }
            ]
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

    def _proxy_text_search(
        self, query: str, top_k: int, fetch: bool, max_chars: int
    ) -> list[dict[str, Any]]:
        data = self._proxy_post(
            "/search/text",
            {
                "query": query,
                "top_k": top_k,
                "fetch": fetch,
                "max_chars": max_chars,
            },
        )
        return self._proxy_parse_results(data)

    def _proxy_image_search(
        self,
        query: str,
        image_url: str | None,
        image_path: str | None,
        top_k: int,
        fetch: bool,
        max_chars: int,
    ) -> list[dict[str, Any]]:
        resolved = self._resolve_image_url_proxy(image_url, image_path)
        data = self._proxy_post(
            "/search/image",
            {
                "query": query,
                "image": resolved,
                "top_k": top_k,
                "fetch": fetch,
                "max_chars": max_chars,
            },
        )
        return self._proxy_parse_results(data)

    def _resolve_image_url_proxy(
        self, image_url: str | None, image_path: str | None
    ) -> str:
        if image_url and str(image_url).startswith(("http://", "https://")):
            return str(image_url)
        if image_path:
            path = Path(image_path)
            if path.is_file():
                return self._proxy_upload_image(path)
        raise FileNotFoundError("image_url or valid image_path required")

    def _proxy_upload_image(self, path: Path) -> str:
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
        with path.open("rb") as fh:
            resp = requests.post(
                f"{self.search_proxy_url}/upload_image",
                files={"file": (path.name, fh, mime)},
                headers=self._proxy_headers(json_body=False),
                timeout=self.proxy_timeout,
                verify=self.proxy_verify_ssl,
            )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok", False):
            raise RuntimeError(data.get("error", "upload_image failed"))
        url = data.get("url", "")
        if not url:
            raise RuntimeError("upload_image returned empty url")
        return url

    def health_check(self) -> dict[str, Any]:
        """检查 search-proxy 是否可用。"""
        url = f"{self.search_proxy_url}/health"
        resp = requests.get(
            url,
            headers=self._proxy_headers(json_body=False),
            timeout=min(self.proxy_timeout, 30),
            verify=self.proxy_verify_ssl,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Direct mode (Serper + Jina)
    # ------------------------------------------------------------------
    def _serper_post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }
        resp = requests.post(
            url, json=payload, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def _serper_text(self, query: str, top_k: int) -> list[dict[str, Any]]:
        data = self._serper_post(SERPER_SEARCH_URL, {"q": query, "num": top_k})
        return list(data.get("organic", []) or [])

    def _serper_lens(self, image_url: str, top_k: int) -> list[dict[str, Any]]:
        data = self._serper_post(SERPER_LENS_URL, {"url": image_url})
        items = data.get("organic") or data.get("visual_matches") or []
        return list(items[:top_k])

    def _jina_fetch(self, url: str, max_chars: int) -> str:
        if not url:
            return ""
        reader_url = JINA_READER_BASE + url
        headers = {"Accept": "text/plain"}
        if self.jina_api_key:
            headers["Authorization"] = f"Bearer {self.jina_api_key}"
        try:
            resp = requests.get(
                reader_url, headers=headers, timeout=self.jina_timeout
            )
            resp.raise_for_status()
            text = resp.text or ""
            if max_chars and len(text) > max_chars:
                text = text[:max_chars] + f"\n...[truncated at {max_chars}]"
            return text
        except Exception as exc:
            return f"[jina-error] {exc}"

    def _resolve_image_url_direct(
        self, image_url: str | None, image_path: str | None
    ) -> str:
        if image_url and str(image_url).startswith(("http://", "https://")):
            return str(image_url)
        if image_path:
            path = Path(image_path)
            if path.is_file():
                return self._upload_image_direct(path)
        raise FileNotFoundError("image_url or valid image_path required")

    def _upload_image_direct(self, path: Path) -> str:
        from services.image_upload import upload_path

        return upload_path(path)

    def _normalize_hits_direct(
        self,
        hits: list[dict[str, Any]],
        fetch: bool,
        max_chars: int,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for i, hit in enumerate(hits, 1):
            url = hit.get("link") or hit.get("url") or ""
            entry: dict[str, Any] = {
                "rank": i,
                "title": hit.get("title", ""),
                "url": url,
                "snippet": hit.get("snippet", "") or hit.get("description", ""),
            }
            if fetch and url:
                entry["content"] = self._jina_fetch(url, max_chars)
            results.append(entry)
        return results

    def _format_results(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return "No search results."
        mode = "proxy" if self._use_proxy() else "direct"
        lines = [f"Found {len(results)} result(s) [{mode}]:"]
        for r in results:
            lines.append(f"\n{r['rank']}. {r['title']}")
            lines.append(f"   URL: {r['url']}")
            if r.get("snippet"):
                lines.append(f"   Snippet: {r['snippet'][:200]}")
            if r.get("content"):
                lines.append(f"   Content: {str(r['content'])[:500]}...")
        return "\n".join(lines)
