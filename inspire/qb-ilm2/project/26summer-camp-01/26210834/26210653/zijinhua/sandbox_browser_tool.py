"""沙盒浏览器工具：单页抓取 + 并发多页。"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import aiohttp
import requests

from tools import BaseTool


class SandboxBrowserTool(BaseTool):
    """
    沙盒浏览器访问工具骨架。
    配置 SANDBOX_BROWSER_API_URL / SANDBOX_BROWSER_API_KEY 接入真实沙盒服务。
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_concurrency: int = 5,
    ) -> None:
        self.api_url = (api_url or os.getenv("SANDBOX_BROWSER_API_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("SANDBOX_BROWSER_API_KEY", "")
        self.timeout = timeout
        self.max_concurrency = max_concurrency

    @property
    def name(self) -> str:
        return "sandbox_browser"

    @property
    def description(self) -> str:
        return (
            "Access URLs in a sandboxed browser environment. "
            "Use `url` for a single page, or `urls` for concurrent fetching."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Single URL to fetch.",
                },
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Multiple URLs to fetch concurrently.",
                },
                "extract_text": {
                    "type": "boolean",
                    "description": "Whether to extract main text content only.",
                    "default": True,
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters per page.",
                    "default": 8000,
                },
            },
        }

    def execute(self, **kwargs: Any) -> str:
        url = kwargs.get("url")
        urls = kwargs.get("urls") or []
        extract_text = bool(kwargs.get("extract_text", True))
        max_chars = int(kwargs.get("max_chars", 8000))

        if url:
            urls = [url] + list(urls)
        if not urls:
            return "Error: provide `url` or `urls`."

        try:
            if len(urls) == 1:
                data = self.fetch_url(urls[0], extract_text=extract_text, max_chars=max_chars)
                return self._format_page_result(urls[0], data)
            results = asyncio.run(
                self.fetch_urls_async(urls, extract_text=extract_text, max_chars=max_chars)
            )
            return self._format_multi_results(results)
        except Exception as exc:
            return f"Error: sandbox_browser failed — {exc}"

    def fetch_url(
        self,
        url: str,
        extract_text: bool = True,
        max_chars: int = 8000,
    ) -> dict[str, Any]:
        """同步访问单个 URL，返回解析后的 JSON。"""
        if not self.api_url:
            return self._mock_page(url)

        payload = {
            "url": url,
            "extract_text": extract_text,
            "max_chars": max_chars,
        }
        return self._post_sync("/v1/browse", payload)

    async def fetch_urls_async(
        self,
        urls: list[str],
        extract_text: bool = True,
        max_chars: int = 8000,
    ) -> list[dict[str, Any]]:
        """并发访问多个 URL（aiohttp 异步接口）。"""
        if not self.api_url:
            return [self._mock_page(u) | {"url": u} for u in urls]

        sem = asyncio.Semaphore(self.max_concurrency)
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                self._fetch_one_async(session, sem, u, extract_text, max_chars)
                for u in urls
            ]
            return await asyncio.gather(*tasks)

    async def _fetch_one_async(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        url: str,
        extract_text: bool,
        max_chars: int,
    ) -> dict[str, Any]:
        async with sem:
            api_url = f"{self.api_url}/v1/browse"
            payload = {
                "url": url,
                "extract_text": extract_text,
                "max_chars": max_chars,
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = self.api_key

            async with session.post(api_url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
                data.setdefault("url", url)
                return data

    def _post_sync(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = self.api_key
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _extract_text_from_response(self, data: dict[str, Any]) -> str:
        for key in ("text", "content", "main_text", "body"):
            if key in data and data[key]:
                return str(data[key])
        return json.dumps(data, ensure_ascii=False)[:2000]

    def _format_page_result(self, url: str, data: dict[str, Any]) -> str:
        text = self._extract_text_from_response(data)
        title = data.get("title", "")
        lines = [f"URL: {url}"]
        if title:
            lines.append(f"Title: {title}")
        lines.append(f"Content:\n{text}")
        return "\n".join(lines)

    def _format_multi_results(self, results: list[dict[str, Any]]) -> str:
        parts = []
        for i, data in enumerate(results, 1):
            url = data.get("url", f"page_{i}")
            parts.append(f"=== Page {i}: {url} ===\n{self._format_page_result(url, data)}")
        return "\n\n".join(parts)

    def _mock_page(self, url: str) -> dict[str, Any]:
        return {
            "url": url,
            "title": "[Mock Sandbox Browser]",
            "text": (
                f"Mock page content for {url}. "
                "Set SANDBOX_BROWSER_API_URL to enable real browsing."
            ),
        }
