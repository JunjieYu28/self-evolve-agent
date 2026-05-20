"""browser-service HTTP 客户端（移植自 harness-sii/sandbox_client.py）。"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

SANDBOX_BASE_URL = os.getenv("SANDBOX_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
SANDBOX_API_TOKEN = os.getenv("SANDBOX_API_TOKEN", "") or os.getenv(
    "BROWSER_API_TOKEN", ""
)
SANDBOX_HTTP_TIMEOUT = float(os.getenv("SANDBOX_HTTP_TIMEOUT", "120"))


class BrowserClient:
    """browser-service 同步 HTTP 封装，单 session 跨 tool 调用保持同一 tab。"""

    def __init__(
        self,
        base_url: str,
        token: str = "",
        timeout: float = SANDBOX_HTTP_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"Content-Type": "application/json"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._session_id: Optional[str] = None
        self._lock = threading.Lock()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.request(
                method,
                url,
                headers=self._headers,
                timeout=kwargs.pop("timeout", self.timeout),
                **kwargs,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"{method} {url} failed: {type(exc).__name__}: {exc}"
            ) from exc
        if resp.status_code >= 400:
            raise RuntimeError(
                f"{method} {url} -> {resp.status_code}: {resp.text[:500]}"
            )
        return resp.json()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def ensure_session(self) -> str:
        with self._lock:
            if self._session_id:
                return self._session_id
            r = self._request("POST", "/session/create")
            sid = r.get("session_id") or ""
            if not sid:
                raise RuntimeError(f"unexpected create_session: {r}")
            self._session_id = sid
            return sid

    def navigate(
        self,
        url: str,
        wait_until: str = "domcontentloaded",
        timeout_ms: int = 30000,
        tab_id: Optional[str] = None,
    ) -> dict:
        return self._request(
            "POST",
            "/browser/navigate",
            json={
                "url": url,
                "session_id": self.ensure_session(),
                "tab_id": tab_id,
                "wait_until": wait_until,
                "timeout_ms": timeout_ms,
            },
            timeout=max(self.timeout, timeout_ms / 1000 + 15),
        )

    def get_text(
        self,
        selector: Optional[str] = None,
        tab_id: Optional[str] = None,
    ) -> str:
        r = self._request(
            "POST",
            "/browser/get_text",
            json={
                "session_id": self.ensure_session(),
                "tab_id": tab_id,
                "selector": selector,
            },
        )
        return r.get("text", "")

    def click(
        self,
        selector: str,
        timeout_ms: int = 10000,
        tab_id: Optional[str] = None,
    ) -> dict:
        return self._request(
            "POST",
            "/browser/click",
            json={
                "selector": selector,
                "session_id": self.ensure_session(),
                "tab_id": tab_id,
                "timeout_ms": timeout_ms,
            },
        )

    def type_text(
        self,
        selector: str,
        text: str,
        clear: bool = True,
        press_enter: bool = False,
        tab_id: Optional[str] = None,
    ) -> dict:
        return self._request(
            "POST",
            "/browser/type",
            json={
                "selector": selector,
                "text": text,
                "session_id": self.ensure_session(),
                "tab_id": tab_id,
                "clear": clear,
                "press_enter": press_enter,
            },
        )

    def title(self, tab_id: Optional[str] = None) -> dict:
        params: dict[str, str] = {"session_id": self.ensure_session()}
        if tab_id:
            params["tab_id"] = tab_id
        return self._request("GET", "/browser/title", params=params)

    def new_tab(self, url: Optional[str] = None) -> dict:
        return self._request(
            "POST",
            "/tab/new",
            json={"session_id": self.ensure_session(), "url": url},
        )

    def close_tab(self, tab_id: str) -> dict:
        return self._request(
            "POST",
            "/tab/close",
            json={"session_id": self.ensure_session(), "tab_id": tab_id},
        )

    def eval_js(self, script: str, tab_id: Optional[str] = None) -> Any:
        r = self._request(
            "POST",
            "/browser/eval",
            json={
                "script": script,
                "session_id": self.ensure_session(),
                "tab_id": tab_id,
            },
        )
        return r.get("result")


_client: Optional[BrowserClient] = None
_lock = threading.Lock()


def get_browser_client(base_url: Optional[str] = None) -> BrowserClient:
    global _client
    url = (base_url or SANDBOX_BASE_URL).rstrip("/")
    with _lock:
        if _client is None:
            _client = BrowserClient(url, token=SANDBOX_API_TOKEN)
        return _client
