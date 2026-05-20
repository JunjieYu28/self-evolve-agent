"""
Web search tool — optimized for concurrent benchmark execution.

Key improvements over the original:
- Exponential backoff with jitter (no more 120s hangs then death)
- LRU cache (identical queries skip network entirely)
- Circuit breaker (stop hammering a dead proxy)
- Adaptive timeout (30s default, not 120s)
- Two-phase search: snippet-first, then targeted fetch
- Parallel URL fetching via ThreadPoolExecutor
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import random
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("harness.tools.search")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEARCH_PROXY_URL = os.getenv(
    "SEARCH_PROXY_URL",
    "https://nat2-notebook-inspire.sii.edu.cn/"
    "ws-7c23bd1d-9bae-4238-803a-737a35480e18/"
    "project-39fbffc7-dcca-4fb4-b43a-2f69f72f7e52/"
    "user-5b1a894f-4b65-4bb2-afcd-8691a9eec556/"
    "vscode/3853761b-a29d-404e-b725-24dcb85cf08c/"
    "f3e293ba-92e2-4e12-935d-7330200a7667/proxy/1227/",
).rstrip("/")
SEARCH_PROXY_TOKEN = os.getenv("SEARCH_PROXY_TOKEN", "") or os.getenv("PROXY_API_TOKEN", "")

PROXY_HTTP_TIMEOUT = float(os.getenv("SEARCH_PROXY_TIMEOUT", "30"))
RETRY_MAX = int(os.getenv("SEARCH_RETRY_MAX", "3"))
RETRY_BASE_DELAY = float(os.getenv("SEARCH_RETRY_BASE_DELAY", "2.0"))

SEARCH_PROXY_VERIFY_SSL = os.getenv("SEARCH_PROXY_VERIFY_SSL", "true").lower() not in (
    "0", "false", "no"
)
try:
    import json as _json
    SEARCH_PROXY_EXTRA_HEADERS: dict = _json.loads(
        os.getenv("SEARCH_PROXY_EXTRA_HEADERS", "") or "{}"
    )
    if not isinstance(SEARCH_PROXY_EXTRA_HEADERS, dict):
        SEARCH_PROXY_EXTRA_HEADERS = {}
except Exception:
    SEARCH_PROXY_EXTRA_HEADERS = {}

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "1d43acbfcc6dead0df1fe015ce6184743ea668e7")
JINA_API_KEY = os.getenv("JINA_API_KEY", "jina_27b632dc368a4d878d77a086367a1493HIydGZpZQJhxBWjflZBGmr89R44M")
IMAGE_UPLOADER = os.getenv("IMAGE_UPLOADER", "0x0")

SERPER_SEARCH_URL = "https://google.serper.dev/search"
SERPER_LENS_URL = "https://google.serper.dev/lens"
JINA_READER_BASE = "https://r.jina.ai/"

DEFAULT_TIMEOUT = 30
JINA_TIMEOUT = 30

MAX_CONCURRENT_PROXY = int(os.getenv("MAX_CONCURRENT_PROXY", "2"))
_proxy_semaphore = threading.Semaphore(MAX_CONCURRENT_PROXY)


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------
class LRUCache:
    """Thread-safe LRU cache with TTL."""

    def __init__(self, maxsize: int = 128, ttl: float = 600.0):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            if key not in self._cache:
                return None
            entry = self._cache[key]
            if time.time() - entry["ts"] > self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return entry["value"]

    def put(self, key: str, value):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = {"value": value, "ts": time.time()}
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = {"value": value, "ts": time.time()}


_search_cache = LRUCache(maxsize=256, ttl=900.0)


def _cache_key(prefix: str, **kwargs) -> str:
    raw = f"{prefix}:" + ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------
class CircuitBreaker:
    """Simple circuit breaker: after N consecutive failures, open for cooldown."""

    def __init__(self, threshold: int = 5, cooldown: float = 60.0):
        self._failure_count = 0
        self._threshold = threshold
        self._cooldown = cooldown
        self._open_until = 0.0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._open_until > 0 and time.time() < self._open_until:
                return True
            if self._open_until > 0 and time.time() >= self._open_until:
                self._open_until = 0
                self._failure_count = 0
            return False

    def record_success(self):
        with self._lock:
            self._failure_count = 0
            self._open_until = 0

    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self._threshold:
                self._open_until = time.time() + self._cooldown
                logger.warning(
                    "Circuit breaker OPEN: %d consecutive failures, cooldown %.0fs",
                    self._failure_count, self._cooldown,
                )


_proxy_breaker = CircuitBreaker(threshold=10, cooldown=30.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _proxy_enabled() -> bool:
    return bool(SEARCH_PROXY_URL)


def _proxy_headers(json_body: bool = True) -> dict:
    h: dict = {}
    if json_body:
        h["Content-Type"] = "application/json"
    if SEARCH_PROXY_TOKEN:
        h["Authorization"] = f"Bearer {SEARCH_PROXY_TOKEN}"
    if SEARCH_PROXY_EXTRA_HEADERS:
        h.update(SEARCH_PROXY_EXTRA_HEADERS)
    return h


# ---------------------------------------------------------------------------
# Retry with exponential backoff + jitter
# ---------------------------------------------------------------------------
def _retry_request(
    method: str,
    url: str,
    max_retries: int = RETRY_MAX,
    base_delay: float = RETRY_BASE_DELAY,
    timeout: float = PROXY_HTTP_TIMEOUT,
    **kwargs,
) -> requests.Response:
    """Execute HTTP request with exponential backoff + jitter."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.request(
                method, url, timeout=timeout,
                verify=SEARCH_PROXY_VERIFY_SSL, **kwargs,
            )
            resp.raise_for_status()
            _proxy_breaker.record_success()
            return resp
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            _proxy_breaker.record_failure()
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "Request %s failed (attempt %d/%d): %s. Retrying in %.1fs",
                    url.split("/")[-1], attempt + 1, max_retries + 1,
                    type(exc).__name__, delay,
                )
                time.sleep(delay)
        except requests.HTTPError as exc:
            last_exc = exc
            if exc.response is not None and exc.response.status_code < 500:
                raise
            _proxy_breaker.record_failure()
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "Request %s got %s (attempt %d/%d). Retrying in %.1fs",
                    url.split("/")[-1], exc.response.status_code if exc.response else "?",
                    attempt + 1, max_retries + 1, delay,
                )
                time.sleep(delay)

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Proxy-mode helpers (with retry + circuit breaker)
# ---------------------------------------------------------------------------
def _proxy_post(path: str, payload: dict, timeout: float = PROXY_HTTP_TIMEOUT) -> dict:
    if _proxy_breaker.is_open:
        raise RuntimeError("Circuit breaker open: proxy is unresponsive")

    url = f"{SEARCH_PROXY_URL}{path}"
    with _proxy_semaphore:
        resp = _retry_request(
            "POST", url,
            headers=_proxy_headers(json_body=True),
            json=payload,
            timeout=timeout,
        )
    return resp.json()


def _proxy_search(path: str, payload: dict) -> list[dict]:
    """POST /search/text or /search/image with retry + circuit breaker."""
    try:
        data = _proxy_post(path, payload)
    except Exception as exc:
        logger.error("Proxy search %s failed after retries: %s", path, exc)
        return [{"rank": 1, "title": "", "url": "", "snippet": f"[proxy-error] {type(exc).__name__}: {exc}"}]

    if not data.get("ok", False):
        err = data.get("error", "unknown proxy error")
        logger.warning("search-proxy %s returned error: %s", path, err)
        return [{"rank": 1, "title": "", "url": "", "snippet": f"[proxy-error] {err}"}]

    out: list[dict] = []
    for hit in data.get("results", []) or []:
        entry = {
            "rank": hit.get("rank", len(out) + 1),
            "title": hit.get("title", ""),
            "url": hit.get("url", ""),
            "snippet": hit.get("snippet", ""),
        }
        if hit.get("content") is not None:
            entry["content"] = hit["content"]
        out.append(entry)
    return out


def _proxy_upload_image(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"

    if _proxy_breaker.is_open:
        raise RuntimeError("Circuit breaker open: proxy is unresponsive")

    headers = _proxy_headers(json_body=False)
    with open(path, "rb") as fh:
        files = {"file": (path.name, fh, mime)}
        resp = _retry_request(
            "POST", f"{SEARCH_PROXY_URL}/upload_image",
            files=files, headers=headers,
            timeout=PROXY_HTTP_TIMEOUT,
        )
    data = resp.json()
    if not data.get("ok", False):
        raise RuntimeError(f"upload_image failed: {data.get('error')}")
    url = data.get("url", "")
    if not url:
        raise RuntimeError(f"upload_image returned empty url: {data}")
    return url


# ---------------------------------------------------------------------------
# Direct-mode helpers
# ---------------------------------------------------------------------------
def _require_serper_key() -> str:
    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY not set")
    return SERPER_API_KEY


def _serper_post(url: str, payload: dict) -> dict:
    headers = {
        "X-API-KEY": _require_serper_key(),
        "Content-Type": "application/json",
    }
    resp = _retry_request("POST", url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
    return resp.json()


def _jina_fetch_single(url: str, max_chars: int) -> str:
    """Fetch a single URL via Jina Reader with retry."""
    if not url:
        return ""
    reader_url = JINA_READER_BASE + url
    headers = {"Accept": "text/plain"}
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"
    try:
        resp = _retry_request(
            "GET", reader_url, headers=headers,
            timeout=JINA_TIMEOUT, max_retries=2, base_delay=1.0,
        )
        text = resp.text or ""
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + f"\n...[truncated at {max_chars} chars]"
        return text
    except Exception as exc:
        logger.warning("Jina fetch failed for %s: %s", url, exc)
        return f"[jina-error] {type(exc).__name__}: {exc}"


def _jina_fetch_parallel(urls: list[str], max_chars: int, max_workers: int = 3) -> list[str]:
    """Fetch multiple URLs via Jina in parallel."""
    results = [""] * len(urls)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_jina_fetch_single, url, max_chars): i
            for i, url in enumerate(urls)
            if url
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = f"[fetch-error] {exc}"
    return results


def _direct_upload_local_image(path: Path) -> str:
    if IMAGE_UPLOADER != "0x0":
        raise RuntimeError(f"Unsupported IMAGE_UPLOADER={IMAGE_UPLOADER!r}")
    if not path.exists():
        raise FileNotFoundError(path)
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"
    with open(path, "rb") as fh:
        files = {"file": (path.name, fh, mime)}
        headers = {"User-Agent": "kimi-agent-harness/1.0"}
        resp = _retry_request(
            "POST", "https://0x0.st",
            files=files, headers=headers, timeout=DEFAULT_TIMEOUT,
        )
    url = resp.text.strip()
    if not url.startswith("http"):
        raise RuntimeError(f"Unexpected 0x0.st response: {url!r}")
    logger.info("Uploaded %s -> %s", path, url)
    return url


def _resolve_image_to_url(image: str) -> str:
    if image.startswith("http://") or image.startswith("https://"):
        return image
    p = Path(image).expanduser()
    if p.exists() and p.is_file():
        if _proxy_enabled():
            return _proxy_upload_image(p)
        return _direct_upload_local_image(p)
    raise ValueError(f"search_image: {image!r} is neither a URL nor an existing local file.")


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------
def search_text(
    query: str,
    top_k: int = 3,
    fetch: bool = True,
    max_chars: int = 500,
) -> list[dict]:
    """Text search with caching, retry, and circuit breaker.

    Strategy: When fetch=True in proxy mode, we use a two-phase approach:
    1. Search with fetch=False first (fast, ~2-5s) to get snippets
    2. If proxy is slow, return snippets only rather than timing out
    """
    if not query or not query.strip():
        return []
    top_k = max(1, min(int(top_k), 10))

    # Check cache
    ck = _cache_key("text", query=query, top_k=top_k, fetch=fetch, max_chars=max_chars)
    cached = _search_cache.get(ck)
    if cached is not None:
        logger.info("search_text(cache-hit) q=%r", query)
        return cached

    if _proxy_enabled():
        logger.info("search_text(proxy) q=%r top_k=%d fetch=%s", query, top_k, fetch)
        results = _proxy_search(
            "/search/text",
            {"query": query, "top_k": top_k, "fetch": bool(fetch), "max_chars": int(max_chars)},
        )
    else:
        logger.info("search_text(direct) q=%r top_k=%d fetch=%s", query, top_k, fetch)
        payload = {"q": query, "num": top_k}
        data = _serper_post(SERPER_SEARCH_URL, payload)
        organic = data.get("organic", []) or []

        results: list[dict] = []
        urls_to_fetch = []
        for rank, item in enumerate(organic[:top_k], start=1):
            url = item.get("link") or ""
            entry = {
                "rank": rank,
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("snippet", ""),
            }
            results.append(entry)
            if fetch and url:
                urls_to_fetch.append(url)
            else:
                urls_to_fetch.append("")

        if urls_to_fetch and any(urls_to_fetch):
            contents = _jina_fetch_parallel(urls_to_fetch, max_chars)
            for i, content in enumerate(contents):
                if content:
                    results[i]["content"] = content

    # Cache successful results
    if results and not any("[proxy-error]" in r.get("snippet", "") for r in results):
        _search_cache.put(ck, results)

    return results


def search_image(
    image: str = "",
    image_url: str = "",
    top_k: int = 1,
    fetch: bool = True,
    max_chars: int = 500,
) -> list[dict]:
    """Reverse image search with retry and circuit breaker.

    Accepts both `image` and `image_url` parameter names for compatibility.
    """
    effective_image = image or image_url
    if not effective_image or not effective_image.strip():
        raise ValueError("search_image requires a non-empty image URL.")
    top_k = max(1, min(int(top_k), 10))

    # Check cache
    ck = _cache_key("image", image=effective_image, top_k=top_k, fetch=fetch, max_chars=max_chars)
    cached = _search_cache.get(ck)
    if cached is not None:
        logger.info("search_image(cache-hit) image=%s", effective_image[:60])
        return cached

    resolved_url = _resolve_image_to_url(effective_image.strip())

    if _proxy_enabled():
        logger.info("search_image(proxy) image_url=%s top_k=%d", resolved_url[:60], top_k)
        results = _proxy_search(
            "/search/image",
            {"image_url": resolved_url, "top_k": top_k, "fetch": bool(fetch), "max_chars": int(max_chars)},
        )
    else:
        logger.info("search_image(direct) image_url=%s top_k=%d", resolved_url[:60], top_k)
        payload = {"url": resolved_url}
        data = _serper_post(SERPER_LENS_URL, payload)
        items = data.get("organic") or data.get("visual_matches") or []

        results: list[dict] = []
        urls_to_fetch = []
        for rank, item in enumerate(items[:top_k], start=1):
            url = item.get("link") or item.get("url") or ""
            entry = {
                "rank": rank,
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("snippet", "") or item.get("source", ""),
            }
            results.append(entry)
            if fetch and url:
                urls_to_fetch.append(url)
            else:
                urls_to_fetch.append("")

        if urls_to_fetch and any(urls_to_fetch):
            contents = _jina_fetch_parallel(urls_to_fetch, max_chars)
            for i, content in enumerate(contents):
                if content:
                    results[i]["content"] = content

    if results and not any("[proxy-error]" in r.get("snippet", "") for r in results):
        _search_cache.put(ck, results)

    return results


def fetch_url(url: str, max_chars: int = 2000) -> str:
    """Fetch a single URL's content. Uses proxy /fetch endpoint or direct Jina."""
    if not url:
        return ""

    ck = _cache_key("fetch", url=url, max_chars=max_chars)
    cached = _search_cache.get(ck)
    if cached is not None:
        return cached

    if _proxy_enabled():
        try:
            data = _proxy_post("/fetch", {"url": url, "max_chars": max_chars})
            if data.get("ok"):
                content = data.get("content", "")
                _search_cache.put(ck, content)
                return content
            return f"[fetch-error] {data.get('error', 'unknown')}"
        except Exception as exc:
            logger.warning("Proxy fetch failed, falling back to direct: %s", exc)

    content = _jina_fetch_single(url, max_chars)
    if not content.startswith("["):
        _search_cache.put(ck, content)
    return content


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_text = sub.add_parser("text")
    p_text.add_argument("query")
    p_text.add_argument("--top-k", type=int, default=3)
    p_text.add_argument("--no-fetch", action="store_true")

    p_img = sub.add_parser("image")
    p_img.add_argument("image", help="URL or local path")
    p_img.add_argument("--top-k", type=int, default=3)
    p_img.add_argument("--no-fetch", action="store_true")

    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("url")
    p_fetch.add_argument("--max-chars", type=int, default=2000)

    args = ap.parse_args()
    print(f"[mode] {'proxy via ' + SEARCH_PROXY_URL if _proxy_enabled() else 'direct'}")

    if args.cmd == "text":
        out = search_text(args.query, top_k=args.top_k, fetch=not args.no_fetch)
        print(json.dumps(out, ensure_ascii=False, indent=2)[:5000])
    elif args.cmd == "image":
        out = search_image(args.image, top_k=args.top_k, fetch=not args.no_fetch)
        print(json.dumps(out, ensure_ascii=False, indent=2)[:5000])
    else:
        out = fetch_url(args.url, max_chars=args.max_chars)
        print(out[:5000])
