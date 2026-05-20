"""浏览器工具实现（移植自 harness-sii/tools/browser_tool.py）。"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional
from urllib.parse import urlparse

from services.browser_client import get_browser_client

logger = logging.getLogger(__name__)

_RE_BLANK = re.compile(r"\n{3,}")
_RE_WS = re.compile(r"[ \t]{2,}")


def _clean(s: str) -> str:
    if not s:
        return ""
    s = _RE_BLANK.sub("\n\n", s)
    s = _RE_WS.sub(" ", s)
    return s.strip()


def _truncate(s: str, n: int) -> tuple[str, bool]:
    if n and len(s) > n:
        return s[:n], True
    return s, False


def _norm_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    # data:/about:/file: 等单冒号 scheme 也需保留，勿加 https://
    if urlparse(url).scheme:
        return url
    return "https://" + url


def _err(msg: str, **kw: Any) -> dict:
    d = {"ok": False, "error": msg}
    d.update(kw)
    return d


def _network_err(exc: Exception) -> dict:
    return _err(
        f"网络异常或超时，无法访问该网页({exc})。"
        "请尝试提取页面核心词，改用 search_text 工具重新搜索。"
    )


def browser_navigate(
    url: str,
    wait_until: str = "domcontentloaded",
    include_text: bool = True,
    max_text: int = 2000,
    timeout: int = 30,
) -> dict:
    if not url or not url.strip():
        return _err("url is empty")
    real = _norm_url(url)
    wu = wait_until if wait_until in {"load", "domcontentloaded", "networkidle"} else "domcontentloaded"
    try:
        cli = get_browser_client()
        nav = cli.navigate(real, wait_until=wu, timeout_ms=max(1, timeout) * 1000)
    except Exception as exc:
        return _network_err(exc)

    out: dict = {
        "ok": True,
        "url": nav.get("url", real),
        "title": nav.get("title", ""),
        "wait_until": wu,
    }
    if include_text:
        try:
            text = _clean(cli.get_text())
            txt, truncated = _truncate(text, int(max_text))
            out["text_preview"] = txt
            out["truncated"] = truncated
        except Exception as exc:
            out["text_preview"] = ""
            out["text_error"] = str(exc)
    return out


def browser_get_text(max_chars: int = 5000, timeout: int = 15) -> dict:
    _ = timeout
    try:
        cli = get_browser_client()
        text = _clean(cli.get_text())
        meta = cli.title()
    except Exception as exc:
        return _network_err(exc)
    txt, truncated = _truncate(text, int(max_chars))
    return {
        "ok": True,
        "url": meta.get("url", ""),
        "title": meta.get("title", ""),
        "text": txt,
        "truncated": truncated,
        "total_chars": len(text),
    }


def browser_click(selector: str, nth: int = 0, timeout: int = 10) -> dict:
    if not selector or not selector.strip():
        return _err("selector is empty")
    try:
        cli = get_browser_client()
        before = cli.title()
        if int(nth) > 0:
            ok = cli.eval_js(
                f"(() => {{ const els = document.querySelectorAll({selector!r});"
                f" if (!els || els.length <= {int(nth)}) return false;"
                f" els[{int(nth)}].click(); return true; }})()"
            )
            if not ok:
                return _err(f"selector matched <= {nth} elements")
        else:
            cli.click(selector, timeout_ms=max(1, timeout) * 1000)
        after = cli.title()
    except Exception as exc:
        return _err(f"click failed: {exc}")
    return {
        "ok": True,
        "selector": selector,
        "current_url": after.get("url", ""),
        "current_title": after.get("title", ""),
        "navigated": bool(after.get("url") and after.get("url") != before.get("url")),
    }


def browser_type(
    selector: str,
    text: str,
    submit: bool = False,
    clear: bool = True,
    timeout: int = 10,
) -> dict:
    _ = timeout
    if not selector or not selector.strip():
        return _err("selector is empty")
    try:
        cli = get_browser_client()
        cli.type_text(selector, str(text), clear=clear, press_enter=submit)
        meta = cli.title()
    except Exception as exc:
        return _err(f"type failed: {exc}")
    return {
        "ok": True,
        "selector": selector,
        "submitted": bool(submit),
        "current_url": meta.get("url", ""),
        "current_title": meta.get("title", ""),
    }


def _parallel_one(
    cli: Any,
    url: str,
    mode: str,
    max_chars: int,
    wait_until: str,
    timeout_ms: int,
) -> dict:
    tab_id = ""
    try:
        new = cli.new_tab()
        tab_id = new.get("tab_id", "")
        cli.navigate(url, wait_until=wait_until, timeout_ms=timeout_ms, tab_id=tab_id)
        meta = cli.title(tab_id=tab_id)
        text = _clean(cli.get_text(tab_id=tab_id) if tab_id else cli.get_text())
        if mode == "navigate":
            txt, truncated = _truncate(text, max_chars)
            return {
                "ok": True,
                "url": meta.get("url", url),
                "title": meta.get("title", ""),
                "text_preview": txt,
                "truncated": truncated,
            }
        txt, truncated = _truncate(text, max_chars)
        return {
            "ok": True,
            "url": meta.get("url", url),
            "title": meta.get("title", ""),
            "text": txt,
            "truncated": truncated,
            "total_chars": len(text),
        }
    except Exception as exc:
        err = _network_err(exc)
        err["url"] = url
        return err
    finally:
        if tab_id:
            try:
                cli.close_tab(tab_id)
            except Exception:
                pass


def browser_parallel(
    urls: list,
    mode: str = "navigate",
    max_chars: Optional[int] = None,
    wait_until: str = "domcontentloaded",
    max_concurrency: int = 4,
    timeout: int = 30,
) -> list:
    if not urls:
        return []
    mode = mode if mode in ("navigate", "get_text") else "navigate"
    cap = int(max_chars or (2000 if mode == "navigate" else 5000))
    wu = wait_until if wait_until in {"load", "domcontentloaded", "networkidle"} else "domcontentloaded"
    timeout_ms = max(1, int(timeout)) * 1000
    workers = max(1, min(int(max_concurrency), 8))
    cli = get_browser_client()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_parallel_one, cli, _norm_url(u), mode, cap, wu, timeout_ms): u
            for u in urls
        }
        for fut in as_completed(futs):
            results.append(fut.result())
    return results
