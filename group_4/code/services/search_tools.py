"""搜索工具实现（search-proxy @ 8090，移植自 harness-sii）。"""

from __future__ import annotations

import logging
import os
from typing import Any

from services import search_client as sc
from services.search_reranker import get_search_reranker

logger = logging.getLogger(__name__)

_DEFAULT_FETCH = os.getenv("SEARCH_FETCH_DEFAULT", "true").lower() in (
    "1",
    "true",
    "yes",
)
# search-proxy 广度召回条数（Serper），本地 Cross-Encoder 再精排为 top_k
_DEFAULT_FETCH_K = int(os.getenv("SEARCH_FETCH_K", "20"))
_PROXY_RECALL_MAX = int(os.getenv("SEARCH_PROXY_RECALL_MAX", "20"))


def _clamp_recall_k(fetch_k: int | None) -> int:
    k = _DEFAULT_FETCH_K if fetch_k is None else int(fetch_k)
    return max(1, min(k, _PROXY_RECALL_MAX))


def _attach_fetch_to_hits(
    hits: list[dict[str, Any]],
    *,
    max_chars: int,
) -> list[dict[str, Any]]:
    for hit in hits:
        if hit.get("content"):
            continue
        url = str(hit.get("url") or "").strip()
        if not url:
            continue
        content = sc.fetch_page_content(url, max_chars=max_chars)
        if content:
            hit["content"] = content
    return hits


def _finalize_ranks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, hit in enumerate(results, start=1):
        item = dict(hit)
        item["rank"] = i
        out.append(item)
    return out


def search_text(
    query: str,
    top_k: int = 3,
    fetch: bool | None = None,
    max_chars: int = 500,
    fetch_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    文搜：先向 search-proxy 召回 fetch_k 条（默认 20），再用 BGE Reranker 精排为 top_k。
    Reranker 不可用时降级为 proxy 原始顺序的前 top_k 条。
    """
    if fetch is None:
        fetch = _DEFAULT_FETCH
    if not query or not query.strip():
        return []

    query = query.strip()
    top_k = max(1, min(int(top_k), 10))
    recall_k = max(_clamp_recall_k(fetch_k), top_k)

    logger.info(
        "search_text q=%r recall_k=%d top_k=%d fetch=%s",
        query,
        recall_k,
        top_k,
        fetch,
    )

    # 广度召回：fetch 在 rerank 前建议为 false，避免对 20 条结果逐条 Jina 抓全文
    recall_fetch = fetch if recall_k <= top_k else False
    if recall_k > top_k and fetch:
        logger.debug(
            "search_text: fetch disabled for recall_k=%d (will rerank then trim)",
            recall_k,
        )

    try:
        candidates = sc.proxy_search(
            "/search/text",
            {
                "query": query,
                "top_k": recall_k,
                "fetch": bool(recall_fetch),
                "max_chars": int(max_chars),
            },
        )
    except Exception as exc:
        logger.warning("search_text proxy failed: %s", exc)
        return []

    if not candidates:
        return []

    reranker = get_search_reranker()
    if reranker.available:
        ranked = reranker.rerank(query, candidates, top_k)
        logger.info(
            "search_text reranked %d -> %d (model=%s)",
            len(candidates),
            len(ranked),
            os.getenv("SEARCH_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        )
        if fetch and not recall_fetch:
            ranked = _attach_fetch_to_hits(ranked, max_chars=max_chars)
        return _finalize_ranks(ranked)

    if reranker.load_error:
        logger.info(
            "search_text reranker skipped (%s); using proxy top-%d",
            reranker.load_error,
            top_k,
        )
    trimmed = candidates[:top_k]
    if fetch and not recall_fetch:
        trimmed = _attach_fetch_to_hits(trimmed, max_chars=max_chars)
    return _finalize_ranks(trimmed)


def search_image(
    image: str,
    top_k: int = 1,
    fetch: bool | None = None,
    max_chars: int = 500,
) -> list[dict[str, Any]]:
    if fetch is None:
        fetch = _DEFAULT_FETCH
    if not image or not image.strip():
        raise ValueError("search_image requires non-empty image")
    top_k = max(1, min(int(top_k), 10))
    image_url = sc.resolve_image(image.strip())
    logger.info("search_image url=%s top_k=%d", image_url, top_k)
    return sc.proxy_search(
        "/search/image",
        {
            "image_url": image_url,
            "top_k": top_k,
            "fetch": bool(fetch),
            "max_chars": int(max_chars),
        },
    )
