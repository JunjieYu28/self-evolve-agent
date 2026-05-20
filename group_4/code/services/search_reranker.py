"""Cross-Encoder 重排：对 search-proxy 召回结果做本地精排（BGE Reranker）。"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 根分区常满：默认把 HF 缓存放到 /data（与 scripts/download_gaia.* 一致）
_DEFAULT_HF_HOME = Path(__file__).resolve().parent.parent / "huggingface_cache"


def _configure_hf_cache() -> Path:
    """在 import transformers 之前设置 HF_HOME / HF_HUB_CACHE（可用环境变量覆盖）。"""
    hf_home = Path(
        os.environ.get("HF_HOME", "").strip() or str(_DEFAULT_HF_HOME)
    ).expanduser()
    hub_cache = Path(
        os.environ.get("HF_HUB_CACHE", "").strip() or str(hf_home / "hub")
    ).expanduser()
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hub_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
    hub_cache.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    return hf_home


_HF_HOME = _configure_hf_cache()

_DEFAULT_MODEL = os.getenv(
    "SEARCH_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
).strip()
_ENABLED = os.getenv("SEARCH_RERANKER_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
_MAX_LENGTH = int(os.getenv("SEARCH_RERANKER_MAX_LENGTH", "512"))
_BATCH_SIZE = max(1, int(os.getenv("SEARCH_RERANKER_BATCH_SIZE", "16")))


class SearchResultReranker:
    """懒加载 BGE Cross-Encoder；加载失败时 is_available() 为 False。"""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = (model_name or _DEFAULT_MODEL).strip()
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: str = "cpu"
        self._load_error: str | None = None
        self._init_lock = threading.Lock()

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None and self._tokenizer is not None

    @property
    def load_error(self) -> str | None:
        self._ensure_loaded()
        return self._load_error

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        with self._init_lock:
            if self._model is not None or self._load_error is not None:
                return
            if not _ENABLED:
                self._load_error = "disabled by SEARCH_RERANKER_ENABLED"
                return
            try:
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                logger.info(
                    "Loading search reranker: %s (HF_HOME=%s)",
                    self._model_name,
                    _HF_HOME,
                )
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self._model_name
                )
                self._model.eval()
                if torch.cuda.is_available():
                    want = os.getenv("SEARCH_RERANKER_DEVICE", "cuda").lower()
                    self._device = "cuda" if want != "cpu" else "cpu"
                else:
                    self._device = "cpu"
                self._model.to(self._device)
                logger.info("Search reranker ready on %s", self._device)
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"{type(exc).__name__}: {exc}"
                self._tokenizer = None
                self._model = None
                logger.warning(
                    "Search reranker unavailable (%s); fallback to proxy order",
                    self._load_error,
                )

    @staticmethod
    def _passage_text(hit: dict[str, Any]) -> str:
        title = str(hit.get("title") or "").strip()
        snippet = str(hit.get("snippet") or "").strip()
        content = str(hit.get("content") or "").strip()
        body = content or snippet or title
        if title and body and title not in body:
            return f"{title}\n{body}"
        return body or title

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not results or top_k <= 0:
            return []
        if not self.available:
            return results[:top_k]

        query = query.strip()
        if not query:
            return results[:top_k]

        try:
            import torch

            pairs = [[query, self._passage_text(hit)] for hit in results]
            scores: list[float] = []
            assert self._tokenizer is not None and self._model is not None

            with torch.no_grad():
                for start in range(0, len(pairs), _BATCH_SIZE):
                    batch = pairs[start : start + _BATCH_SIZE]
                    inputs = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=_MAX_LENGTH,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                    logits = self._model(**inputs, return_dict=True).logits.view(-1)
                    scores.extend(logits.float().cpu().tolist())

            ranked = sorted(
                zip(scores, results),
                key=lambda x: x[0],
                reverse=True,
            )
            out: list[dict[str, Any]] = []
            for i, (_score, hit) in enumerate(ranked[:top_k], start=1):
                item = dict(hit)
                item["rank"] = i
                item["rerank_score"] = round(float(_score), 4)
                out.append(item)
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("Rerank failed (%s); using proxy order", exc)
            return results[:top_k]


_reranker: SearchResultReranker | None = None
_reranker_lock = threading.Lock()


def get_search_reranker() -> SearchResultReranker:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = SearchResultReranker()
    return _reranker


def rerank_search_results(
    query: str,
    results: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    return get_search_reranker().rerank(query, results, top_k)
