"""
评测/提交用答案后处理：规则抽取 + 按长度可选 LLM 精简。

重要：本模块 **绝不使用 gold**。标答仅在外层（如 eval_benchmark）做 is_correct。

环境变量:
  EVAL_ANSWER_LLM_EXTRACT    auto|0|1  默认 auto
  EVAL_ANSWER_LLM_THRESHOLD  超过该字符数则尝试 LLM 精简，默认 20
  EVAL_EXTRACT_MAX_TOKENS    LLM 精简 max_tokens，默认 64
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_client import SGLangLLMClient

_EXTRACT_SYSTEM = (
    "You compress an agent's verbose reply into the shortest final answer for exact-match scoring.\n"
    "Read the question and the agent response.\n"
    "Output ONLY the answer itself: one name, one place, yes/no, a number, or a very short phrase.\n"
    "A single word or number is perfect when that is the answer.\n"
    "No explanation, no reasoning, no markdown, no quotes, no prefix like 'Answer:'."
)

_COMPARISON_RE = re.compile(
    r"(?:therefore|thus|so|hence|因此)[,:]?\s*\*?\*?(.+?)\*?\*?\s+"
    r"(?:was\s+)?(?:born\s+)?(earlier|later|younger|older|first)\b",
    re.IGNORECASE | re.DOTALL,
)
_ANSWER_LINE_RE = re.compile(
    r"^(?:final\s+answer|answer|答案|最终答案)\s*[:：]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_JUNK_LINE_RE = re.compile(
    r"^(reasoning\s*[:：]?|推理过程\s*[:：]?|思考\s*[:：]?|"
    r"step\s*\d+[:：.]?|\*+\s*$)",
    re.IGNORECASE,
)


def normalize_answer(text: str) -> str:
    if not text:
        return ""
    s = text.lower().strip()
    s = re.sub(r"[\s\u3000]+", " ", s)
    s = re.sub(r"[，。！？、；：""''（）\[\]【】]", "", s)
    s = re.sub(r"[^\w\s\u4e00-\u9fff]", "", s)
    return s.strip()


def is_correct(pred: str, gold: str) -> bool:
    """仅用于事后评测，不参与答案抽取。"""
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not g:
        return False
    if p == g:
        return True
    return g in p or p in g


def _strip_md(pred: str) -> str:
    pred = re.sub(r"\*\*([^*]+)\*\*", r"\1", pred)
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", pred)


def _extract_comparison_entity(pred: str) -> str | None:
    m = _COMPARISON_RE.search(pred)
    if m:
        name = m.group(1).strip().strip(".,; ")
        if name and len(name) < 120:
            return name
    return None


def extract_short_answer(pred: str) -> str:
    """规则抽取短答案（无 LLM、无 gold）。"""
    pred = pred.strip()
    if not pred:
        return pred
    if pred.startswith("[") and "Error" in pred[:40]:
        return pred

    pred = _strip_md(pred)

    m = _ANSWER_LINE_RE.search(pred)
    if m:
        return m.group(1).strip()[:120]

    comp = _extract_comparison_entity(pred)
    if comp:
        return comp[:120]

    lines = [ln.strip() for ln in pred.splitlines() if ln.strip()]
    candidates: list[str] = []
    for line in lines:
        if line.startswith("[") and line.endswith("]"):
            continue
        if _JUNK_LINE_RE.match(line):
            continue
        if re.match(r"^[\*\-\d]+\.?\s", line):
            continue
        line = re.sub(
            r"^(答案[:：]|答[:：]|最终答案[:：]|A[:：]\s*)", "", line
        ).strip()
        line = line.strip("。．. ")
        if line:
            candidates.append(line)

    if not candidates:
        return pred[:120]

    def score(s: str) -> tuple[int, int]:
        explain = 1 if any(
            w in s
            for w in (
                "根据",
                "图片",
                "because",
                "while ",
                "这是",
                "位于",
                "was born",
                "推理",
                "reasoning",
            )
        ) else 0
        return (explain, len(s))

    best = min(candidates, key=score)

    if len(best) > 80:
        if re.search(r"\b(?:therefore|因此)\b", best, re.IGNORECASE):
            tail = re.split(r"\b(?:[Tt]herefore|因此),?\s*", best)[-1].strip()
            if tail:
                best = tail
        else:
            parts = re.split(r"[，,；;]", best)
            if parts and len(parts[0]) >= 2:
                best = parts[0].strip()

    return best[:120]


def _llm_extract_answer(
    question: str,
    raw_pred: str,
    client: SGLangLLMClient,
) -> str:
    user = (
        f"Question:\n{question[:800]}\n\n"
        f"Agent response:\n{raw_pred[:2000]}\n\n"
        "Shortest final answer only:"
    )
    max_tokens = int(os.getenv("EVAL_EXTRACT_MAX_TOKENS", "64"))
    try:
        resp = client.chat_completion(
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        text = (resp.get("content") or "").strip()
        if not text or (text.startswith("[") and "Error" in text[:30]):
            return ""
        line = text.split("\n")[0].strip()
        line = re.sub(r"^(答案[:：]|答[:：]|final answer[:：])\s*", "", line, flags=re.I)
        return line.strip(" \"'")[:120]
    except Exception:
        return ""


def _llm_extract_enabled() -> bool:
    v = os.getenv("EVAL_ANSWER_LLM_EXTRACT", "auto").strip().lower()
    return v in ("1", "true", "yes", "auto")


def _llm_threshold() -> int:
    return int(os.getenv("EVAL_ANSWER_LLM_THRESHOLD", "20"))


def finalize_answer(
    raw_pred: str,
    *,
    question: str = "",
    llm_client: SGLangLLMClient | None = None,
) -> tuple[str, str]:
    """
    将 Agent 长回答变为提交用短答案。**不使用 gold。**

    返回 (pred, method):
      - rule_short: 规则结果长度 <= 阈值，未调 LLM
      - llm:        超过阈值后由 LLM 精简
      - rule:       超过阈值但 LLM 未启用/失败，退回规则结果
      - empty:      空输入
    """
    raw = raw_pred.strip()
    if not raw:
        return raw, "empty"

    ruled = extract_short_answer(raw)
    threshold = _llm_threshold()

    # 已超过阈值 → 尝试 LLM（raw 很长但 ruled 很短时也触发，避免只抽到 Reasoning:）
    need_llm = _llm_extract_enabled() and (
        len(ruled) > threshold or len(raw) > threshold
    )

    if not need_llm:
        return (ruled or raw).strip()[:120], "rule_short"

    if llm_client is not None:
        llm_out = _llm_extract_answer(question, raw, llm_client)
        if llm_out:
            return llm_out, "llm"

    return (ruled or raw[:120]).strip(), "rule"


def finalize_answer_for_scoring(
    raw_pred: str,
    *,
    gold: str = "",
    question: str = "",
    llm_client: SGLangLLMClient | None = None,
) -> tuple[str, str]:
    """
    兼容旧调用名；**gold 参数已忽略**，仅使用 finalize_answer。
    请在调用方用 is_correct(pred, gold) 单独评测。
    """
    del gold  # 明确表示不使用标答
    return finalize_answer(
        raw_pred, question=question, llm_client=llm_client
    )
