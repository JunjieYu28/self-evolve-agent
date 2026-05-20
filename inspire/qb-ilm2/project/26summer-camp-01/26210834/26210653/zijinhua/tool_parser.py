"""从模型文本回复中解析 Hermes/Qwen 风格的 <tool_call> 块。"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_tool_calls_from_content(content: str | None) -> list[dict[str, Any]] | None:
    if not content or "<tool_call>" not in content.lower():
        return None

    calls: list[dict[str, Any]] = []
    blocks = re.findall(
        r"<tool_call>(.*?)</tool_call>",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for i, block in enumerate(blocks):
        fn_match = re.search(r"<function=(\w+)>", block, flags=re.IGNORECASE)
        if not fn_match:
            continue
        args: dict[str, str] = {}
        for pm in re.finditer(
            r"<parameter=(\w+)>\s*(.*?)\s*</parameter>",
            block,
            flags=re.DOTALL | re.IGNORECASE,
        ):
            args[pm.group(1)] = pm.group(2).strip()

        calls.append(
            {
                "id": f"call_parsed_{i}",
                "type": "function",
                "function": {
                    "name": fn_match.group(1),
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )
    return calls or None


def normalize_llm_response(response: dict[str, Any]) -> dict[str, Any]:
    """将文本形式的 tool_call 转为标准 tool_calls 字段。"""
    if response.get("tool_calls"):
        return response

    parsed = extract_tool_calls_from_content(response.get("content"))
    if not parsed:
        return response

    normalized = dict(response)
    normalized["tool_calls"] = parsed
    normalized["content"] = None
    return normalized
