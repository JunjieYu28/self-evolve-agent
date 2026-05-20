"""
TeacherClient — 32B 模型 API 客户端
====================================

封装 OpenAI SDK 调用，指向 32B vLLM 服务。
提供统一的 complete() 接口供 planner/verifier/reflector 使用。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

import config

logger = logging.getLogger("harness.teacher")


@dataclass
class TeacherResponse:
    content: str
    usage_tokens: int = 0
    success: bool = True
    error: Optional[str] = None


class TeacherClient:
    def __init__(
        self,
        base_url: str = config.TEACHER_BASE_URL,
        model: str = config.TEACHER_MODEL,
        max_tokens: int = config.TEACHER_MAX_TOKENS,
        timeout: int = config.TEACHER_TIMEOUT,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",
            timeout=timeout,
        )
        self.model = model
        self.max_tokens = max_tokens
        self._total_calls = 0
        self._total_tokens = 0

    def complete(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        response_format: Optional[dict] = None,
    ) -> TeacherResponse:
        """Send a completion request to 32B model."""
        self._total_calls += 1
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature,
                "extra_body": {"enable_thinking": True},
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, 'reasoning_content', '') or ""
            tokens = response.usage.total_tokens if response.usage else 0
            self._total_tokens += tokens

            # 如果 content 为空但 reasoning_content 有内容，从 reasoning 中提取
            if not content.strip() and reasoning:
                content = self._extract_from_reasoning(reasoning)

            return TeacherResponse(content=content, usage_tokens=tokens)

        except Exception as exc:
            logger.error("32B call failed: %s", exc)
            return TeacherResponse(
                content="", success=False, error=str(exc)
            )

    def _extract_from_reasoning(self, reasoning: str) -> str:
        """当 content 为空时，从 reasoning_content 中提取有效内容（JSON 或答案）。"""
        import re
        # 尝试找 JSON 块
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', reasoning, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # 找 "answer:" 或 "Answer:" 后面的内容
        answer_match = re.search(
            r'(?:final\s*answer|answer)\s*[:：]\s*(.+?)(?:\n|$)',
            reasoning, re.IGNORECASE
        )
        if answer_match:
            return answer_match.group(1).strip()

        # 最后手段：取 reasoning 最后一段非空行
        lines = [l.strip() for l in reasoning.strip().split('\n') if l.strip()]
        if lines:
            return lines[-1]
        return ""

    def parse_json_response(self, resp: TeacherResponse) -> Optional[dict]:
        """Extract JSON from 32B response (handles markdown code blocks)."""
        if not resp.success or not resp.content:
            return None

        text = resp.content.strip()

        # Remove markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # remove ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # Remove <think>...</think> blocks
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.warning("Failed to parse 32B JSON response: %s...", text[:200])
            return None

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
        }
