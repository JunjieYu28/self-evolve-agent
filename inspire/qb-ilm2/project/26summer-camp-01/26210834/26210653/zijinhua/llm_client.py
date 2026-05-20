"""LLM 客户端：PAI-EAS Qwen 与本地 SGLang OpenAI 兼容双后端。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol

from config import EASConfig, LLMConfig, ReflectionLLMConfig, SGLangConfig


class ChatLLMClient(Protocol):
    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        ...


def _extract_visible_answer(text: str) -> str:
    """从 Qwen thinking 轨迹中抽取可展示的简短答案。"""
    if not text or not text.strip():
        return text
    for marker in (
        "最终答案：",
        "最终答案:",
        "Final Answer:",
        "Final answer:",
        "答案是",
        "Answer:",
    ):
        if marker in text:
            tail = text.split(marker, 1)[-1].strip()
            if tail:
                return tail.split("\n")[0].strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2 and lines[0].lower().startswith("thinking"):
        return lines[-1]
    return text.strip()


def _assistant_from_openai_message(message: Any) -> dict[str, Any]:
    """将 OpenAI SDK message 或 dict 转为 agent 使用的 assistant dict。"""
    if hasattr(message, "model_dump"):
        msg = message.model_dump()
    elif isinstance(message, dict):
        msg = message
    else:
        msg = {
            "content": getattr(message, "content", None),
            "reasoning_content": getattr(message, "reasoning_content", None),
            "tool_calls": getattr(message, "tool_calls", None),
        }

    content = msg.get("content")
    if not content:
        # vLLM Qwen3.5 可能把输出放在 reasoning / reasoning_content
        for key in ("reasoning_content", "reasoning"):
            if msg.get(key):
                content = msg[key]
                break
    if content:
        content = _extract_visible_answer(content)

    result: dict[str, Any] = {"role": "assistant", "content": content}

    if tool_calls := msg.get("tool_calls"):
        parsed_calls: list[dict[str, Any]] = []
        for i, tc in enumerate(tool_calls):
            if hasattr(tc, "model_dump"):
                tc = tc.model_dump()
            fn = tc.get("function") or {}
            if hasattr(fn, "model_dump"):
                fn = fn.model_dump()
            parsed_calls.append(
                {
                    "id": tc.get("id", f"call_{i}"),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": fn["name"],
                        "arguments": fn["arguments"],
                    },
                }
            )
        result["tool_calls"] = parsed_calls
    return result


class EASLLMClient:
    """PAI-EAS Qwen：urllib 直连 /v1/chat/completions。"""

    def __init__(self, config: EASConfig) -> None:
        self.config = config
        self.base_url = config.api_url.rstrip("/")
        self.chat_url = f"{self.base_url}/v1/chat/completions"

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if not self.config.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        raw = self._post_json(self.chat_url, payload)
        try:
            message = raw["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"无法解析 EAS 响应: {raw}") from exc
        return _assistant_from_openai_message(message)

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.config.api_token,
        }
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(
                req, timeout=self.config.timeout
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"EAS API 请求失败 HTTP {exc.code}: {detail}"
            ) from exc


class SGLangLLMClient:
    """
    本地 SGLang OpenAI 兼容客户端（移植自 harness task_runner）。
    关键调用：OpenAI(base_url=LLM_BASE_URL).chat.completions.create(
        model, messages, tools, tool_choice='auto',
        extra_body={'enable_thinking': True},
    )
    """

    def __init__(self, config: SGLangConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "SGLang 后端需要 openai 包：pip install openai"
            ) from exc

        self.config = config
        self._client = OpenAI(
            base_url=config.base_url.rstrip("/"),
            api_key=config.api_key,
            timeout=config.timeout,
        )

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        extra_body: dict[str, Any] = {
            "enable_thinking": self.config.enable_thinking,
        }
        if not self.config.enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        request_kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "extra_body": extra_body,
        }
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message
        return _assistant_from_openai_message(message)


def create_llm_client(config: LLMConfig | None = None) -> ChatLLMClient:
    cfg = config or LLMConfig.from_env()
    if cfg.backend == "sglang":
        if cfg.sglang is None:
            raise ValueError("SGLang 配置缺失")
        return SGLangLLMClient(cfg.sglang)
    if cfg.eas is None:
        raise ValueError("EAS 配置缺失")
    return EASLLMClient(cfg.eas)


def create_reflection_llm_client(
    config: ReflectionLLMConfig | LLMConfig | None = None,
) -> ChatLLMClient:
    """反思 / 记忆辅助模型（默认本机专用 9B @ :8004）。"""
    if isinstance(config, ReflectionLLMConfig):
        ref_cfg = config
    elif config is not None and config.reflection is not None:
        ref_cfg = config.reflection
    else:
        ref_cfg = ReflectionLLMConfig.from_env()
    return SGLangLLMClient(
        SGLangConfig(
            base_url=ref_cfg.base_url,
            api_key=ref_cfg.api_key,
            model_name=ref_cfg.model_name,
            temperature=ref_cfg.temperature,
            max_tokens=ref_cfg.max_tokens,
            timeout=ref_cfg.timeout,
            enable_thinking=ref_cfg.enable_thinking,
        )
    )
