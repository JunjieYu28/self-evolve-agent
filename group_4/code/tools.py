"""工具抽象与官方 harness 真实工具注册。"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from tool_schemas import HARNESS_TOOL_SCHEMAS


class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        ...

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        ...


class HarnessFunctionTool(BaseTool):
    """将 harness 函数包装为 BaseTool，返回 JSON 字符串供 LLM 阅读。"""

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any],
        schema: dict[str, Any],
    ) -> None:
        self._name = name
        self._fn = fn
        self._schema = schema

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._schema["function"]["description"]

    @property
    def parameters(self) -> dict[str, Any]:
        return self._schema["function"]["parameters"]

    def execute(self, **kwargs: Any) -> str:
        # 官方 schema 为 image_url，实现函数参数为 image
        if self._name == "search_image":
            if "image" not in kwargs and kwargs.get("image_url"):
                kwargs["image"] = kwargs.pop("image_url")
        try:
            raw = self._fn(**kwargs)
            if isinstance(raw, (dict, list)):
                return json.dumps(raw, ensure_ascii=False)
            return str(raw)
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]


class MockSearchTool(BaseTool):
    """联调用 Mock 搜索（integration_test）。"""

    @property
    def name(self) -> str:
        return "mock_search"

    @property
    def description(self) -> str:
        return "Mock search for offline integration tests."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> str:
        q = kwargs.get("query", "")
        return f"[MockSearch] results for: {q}"


def create_harness_registry(include_mock: bool = False) -> ToolRegistry:
    """注册官方真实工具：search-proxy:8090 + browser-service:8080。"""
    from services import browser_tools, search_tools

    schema_by_name = {s["function"]["name"]: s for s in HARNESS_TOOL_SCHEMAS}
    fn_map: dict[str, Callable[..., Any]] = {
        "search_text": search_tools.search_text,
        "search_image": search_tools.search_image,
        "browser_navigate": browser_tools.browser_navigate,
        "browser_get_text": browser_tools.browser_get_text,
        "browser_click": browser_tools.browser_click,
        "browser_type": browser_tools.browser_type,
        "browser_parallel": browser_tools.browser_parallel,
    }

    registry = ToolRegistry()
    for name, fn in fn_map.items():
        registry.register(
            HarnessFunctionTool(name, fn, schema_by_name[name])
        )
    if include_mock:
        registry.register(MockSearchTool())
    return registry


def create_default_registry() -> ToolRegistry:
    return create_harness_registry(include_mock=True)


def create_search_only_registry(include_mock: bool = False) -> ToolRegistry:
    """仅搜索工具（无 browser-service，适合无 sudo 装 Chromium 依赖的环境）。"""
    from services import search_tools

    schema_by_name = {s["function"]["name"]: s for s in HARNESS_TOOL_SCHEMAS}
    search_names = {"search_text", "search_image"}
    fn_map: dict[str, Callable[..., Any]] = {
        "search_text": search_tools.search_text,
        "search_image": search_tools.search_image,
    }

    registry = ToolRegistry()
    for name, fn in fn_map.items():
        registry.register(
            HarnessFunctionTool(name, fn, schema_by_name[name])
        )
    if include_mock:
        registry.register(MockSearchTool())
    return registry


def create_production_registry(include_mock: bool = False) -> ToolRegistry:
    if os.getenv("DISABLE_BROWSER", "0").lower() in ("1", "true", "yes"):
        return create_search_only_registry(include_mock=include_mock)
    return create_harness_registry(include_mock=include_mock)


def create_empty_registry() -> ToolRegistry:
    """无外部工具（不调用 Serper / browser），用于纯模型能力评测。"""
    return ToolRegistry()
