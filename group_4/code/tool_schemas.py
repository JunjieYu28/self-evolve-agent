"""官方 harness 工具 Schema（与 task_runner.TOOLS_SCHEMA 对齐）。"""

from __future__ import annotations

from typing import Any

# 与 external/harness-sii/task_runner.py TOOLS_SCHEMA 保持一致
HARNESS_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": (
                "基于 Serper (Google) 的联网文字搜索，并用 Jina Reader 抽取每个结果页面的正文"
                "返回 [{rank,title,url,snippet,content}]。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "top_k": {"type": "integer", "description": "返回条数（1-3）", "default": 1},
                    "fetch": {"type": "boolean", "description": "是否抓取正文", "default": True},
                    "max_chars": {"type": "integer", "description": "正文截断字符数", "default": 500},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_image",
            "description": (
                "图搜文：Google Lens 反向图像搜索 + Jina 抽正文。"
                "image 可为 http(s) URL 或本地路径。返回 [{rank,title,url,snippet,content}]。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {
                        "type": "string",
                        "description": "图片 http(s) URL 或本地路径",
                    },
                    "top_k": {"type": "integer", "default": 1},
                    "fetch": {"type": "boolean", "default": True},
                    "max_chars": {"type": "integer", "default": 500},
                },
                "required": ["image_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "沙盒浏览器打开 URL，返回 {ok,url,title,text_preview?,truncated?}。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "wait_until": {
                        "type": "string",
                        "enum": ["domcontentloaded", "load", "networkidle"],
                        "default": "domcontentloaded",
                    },
                    "include_text": {"type": "boolean", "default": True},
                    "max_text": {"type": "integer", "default": 2000},
                    "timeout": {"type": "integer", "default": 30},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_text",
            "description": "返回当前页可见文本 {ok,url,title,text,truncated,total_chars}。",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_chars": {"type": "integer", "default": 5000},
                    "timeout": {"type": "integer", "default": 15},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "CSS 选择器点击元素，返回 {ok,selector,current_url,current_title,navigated}。",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "nth": {"type": "integer", "default": 0},
                    "timeout": {"type": "integer", "default": 10},
                },
                "required": ["selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": "向输入框键入文本，返回 {ok,selector,submitted,current_url,current_title}。",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                    "submit": {"type": "boolean", "default": False},
                    "clear": {"type": "boolean", "default": True},
                    "timeout": {"type": "integer", "default": 10},
                },
                "required": ["selector", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_parallel",
            "description": "并发打开多个 URL，返回 list[dict]。",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {"type": "array", "items": {"type": "string"}},
                    "mode": {"type": "string", "enum": ["navigate", "get_text"], "default": "navigate"},
                    "max_chars": {"type": "integer"},
                    "wait_until": {
                        "type": "string",
                        "enum": ["domcontentloaded", "load", "networkidle"],
                        "default": "domcontentloaded",
                    },
                    "max_concurrency": {"type": "integer", "default": 4},
                    "timeout": {"type": "integer", "default": 30},
                },
                "required": ["urls"],
            },
        },
    },
]
