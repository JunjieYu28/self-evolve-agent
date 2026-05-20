"""将本地图片编码为 OpenAI 兼容的多模态 user content。"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def guess_image_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _MIME_BY_SUFFIX:
        return _MIME_BY_SUFFIX[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "image/jpeg"


def resolve_image_path(path: str | Path) -> Path | None:
    """解析相对/绝对路径；不存在则返回 None。"""
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p if p.is_file() else None


def image_to_data_url(path: str | Path) -> str:
    p = resolve_image_path(path)
    if p is None:
        raise FileNotFoundError(f"图片不存在: {path}")
    raw = p.read_bytes()
    mime = guess_image_mime(p)
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_vision_user_content(
    instruction: str,
    image_path: str | Path | None,
    *,
    image_first: bool = True,
) -> str | list[dict[str, Any]]:
    """
    构造 user message content。
    有有效图片时返回 OpenAI 多模态 parts 列表，否则返回纯文本。
    """
    p = resolve_image_path(image_path) if image_path else None
    if p is None:
        return instruction

    image_part: dict[str, Any] = {
        "type": "image_url",
        "image_url": {"url": image_to_data_url(p)},
    }
    text_part: dict[str, Any] = {"type": "text", "text": instruction}
    if image_first:
        return [image_part, text_part]
    return [text_part, image_part]


def content_to_log_string(content: str | list[dict[str, Any]] | None) -> str:
    """轨迹日志用：多模态消息不写入完整 base64。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if part.get("type") == "text":
            parts.append(part.get("text", ""))
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                parts.append("[image: inline base64]")
            else:
                parts.append(f"[image: {url[:120]}]")
    return "\n".join(parts)
