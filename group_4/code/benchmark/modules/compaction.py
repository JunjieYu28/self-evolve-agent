"""
Compaction — 上下文压缩模块
============================

当对话 token 累积超过阈值时，自动压缩历史消息：
1. 保留最近 N 条消息（不破坏 ToolUse/ToolResult 配对）
2. 将被裁剪的消息生成结构化摘要
3. 注入摘要到对话开头
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import config

logger = logging.getLogger("harness.compaction")


class Compactor:
    def __init__(
        self,
        token_threshold: int = config.COMPACTION_TOKEN_THRESHOLD,
        preserve_recent: int = config.COMPACTION_PRESERVE_RECENT,
    ):
        self.token_threshold = token_threshold
        self.preserve_recent = preserve_recent

    def estimate_tokens(self, messages: list[dict]) -> int:
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total_chars += len(part.get("text", ""))
            if msg.get("tool_calls"):
                total_chars += len(json.dumps(msg["tool_calls"], ensure_ascii=False))
        return total_chars // 3

    def should_compact(self, messages: list[dict], total_tokens: int = 0) -> bool:
        if total_tokens > self.token_threshold:
            return len(messages) > self.preserve_recent + 2
        est = self.estimate_tokens(messages)
        return est > self.token_threshold and len(messages) > self.preserve_recent + 2

    def compact(self, messages: list[dict], total_tokens: int = 0) -> list[dict]:
        if not self.should_compact(messages, total_tokens):
            return messages

        split_idx = self._find_split_point(messages)
        if split_idx <= 2:
            return messages

        # Always preserve messages[0] (system) and messages[1] (original user instruction)
        to_compress = messages[2:split_idx]
        to_keep = messages[split_idx:]

        if not to_compress:
            return messages

        summary = self._generate_summary(to_compress)

        result = [messages[0], messages[1]]
        result.append({
            "role": "assistant",
            "content": f"[Search Progress Summary]\n{summary}",
        })
        result.extend(to_keep)

        logger.info(
            "Compacted: %d messages → %d (removed %d, kept %d)",
            len(messages), len(result), len(to_compress), len(to_keep),
        )
        return result

    def _find_split_point(self, messages: list[dict]) -> int:
        target = len(messages) - self.preserve_recent
        idx = min(target, len(messages) - self.preserve_recent)

        # Never split below index 2 (preserve system + original user instruction)
        while idx > 2:
            msg = messages[idx]
            if msg.get("role") == "tool" or msg.get("tool_call_id"):
                idx -= 1
                continue
            if idx > 0 and messages[idx - 1].get("role") == "assistant" and messages[idx - 1].get("tool_calls"):
                idx -= 1
                continue
            break

        return max(idx, 2)

    def _generate_summary(self, messages: list[dict]) -> str:
        searches_performed = []
        key_findings = []
        candidate_answers = []
        constraints_info = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            if role == "assistant":
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        args = fn.get("arguments", "")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except (json.JSONDecodeError, TypeError):
                                args = {}
                        if name == "search_text":
                            q = args.get("query", "") if isinstance(args, dict) else ""
                            if q:
                                searches_performed.append(q)
                        elif name == "search_image":
                            searches_performed.append("[image search]")

                # Extract candidate answers from assistant reasoning
                bold_matches = re.findall(r'\*\*(.+?)\*\*', content)
                for m in bold_matches:
                    if len(m) < 100 and m not in candidate_answers:
                        candidate_answers.append(m)

                # Extract constraint verification
                verify_match = re.search(r'(?:verify|confirm|check|constraint)', content, re.I)
                if verify_match:
                    line = content[max(0, verify_match.start()-20):verify_match.end()+80]
                    constraints_info.append(line.strip()[:100])

            elif role == "tool" and len(content) > 30:
                if content.startswith("[ERROR") or content.startswith("[proxy-error"):
                    continue
                first_line = content.split("\n")[0][:120]
                if first_line and "Already searched" not in first_line:
                    key_findings.append(first_line)

        parts = []
        if searches_performed:
            parts.append(f"Searches done ({len(searches_performed)}): " +
                        ", ".join(f'"{q}"' for q in searches_performed[:8]))
        if key_findings:
            parts.append("Key findings:\n" + "\n".join(f"- {f}" for f in key_findings[:6]))
        if candidate_answers:
            parts.append(f"Candidate answers found: {', '.join(candidate_answers[:5])}")
        if constraints_info:
            parts.append("Constraint checks:\n" + "\n".join(f"- {c}" for c in constraints_info[:4]))

        return "\n".join(parts) if parts else "(No significant prior information)"
