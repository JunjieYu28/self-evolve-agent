"""反思模块：任务失败后的轨迹诊断与修正策略提取。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class FailureReason(str, Enum):
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    CONSECUTIVE_TOOL_ERRORS = "consecutive_tool_errors"
    WRONG_ANSWER = "wrong_answer"


@dataclass
class ReflectionState:
    """单次任务运行期的反思触发状态。"""

    consecutive_tool_errors: int = 0
    tool_error_threshold: int = 3
    max_steps: int = 10
    steps_taken: int = 0
    got_final_answer: bool = False
    instruction: str = ""
    image_path: str | None = None

    def record_tool_result(self, result: str) -> None:
        s = result.strip()
        if s.lower().startswith("error"):
            self.consecutive_tool_errors += 1
            return
        if s.startswith("{"):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and obj.get("ok") is False:
                    self.consecutive_tool_errors += 1
                    return
            except json.JSONDecodeError:
                pass
        self.consecutive_tool_errors = 0

    def should_trigger(self) -> FailureReason | None:
        if self.got_final_answer:
            return None
        if self.consecutive_tool_errors >= self.tool_error_threshold:
            return FailureReason.CONSECUTIVE_TOOL_ERRORS
        if self.steps_taken >= self.max_steps and not self.got_final_answer:
            return FailureReason.MAX_STEPS_EXCEEDED
        return None


REFLECTION_SYSTEM_PROMPT = """你是一位资深的 AI Agent 诊断专家，专门分析 ReAct 智能体在执行任务失败时的行为轨迹。

你的任务是：
1. 仔细阅读完整的执行轨迹（trajectory），包括用户的初始指令、模型的推理、工具调用及其返回。
2. 精准定位失败根因：
   - 在哪一步调错了工具？（工具名称不匹配、参数错误、重复无效调用）
   - 为什么陷入了死循环？（重复搜索同一 query、忽略工具返回、未收敛到最终答案）
   - 是否遗漏了关键信息？（如有图片路径却未使用图搜、应用浏览器访问来源页等）
3. 提取可执行的【修正策略】，供 Agent 在下一轮重试时直接遵循。

输出格式（严格使用以下 Markdown 结构）：

## 失败诊断
（1-3 句话概括核心失败原因）

## 问题步骤
- Step X: （具体描述哪一步出了什么问题）

## 修正策略
1. （具体、可执行的策略 1）
2. （具体、可执行的策略 2）
3. （可选更多）

## 建议的工具调用序列
（用自然语言描述重试时推荐的工具使用顺序，如：先 web_search → 再 sandbox_browser 访问 top-1 URL → 综合回答）
"""

SUCCESS_REFLECTION_SYSTEM_PROMPT = """你是一位资深的 AI Agent 教练，专门从【成功】的 ReAct 执行轨迹中提炼可复用的经验。

你的任务是：
1. 阅读完整轨迹（用户指令、推理、工具调用与返回、最终答案）。
2. 总结本题为何能答对：关键证据来自哪一步工具返回？搜索 query 有何特点？
3. 提炼可迁移的【成功经验】，供 Agent 在相似新题上参考（不要复述题目原文）。

输出格式（严格使用以下 Markdown 结构）：

## 成功要点
（1-3 句话：本题做对的核心原因）

## 有效工具链
1. （第 1 步工具及参数要点，如 search_text 的 query 关键词）
2. （第 2 步…，若无工具则写「纯推理」）

## 答案依据
（从哪条 snippet/页面信息得到最终答案；答案应如何表述，如 yes/no、人名、日期格式）
"""


class ReflectionManager:
    """
    反思管理器：在任务失败时构造高质量诊断 Prompt，发送给反思模型（默认 Qwen3.5-9B）。
    """

    def __init__(
        self,
        tool_error_threshold: int = 3,
        diagnostic_model_hint: str | None = None,
    ) -> None:
        self.tool_error_threshold = tool_error_threshold
        self.diagnostic_model_hint = (
            diagnostic_model_hint
            or os.getenv("REFLECTION_MODEL_NAME", "qwen-3.5").strip()
        )

    def create_state(
        self,
        instruction: str,
        image_path: str | None = None,
        max_steps: int = 10,
    ) -> ReflectionState:
        return ReflectionState(
            instruction=instruction,
            image_path=image_path,
            max_steps=max_steps,
            tool_error_threshold=self.tool_error_threshold,
        )

    def build_diagnostic_messages(
        self,
        trajectory: list[dict[str, Any]],
        failure_reason: FailureReason,
        instruction: str,
        image_path: str | None = None,
        final_pred: str | None = None,
        ground_truth: str | None = None,
    ) -> list[dict[str, str]]:
        """
        构造发送给外部诊断模型的 messages（OpenAI Chat 格式）。
        返回 [system, user] 两条消息。
        """
        trajectory_text = self.format_trajectory(trajectory)
        user_prompt = self._build_user_prompt(
            trajectory_text=trajectory_text,
            failure_reason=failure_reason,
            instruction=instruction,
            image_path=image_path,
            final_pred=final_pred,
            ground_truth=ground_truth,
        )
        return [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def build_diagnostic_request(
        self,
        trajectory: list[dict[str, Any]],
        failure_reason: FailureReason,
        instruction: str,
        image_path: str | None = None,
        final_pred: str | None = None,
        ground_truth: str | None = None,
    ) -> dict[str, Any]:
        """打包完整请求体，便于直接传给诊断 API。"""
        return {
            "model": self.diagnostic_model_hint,
            "messages": self.build_diagnostic_messages(
                trajectory=trajectory,
                failure_reason=failure_reason,
                instruction=instruction,
                image_path=image_path,
                final_pred=final_pred,
                ground_truth=ground_truth,
            ),
            "temperature": 0.3,
            "max_tokens": 2048,
        }

    _TRAJECTORY_TRUNC_MARKER = "\n\n...[内容已超长被截断]...\n\n"

    @classmethod
    def _truncate_tool_content_for_reflection(cls, content: str) -> str:
        """tool 返回过长时截断，避免反思模型 32k 上下文超限。"""
        if len(content) <= 1500:
            return content
        return (
            content[:1000]
            + cls._TRAJECTORY_TRUNC_MARKER
            + content[-500:]
        )

    @classmethod
    def format_trajectory(cls, trajectory: list[dict[str, Any]]) -> str:
        """将轨迹列表格式化为可读文本。"""
        if not trajectory:
            return "(empty trajectory)"

        lines: list[str] = []
        for rec in trajectory:
            step = rec.get("step_id", "?")
            role = rec.get("role", "?")
            ts = rec.get("timestamp", "")
            content = str(rec.get("content", ""))
            if role == "tool":
                content = cls._truncate_tool_content_for_reflection(content)
            tool_call_id = rec.get("tool_call_id")
            header = f"[Step {step} | {role} | t={ts}]"
            if tool_call_id:
                header += f" tool_call_id={tool_call_id}"
            lines.append(header)
            lines.append(content)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def load_trajectory_from_jsonl(path: str | Path) -> list[dict[str, Any]]:
        """从 trajectory.jsonl 加载轨迹。"""
        records: list[dict[str, Any]] = []
        file_path = Path(path)
        if not file_path.is_file():
            return records
        with file_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _build_user_prompt(
        self,
        trajectory_text: str,
        failure_reason: FailureReason,
        instruction: str,
        image_path: str | None,
        final_pred: str | None,
        ground_truth: str | None = None,
    ) -> str:
        reason_desc = {
            FailureReason.MAX_STEPS_EXCEEDED: (
                "Agent 达到最大循环轮数仍未输出有效最终答案。"
            ),
            FailureReason.CONSECUTIVE_TOOL_ERRORS: (
                f"工具调用连续报错达到阈值（{self.tool_error_threshold} 次）。"
            ),
            FailureReason.WRONG_ANSWER: (
                "Agent 给出了最终答案，但被判为错误（与参考答案不一致；"
                "闭卷评测时不应向反思模型提供参考答案）。"
            ),
        }[failure_reason]

        parts = [
            "## 任务信息",
            f"**用户指令**: {instruction}",
            f"**图片路径**: {image_path or '无'}",
            f"**失败类型**: {failure_reason.value}",
            f"**失败描述**: {reason_desc}",
        ]
        if final_pred:
            parts.append(f"**Agent 最终输出**: {final_pred}")
        if ground_truth:
            parts.append(f"**参考答案**: {ground_truth}")

        parts.extend([
            "",
            "## 完整执行轨迹",
            "```",
            trajectory_text,
            "```",
            "",
            "请根据上述轨迹完成诊断，并输出【修正策略】。",
        ])
        return "\n".join(parts)

    def build_success_messages(
        self,
        trajectory: list[dict[str, Any]],
        instruction: str,
        final_pred: str,
        ground_truth: str | None = None,
        image_path: str | None = None,
    ) -> list[dict[str, str]]:
        trajectory_text = self.format_trajectory(trajectory)
        parts = [
            "## 任务信息",
            f"**用户指令**: {instruction}",
            f"**图片路径**: {image_path or '无'}",
            f"**Agent 最终输出**: {final_pred}",
        ]
        if ground_truth:
            parts.append(f"**参考答案**: {ground_truth}")
        parts.extend(
            [
                "",
                "## 完整执行轨迹",
                "```",
                trajectory_text,
                "```",
                "",
                "本题已答对。请提炼可复用的成功经验（见系统提示中的 Markdown 结构）。",
            ]
        )
        return [
            {"role": "system", "content": SUCCESS_REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(parts)},
        ]

    def build_success_request(
        self,
        trajectory: list[dict[str, Any]],
        instruction: str,
        final_pred: str,
        ground_truth: str | None = None,
        image_path: str | None = None,
    ) -> dict[str, Any]:
        return {
            "model": self.diagnostic_model_hint,
            "messages": self.build_success_messages(
                trajectory=trajectory,
                instruction=instruction,
                final_pred=final_pred,
                ground_truth=ground_truth,
                image_path=image_path,
            ),
            "temperature": 0.2,
            "max_tokens": 1536,
        }

    def call_diagnostic_model(
        self,
        llm_client: Any,
        request: dict[str, Any],
    ) -> str:
        """
        调用外部诊断模型（预留接口）。
        llm_client 需实现 chat_completion(messages) -> dict。
        """
        messages = request["messages"]
        response = llm_client.chat_completion(messages)
        return response.get("content") or ""
