"""ReAct 核心引擎与消息状态管理。"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from llm_client import ChatLLMClient
from logger import AgentLogger
from memory import MemoryManager
from reflection import FailureReason, ReflectionManager
from tool_parser import normalize_llm_response
from tools import ToolRegistry, create_default_registry, create_production_registry
from vision_utils import build_vision_user_content, content_to_log_string

OSINT_SEARCH_WORKFLOW = """
## 核心工作流与搜索法则 (CRITICAL RULES)

**违反以下任一条将导致搜索失败；你必须严格遵守。**

1. 🚫 **严禁长句搜索**：遇到复杂的长篇 OSINT / 多跳问题，**绝不能**把原问题或长段落直接作为 `search_text` 的 `query`！
   你必须从题干中提取**最独特的**专有名词、人名、年份、机构名、地名，组合成 **3–8 个词的短查询**。
   - ❌ 错误：把整段 200 字问题粘贴进 query
   - ✅ 正确：`"MasterCraft Boat Holdings" stock buyback 2012`

2. 🕵️ **渐进式推理 (Step-by-Step OSINT)**：
   - **Step 1: 实体解密**。题干若有「隐藏实体」（如「那家生产汽艇的公司」「车祸死亡的青少年」），**禁止**直接搜最终答案。
     先用背景线索搜出**确切实体名称**（公司全名、人名等）。
   - **Step 2: 组装终极查询**。锁定实体后，再叠加年份、事件、指标词，搜索最终答案。

3. 📊 **特定领域深挖关键词**：
   - **财务 / 公司**：务必加入目标年份 + `annual report` / `10-K` / `stock buyback` / `shares outstanding` 等。
   - **新闻 / 事件**：善用 `news` / `obituary` / `interview`，并加地名或时间限制。

4. 🧠 **记忆备忘录**：每次收到 tool 返回后，在下一次 tool call **之前**，
   必须在思考中写下已确认的关键数字、人名、日期（一两句即可），防止长上下文丢线索。

5. ⚙️ **工具调用格式**：
   - 需要外部信息时，使用 **function calling** 调用 `search_text`（及可用的其他工具）。
   - `search_text` 的 `query` **必须简短**；`top_k` 建议 3–5；不要开启全文抓取除非必要。
   - 信息足够时，**仅输出纯文本最终答案**，不要再调用任何工具。
"""

REACT_SYSTEM_PROMPT = (
    "You are an expert OSINT ReAct agent solving difficult fact-finding tasks.\n\n"
    "## ReAct Protocol\n"
    "- When you need external information, call tools via **function calling** "
    "(not legacy Action/Action Input text).\n"
    "- After tool results return, reason briefly, then either call another tool "
    "with a **new short query** or give the final answer.\n"
    "- When you have enough evidence, reply with **plain text only** — no more tool calls.\n"
    "- Final answers must be **minimal**: a name, number, date, yes/no, or short phrase. "
    "No markdown, no explanation, no copying the whole question.\n\n"
    "## 强制前置计划法则 (Plan-and-Solve) — 不可跳过\n"
    "**这是硬性要求，不是建议。** 面对任何复杂 OSINT / 多跳 / 长程推理任务，"
    "在**第一次调用任何工具之前**，你必须先在回复中拆解任务并写出搜索计划。\n"
    "**禁止**在未输出 `<plan>` 的情况下直接发起 function calling。\n\n"
    "你的该轮回复必须**严格以 `<plan>` 标签开头**，格式如下（照抄结构，替换方括号内容）：\n\n"
    "<plan>\n"
    "当前终极目标：[用一句话总结你要找什么]\n"
    "缺失的关键线索：[目前还不知道的具体实体名、年份、数字等]\n"
    "计划步骤 1：[下一步立刻要做的动作，例如：先搜索出目标公司的全称]\n"
    "计划步骤 2：[预期的后续动作，例如：查阅该公司 2012 年回购相关披露]\n"
    "计划步骤 3：[最后一步的动作，例如：汇总并给出最终短答案]\n"
    "</plan>\n\n"
    "**输出顺序（必须遵守）：**\n"
    "1. 先输出完整的 `<plan>...</plan>`（任务开始时输出一次；仅当搜索方向错误或关键假设被推翻时，可重新输出修订版计划）。\n"
    "2. 紧接着用 1–3 句写出 Thought（当前思考与为何执行步骤 1）。\n"
    "3. 再通过 **function calling** 调用工具（等价于 Action；参数为 JSON，**不要**输出 `Action:` / `Action Input:` 纯文本格式）。\n"
    "4. 收到 tool 结果后：可简短 Thought → 新工具调用或最终答案；**不必**每轮都重复 `<plan>`，除非需要重大方向调整。\n\n"
    "**违规示例（严禁）：** 一上来就 `search_text`、没有 `<plan>`、或把整段问题当作 query。\n"
    + OSINT_SEARCH_WORKFLOW
)


Role = Literal["user", "assistant", "tool", "system"]


@dataclass
class ToolCallFunction:
    name: str
    arguments: str  # JSON 字符串，与 OpenAI API 一致


@dataclass
class ToolCall:
    id: str
    type: str
    function: ToolCallFunction


@dataclass
class Message:
    """标准消息结构，兼容 OpenAI Chat Completions 格式。"""

    role: Role
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        tool_calls = None
        if raw_calls := data.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    type=tc.get("type", "function"),
                    function=ToolCallFunction(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in raw_calls
            ]
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class AgentState:
    """会话状态：消息历史 + 元信息。"""

    messages: list[Message] = field(default_factory=list)
    instruction: str = ""
    image_path: str | None = None

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def history_dicts(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self.messages]


class ReactAgent:
    """
    ReAct 智能体主循环：
    推理 (Reasoning) → 行动 (Acting via tools) → 观察 (Observation) → 循环
    """

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        logger: AgentLogger | None = None,
        llm_client: ChatLLMClient | None = None,
        reflection_llm_client: ChatLLMClient | None = None,
        reflection_manager: ReflectionManager | None = None,
        memory_manager: MemoryManager | None = None,
        max_steps: int = 10,
        enable_reflection: bool = True,
        enable_memory: bool = True,
    ) -> None:
        self.tool_registry = tool_registry or create_default_registry()
        self.logger = logger or AgentLogger()
        self.llm_client = llm_client
        self.reflection_llm_client = reflection_llm_client
        self.reflection_manager = reflection_manager or ReflectionManager()
        self.memory_manager = memory_manager or MemoryManager()
        self.max_steps = max_steps
        self.enable_reflection = enable_reflection
        self.enable_memory = enable_memory
        self._mock_llm_turn = 0
        self._system_prompt = REACT_SYSTEM_PROMPT
        self.last_reflection: str | None = None
        self.last_success_reflection: str | None = None
        self.last_failure_reason: FailureReason | None = None
        self.last_retried: bool = False
        self._memory_user_suffix: str = ""

    def run(
        self,
        instruction: str,
        image_path: str | None = None,
        ground_truth: str | None = None,
        task_index: int | None = None,
        system_prompt: str | None = None,
        retry_on_wrong: bool = False,
    ) -> str:
        """执行 ReAct 主循环，返回最终答案。"""
        self._mock_llm_turn = 0
        if self.llm_client is None:
            raise ValueError(
                "未配置 LLM 客户端。请传入 llm_client，或在 main.py 中通过 create_llm_client() 创建。"
            )
        self.logger.reset_trajectory_step()
        self.last_reflection = None
        self.last_success_reflection = None
        self.last_failure_reason = None
        self.last_retried = False

        base_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self._system_prompt = base_prompt
        self._memory_user_suffix = ""
        if self.enable_memory:
            self._memory_user_suffix = self.memory_manager.build_memory_user_suffix(
                instruction, top_k=3, image_path=image_path
            )

        reflect_state = self.reflection_manager.create_state(
            instruction=instruction,
            image_path=image_path,
            max_steps=self.max_steps,
        )

        state = AgentState(instruction=instruction, image_path=image_path)

        user_content = self._append_user_suffix(
            self._build_user_content(instruction, image_path),
            self._memory_user_suffix,
        )
        user_msg = Message(role="user", content=user_content)
        state.append(user_msg)
        self._log_message(user_msg)

        final_answer = ""

        for step in range(self.max_steps):
            reflect_state.steps_taken = step + 1
            compacted_msgs = self._compact_messages(state.history_dicts())
            llm_response = normalize_llm_response(self._call_llm(compacted_msgs))
            # 终极拦截：在 Message 化之前直接重写原生 dict，避免脏 JSON 进入 state
            self._sanitize_llm_response_tool_calls_dict(llm_response)
            assistant_msg = Message.from_dict(llm_response)
            state.append(assistant_msg)
            self._log_message(assistant_msg)

            if assistant_msg.tool_calls:
                for tc in assistant_msg.tool_calls:
                    tool_name = tc.function.name
                    args = json.loads(tc.function.arguments)

                    tool = self.tool_registry.get(tool_name)
                    if tool is None:
                        result = f"Error: unknown tool '{tool_name}'"
                    else:
                        result = tool.execute(**args)

                    result = self._truncate_tool_result_for_agent(result)
                    reflect_state.record_tool_result(result)

                    tool_msg = Message(
                        role="tool",
                        content=result,
                        tool_call_id=tc.id,
                    )
                    state.append(tool_msg)
                    self._log_message(tool_msg)

                    failure = reflect_state.should_trigger()
                    if failure == FailureReason.CONSECUTIVE_TOOL_ERRORS:
                        self._run_reflection(
                            reflect_state, failure, instruction, image_path
                        )
                        final_answer = (
                            f"[Aborted: consecutive tool errors]\n"
                            f"{self.last_reflection or ''}"
                        )
                        break
                if final_answer:
                    break
                continue

            if assistant_msg.content and assistant_msg.content.strip():
                final_answer = assistant_msg.content.strip()
                reflect_state.got_final_answer = True
                break

            # 无 tool_calls 且无正文：常见于 thinking 占满 token，追加追问避免空转
            nudge = Message(
                role="user",
                content="请直接用一句话给出最终答案，不要调用任何工具。",
            )
            state.append(nudge)
            self._log_message(nudge)

        if not final_answer:
            final_answer = self._force_final_answer_after_max_steps(state)
            self._run_reflection(
                reflect_state,
                FailureReason.MAX_STEPS_EXCEEDED,
                instruction,
                image_path,
                final_pred=final_answer,
            )
        elif (
            self.enable_reflection
            and self.enable_memory
            and ground_truth
            and not final_answer.startswith("[")
            and self.last_reflection is None
            and not self._answers_match(final_answer, ground_truth)
        ):
            self._run_reflection(
                reflect_state,
                FailureReason.WRONG_ANSWER,
                instruction,
                image_path,
                final_pred=final_answer,
                ground_truth=ground_truth,
            )
        elif (
            self._should_store_success_memory()
            and ground_truth
            and final_answer.strip()
            and not final_answer.startswith("[")
            and self._answers_match(final_answer, ground_truth)
        ):
            self._run_success_reflection(
                instruction=instruction,
                image_path=image_path,
                final_pred=final_answer,
                ground_truth=ground_truth,
            )

        # 重试与 ground_truth 解耦：只要有反思诊断且允许重试即可（闭源打榜无标答时仍可用）
        if retry_on_wrong and self.last_reflection:
            strategy = MemoryManager.extract_correction_strategy(self.last_reflection)
            if strategy:
                retry_answer = self._retry_with_correction(
                    instruction=instruction,
                    image_path=image_path,
                    strategy=strategy,
                )
                if retry_answer.strip():
                    self.last_retried = True
                    final_answer = retry_answer.strip()
                    self.logger.log_trajectory(
                        role="user",
                        content=f"[同题重试] 已注入修正策略，新答案: {final_answer[:200]}",
                    )

        if not os.getenv("EVAL_DEFER_RESULT_LOG"):
            self.logger.log_result(
                instruction=instruction,
                image=image_path,
                answer=ground_truth,
                pred=final_answer,
                index=task_index,
                retried=self.last_retried,
            )
        return final_answer

    _TOOL_RESULT_TRUNC_MARKER = "\n...[内容过长，已被系统截断]...\n"
    _HISTORY_FOLD_MARKER = "\n...[历史记录已折叠以节省空间]..."

    @staticmethod
    def _message_char_size(msg: dict[str, Any]) -> int:
        """粗略估算单条消息占用字符数（含 tool_calls）。"""
        size = 0
        content = msg.get("content")
        if content is not None:
            if isinstance(content, str):
                size += len(content)
            elif isinstance(content, list):
                size += len(content_to_log_string(content))
            else:
                size += len(str(content))
        if raw_calls := msg.get("tool_calls"):
            size += len(json.dumps(raw_calls, ensure_ascii=False))
        return size

    @classmethod
    def _history_total_chars(cls, messages: list[dict[str, Any]]) -> int:
        return sum(cls._message_char_size(m) for m in messages)

    @classmethod
    def _fold_text(cls, text: str, head: int = 150) -> str:
        if len(text) <= head:
            return text
        return text[:head] + cls._HISTORY_FOLD_MARKER

    @classmethod
    def _fold_assistant_message(cls, msg: dict[str, Any], head: int = 150) -> bool:
        changed = False
        content = msg.get("content")
        if isinstance(content, str) and content:
            folded = cls._fold_text(content, head=head)
            if folded != content:
                msg["content"] = folded
                changed = True
        if raw_calls := msg.get("tool_calls"):
            compact_calls: list[dict[str, Any]] = []
            for tc in raw_calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                # 折叠时不得截断 JSON（会触发 vLLM Unterminated string 400）
                compact_calls.append(
                    {
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": {"name": name, "arguments": "{}"},
                    }
                )
            msg["tool_calls"] = compact_calls
            changed = True
        return changed

    @classmethod
    def _fold_tool_message(cls, msg: dict[str, Any], head: int = 150) -> bool:
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            return False
        folded = cls._fold_text(content, head=head)
        if folded == content:
            return False
        msg["content"] = folded
        return True

    @classmethod
    def _fold_user_message(cls, msg: dict[str, Any], text_head: int = 400) -> bool:
        """折叠中间轮次 user（多模态图替换为占位符），避免首条大图撑爆上下文。"""
        content = msg.get("content")
        if content is None:
            return False
        if isinstance(content, str):
            folded = cls._fold_text(content, head=text_head)
            if folded == content:
                return False
            msg["content"] = folded
            return True
        if not isinstance(content, list):
            return False
        changed = False
        new_parts: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:"):
                    new_parts.append(
                        {"type": "text", "text": "[image: folded for context]"}
                    )
                    changed = True
                    continue
            if part.get("type") == "text":
                text = part.get("text", "")
                folded = cls._fold_text(text, head=text_head)
                if folded != text:
                    changed = True
                new_parts.append({"type": "text", "text": folded})
                continue
            new_parts.append(part)
        if changed:
            msg["content"] = new_parts
        return changed

    def _compact_messages(
        self,
        messages: list[dict[str, Any]],
        max_chars: int = 12000,
    ) -> list[dict[str, Any]]:
        """
        请求 LLM 前压缩历史，不修改原始 messages（避免污染 logger 状态）。
        保留 system 与最后 3 条；从最早交互折叠中间 tool/assistant/user。
        """
        compacted = copy.deepcopy(messages)
        if self._history_total_chars(compacted) <= max_chars:
            return compacted

        protected_start = max(0, len(compacted) - 3)
        max_passes = max(len(compacted) * 3, 1)

        for _ in range(max_passes):
            if self._history_total_chars(compacted) <= max_chars:
                break

            changed = False
            for i in range(protected_start):
                msg = compacted[i]
                role = msg.get("role")
                if role == "system":
                    continue
                if role == "tool":
                    changed = self._fold_tool_message(msg) or changed
                elif role == "assistant":
                    changed = self._fold_assistant_message(msg) or changed
                elif role == "user":
                    changed = self._fold_user_message(msg) or changed

            if not changed:
                break

        self._sanitize_messages_tool_calls_dicts(compacted)
        return compacted

    @staticmethod
    def _sanitize_llm_response_tool_calls_dict(llm_response: dict[str, Any]) -> None:
        """在 Message.from_dict 之前净化 tool_calls.arguments（操作原生 dict）。"""
        tool_calls = llm_response.get("tool_calls")
        if not isinstance(tool_calls, list):
            return
        for tc_dict in tool_calls:
            fn = tc_dict.get("function")
            if not isinstance(fn, dict) or "arguments" not in fn:
                continue
            raw_args = fn.get("arguments", "")
            if isinstance(raw_args, dict):
                fn["arguments"] = json.dumps(raw_args, ensure_ascii=False)
                continue
            try:
                parsed_args = json.loads(raw_args if isinstance(raw_args, str) else "{}")
                if isinstance(parsed_args, dict):
                    fn["arguments"] = json.dumps(parsed_args, ensure_ascii=False)
                else:
                    fn["arguments"] = "{}"
            except Exception:
                fn["arguments"] = "{}"

    @classmethod
    def _sanitize_messages_tool_calls_dicts(cls, messages: list[dict[str, Any]]) -> None:
        """压缩后的 history 发往 vLLM 前，确保所有 assistant tool_calls 为合法 JSON。"""
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tc_dict in tool_calls:
                fn = tc_dict.get("function")
                if not isinstance(fn, dict) or "arguments" not in fn:
                    continue
                _, sanitized = cls._sanitize_tool_call_arguments(
                    fn.get("arguments"), fn.get("name", "")
                )
                fn["arguments"] = sanitized

    @staticmethod
    def _sanitize_tool_call_arguments(
        raw_args: str | dict[str, Any] | None,
        tool_name: str = "",
    ) -> tuple[dict[str, Any], str]:
        """
        解析 tool_call arguments，并返回可安全写回 history 的 JSON 字符串。
        修复尾随垃圾（Extra data）、尾逗号等，避免 vLLM Chat Template 400。
        """
        _ = tool_name
        if isinstance(raw_args, dict):
            return raw_args, json.dumps(raw_args, ensure_ascii=False, separators=(",", ":"))
        if not isinstance(raw_args, str) or not raw_args.strip():
            return {}, "{}"

        raw = raw_args.strip()

        def _pack(parsed: Any) -> tuple[dict[str, Any], str]:
            if isinstance(parsed, dict):
                return parsed, json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
            return {}, "{}"

        try:
            return _pack(json.loads(raw))
        except json.JSONDecodeError as exc:
            if "Extra data" in str(exc):
                try:
                    parsed, _ = json.JSONDecoder().raw_decode(raw)
                    return _pack(parsed)
                except json.JSONDecodeError:
                    pass
            fixed = re.sub(r",\s*([}\]])", r"\1", raw)
            try:
                return _pack(json.loads(fixed))
            except json.JSONDecodeError:
                try:
                    parsed, _ = json.JSONDecoder().raw_decode(fixed)
                    return _pack(parsed)
                except json.JSONDecodeError:
                    pass
        return {}, "{}"

    @classmethod
    def _truncate_tool_result_for_agent(cls, result: Any, max_len: int = 2000) -> str:
        """限制 tool 返回进入对话历史的体积，避免基座模型迷失或超时。"""
        if isinstance(result, str):
            text = result
        elif isinstance(result, (dict, list)):
            text = json.dumps(result, ensure_ascii=False)
        else:
            text = str(result)
        if len(text) <= max_len:
            return text
        return text[:1000] + cls._TOOL_RESULT_TRUNC_MARKER + text[-500:]

    def _force_final_answer_after_max_steps(self, state: AgentState) -> str:
        """耗尽 max_steps 后强制榨取一次最终答案，避免白卷占位符。"""
        squeeze = Message(
            role="user",
            content=(
                "你已耗尽最大探索步数。请不要再调用任何工具，立刻回顾上述所有搜索和浏览记录，"
                "尽你最大可能推断出最终答案。即使不确定，也要给出一个最有可能的名词或实体作为最终答案。"
            ),
        )
        state.append(squeeze)
        self._log_message(squeeze)
        compacted_msgs = self._compact_messages(state.history_dicts())
        llm_response = normalize_llm_response(
            self._call_llm(compacted_msgs, use_tools=False)
        )
        content = (llm_response.get("content") or "").strip()
        if content and not llm_response.get("tool_calls"):
            return content
        return "[Agent reached max_steps without a final answer]"

    def _build_user_content(
        self, instruction: str, image_path: str | None
    ) -> str | list[dict[str, Any]]:
        text = instruction
        if image_path:
            text += (
                f"\n\n[System Note: 本次任务附带了一张本地图片，路径为 {image_path}。"
                "当你需要调用 search_image 工具时，请务必直接将此路径字符串作为参数传入，"
                "绝对不要自行编造 HTTP URL！]"
            )
        return build_vision_user_content(text, image_path)

    @staticmethod
    def _append_user_suffix(
        content: str | list[dict[str, Any]],
        suffix: str,
    ) -> str | list[dict[str, Any]]:
        if not suffix:
            return content
        if isinstance(content, str):
            return f"{content}\n\n{suffix}"
        return [*content, {"type": "text", "text": suffix}]

    def _retry_with_correction(
        self,
        instruction: str,
        image_path: str | None,
        strategy: str,
    ) -> str:
        """答错反思后同题重答一次（Hermes 式 in-turn recovery 的 VQA 变体）。"""
        hint = (
            "<correction-hint>\n"
            "[System note: 上一轮答案有误。以下是诊断后的修正策略，不是用户新输入。]\n\n"
            f"{strategy}\n"
            "</correction-hint>\n"
            "请结合图片与修正策略重新作答：只输出最短的最终答案（一个词/短语/数字），"
            "不要解释，不要调用工具。"
        )
        user_content = self._append_user_suffix(
            self._build_user_content(instruction, image_path), hint
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        compacted_msgs = self._compact_messages(messages)
        llm_response = normalize_llm_response(self._call_llm(compacted_msgs))
        content = llm_response.get("content") or ""
        return content.strip()

    def _log_message(self, message: Message) -> None:
        content = content_to_log_string(message.content)
        tool_calls: list[dict[str, Any]] | None = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        self.logger.log_trajectory(
            role=message.role,
            content=content,
            tool_call_id=message.tool_call_id,
            tool_calls=tool_calls,
        )

    def _prepare_messages_for_api(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """注入 system prompt；记忆已在首条 user 中（见 run）。"""
        prepared: list[dict[str, Any]] = []
        if not messages or messages[0].get("role") != "system":
            prepared.append({"role": "system", "content": self._system_prompt})
        prepared.extend(messages)
        return prepared

    def _call_llm(
        self, messages: list[dict[str, Any]], *, use_tools: bool = True
    ) -> dict[str, Any]:
        """
        调用 LLM（EAS 或 SGLang，由 LLM_BACKEND 决定）。
        与 harness task_runner 一致：传入 tools + tool_choice=auto，解析 tool_calls。
        """
        api_messages = self._prepare_messages_for_api(messages)
        tools = self.get_tool_schemas() if use_tools else None
        return self.llm_client.chat_completion(  # type: ignore[union-attr]
            api_messages,
            tools=tools,
        )

    def _call_llm_mock(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """本地 Mock，仅用于无 Token 时的离线调试。"""
        self._mock_llm_turn += 1

        if self._mock_llm_turn == 1:
            user_text = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_text = content_to_log_string(msg.get("content"))
                    break
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_mock_001",
                        "type": "function",
                        "function": {
                            "name": "mock_search",
                            "arguments": json.dumps(
                                {"query": user_text[:80]},
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }

        last_tool_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "tool":
                last_tool_content = msg.get("content", "")
                break

        return {
            "role": "assistant",
            "content": (
                f"根据工具返回的信息，我的结论是：\n"
                f"{last_tool_content[:200]}\n"
                f"（以上为 Mock LLM 生成的最终回答，接入真实 API 后将由模型生成。）"
            ),
        }

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """供真实 LLM 调用时传入 tools 参数。"""
        return self.tool_registry.get_schemas()

    @staticmethod
    def _answers_match(pred: str, gold: str) -> bool:
        def norm(text: str) -> str:
            s = text.lower().strip()
            s = re.sub(r"[\s\u3000]+", " ", s)
            s = re.sub(r"[，。！？、；：""''（）\[\]【】]", "", s)
            return s.strip()

        p, g = norm(pred), norm(gold)
        if not g:
            return False
        return p == g or g in p or p in g

    def _should_store_success_memory(self) -> bool:
        if not (self.enable_reflection and self.enable_memory):
            return False
        v = os.getenv("MEMORY_STORE_SUCCESS", "true").lower()
        return v in ("1", "true", "yes")

    def _run_success_reflection(
        self,
        instruction: str,
        image_path: str | None,
        final_pred: str,
        ground_truth: str | None = None,
    ) -> None:
        """答对后：将本题轨迹送反思模型总结成功经验并写入记忆库。"""
        if not self.enable_reflection or not self.enable_memory:
            return

        self.last_success_reflection = None
        trajectory = self.logger.read_trajectory()
        request = self.reflection_manager.build_success_request(
            trajectory=trajectory,
            instruction=instruction,
            final_pred=final_pred,
            ground_truth=ground_truth,
            image_path=image_path,
        )
        diagnostic_client = self.reflection_llm_client or self.llm_client
        if diagnostic_client is not None:
            try:
                self.last_success_reflection = (
                    self.reflection_manager.call_diagnostic_model(
                        diagnostic_client, request
                    )
                )
            except Exception as exc:
                self.last_success_reflection = (
                    f"[Success reflection failed: {exc}]"
                )
        else:
            self.last_success_reflection = request["messages"][1]["content"]

        if self.last_success_reflection and not self.last_success_reflection.startswith(
            "[Success reflection failed"
        ):
            self.memory_manager.add_memory_from_success_reflection(
                instruction=instruction,
                reflection_text=self.last_success_reflection,
                image_path=image_path,
            )
            self.logger.log_trajectory(
                role="user",
                content=(
                    "[成功经验已写入记忆库]\n"
                    f"{self.last_success_reflection[:600]}"
                ),
            )

    def _run_reflection(
        self,
        reflect_state: Any,
        failure_reason: FailureReason,
        instruction: str,
        image_path: str | None,
        final_pred: str | None = None,
        ground_truth: str | None = None,
    ) -> None:
        """构造反思诊断 Prompt；若配置了诊断 LLM 则自动调用。"""
        if not self.enable_reflection:
            return

        self.last_failure_reason = failure_reason
        trajectory = self.logger.read_trajectory()
        request = self.reflection_manager.build_diagnostic_request(
            trajectory=trajectory,
            failure_reason=failure_reason,
            instruction=instruction,
            image_path=image_path,
            final_pred=final_pred,
            ground_truth=ground_truth,
        )

        diagnostic_client = self.reflection_llm_client or self.llm_client
        if diagnostic_client is not None:
            try:
                self.last_reflection = self.reflection_manager.call_diagnostic_model(
                    diagnostic_client, request
                )
            except Exception as exc:
                self.last_reflection = (
                    f"[Reflection prompt built; diagnostic call failed: {exc}]\n\n"
                    f"User prompt preview:\n{request['messages'][1]['content'][:800]}..."
                )
        else:
            self.last_reflection = request["messages"][1]["content"]

        if self.enable_memory and self.last_reflection:
            self.memory_manager.add_memory_from_reflection(
                instruction=instruction,
                reflection_text=self.last_reflection,
                image_path=image_path,
            )
