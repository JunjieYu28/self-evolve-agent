"""
Pro Max Agent — Self-Evolving ReAct Agent (V2 Architecture)
============================================================

核心改进:
1. 工具调用清洁化 — strip XML 重复, cap per-step/per-task
2. 强制终结答案 — 永远不返回空答案
3. 上下文压缩集成 — 动态 compaction 防止 token 爆炸
4. 自适应步数 — 根据任务难度调整预算
5. Stuck 检测 — 重复搜索自动干预
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Optional

from openai import OpenAI

import config
from roles import Role
from trajectory import Trajectory
from modules.planner import Planner, PlanResult
from modules.prompt_builder import PromptBuilder
from modules.reflection import ReflectionModule
from modules.memory import MemoryStore
from modules.compaction import Compactor

# Layer 2 imports (conditional)
from modules.teacher_client import TeacherClient
from modules.planner_32b import Planner32B
from modules.verifier_32b import Verifier32B
from modules.reflector_32b import Reflector32B

from tools.search_tool import search_text, search_image, fetch_url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("harness.agent")


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Web text search. Returns [{rank,title,url,snippet,content}]. Use fetch=false first for quick snippets, then fetch=true only when needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords (4-6 words optimal)"},
                    "top_k": {"type": "integer", "description": "Number of results (1-5)", "default": 3},
                    "fetch": {"type": "boolean", "description": "Whether to fetch full page content", "default": False},
                    "max_chars": {"type": "integer", "description": "Max content chars", "default": 800},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_image",
            "description": "Reverse image search. Input must be http(s) image URL. Returns [{rank,title,url,snippet,content}].",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "description": "Image http(s) URL"},
                    "top_k": {"type": "integer", "description": "Number of results", "default": 2},
                    "fetch": {"type": "boolean", "description": "Whether to fetch full content", "default": True},
                    "max_chars": {"type": "integer", "description": "Max content chars", "default": 800},
                },
                "required": ["image_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch full text content from a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Max content chars", "default": 2000},
                },
                "required": ["url"],
            },
        },
    },
]

TOOL_FN_MAP = {
    "search_text": lambda a: search_text(**a),
    "search_image": lambda a: search_image(**a),
    "fetch_url": lambda a: fetch_url(**a),
}

BUDGET_WARNING = (
    "[System Notice] You are running low on steps. "
    "Give your best answer NOW based on all evidence gathered. "
    "Output ONLY the answer — no explanations, no tool calls."
)

FORCE_ANSWER_PROMPT = (
    "[FINAL STEP] Your search budget is exhausted. "
    "Reply with ONLY the answer — a short entity name, number, date, or yes/no. "
    "Rules:\n"
    "1. Output ONLY the answer itself, nothing else.\n"
    "2. No explanations, no reasoning, no 'Based on...' preambles.\n"
    "3. No section headers like 'Analyze the Request:' — ONLY the answer entity.\n"
    "4. No 'I cannot determine' or 'No answer found' — always guess.\n"
    "5. If you found multiple candidates, pick the BEST one.\n"
    "6. Maximum 5 words.\n\n"
    "Your answer:"
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class Agent:
    def __init__(
        self,
        llm_base_url: str = config.LLM_BASE_URL,
        model_name: str = config.MODEL_NAME,
        max_steps: int = config.MAX_STEPS,
        memory_store: Optional[MemoryStore] = None,
        teacher_enabled: bool = config.TEACHER_ENABLED,
    ):
        self.client = OpenAI(base_url=llm_base_url, api_key="EMPTY")
        self.model_name = model_name
        self.max_steps = max_steps
        self.memory_store = memory_store
        self.planner = Planner()
        self.prompt_builder = PromptBuilder()
        self.reflector = ReflectionModule()
        self.compactor = Compactor()

        # Layer 2: 32B components
        self.teacher: Optional[TeacherClient] = None
        self.planner_32b: Optional[Planner32B] = None
        self.verifier_32b: Optional[Verifier32B] = None
        self.reflector_32b: Optional[Reflector32B] = None

        if teacher_enabled:
            self.teacher = TeacherClient()
            if config.TEACHER_PLANNER_ENABLED:
                self.planner_32b = Planner32B(self.teacher)
            if config.TEACHER_VERIFIER_ENABLED:
                self.verifier_32b = Verifier32B(self.teacher)
            if config.TEACHER_REFLECTOR_ENABLED:
                self.reflector_32b = Reflector32B(self.teacher)
            logger.info("Layer 2 enabled: planner=%s, verifier=%s, reflector=%s",
                        self.planner_32b is not None,
                        self.verifier_32b is not None,
                        self.reflector_32b is not None)

        self.total_tokens_used = 0
        self.tool_call_count = 0
        self._seen_queries: set = set()
        self._stuck_counter = 0

    def run_task(
        self,
        task: dict,
        trajectory_dir: str = "trajectories",
    ) -> dict:
        task_id = task.get("id") or str(uuid.uuid4())[:8]
        instruction = task["instruction"]
        image_b64 = task.get("image_b64")
        image_url = task.get("image_url")
        has_image = bool(image_b64 or image_url)

        logger.info("run_task: task_id=%s", task_id)
        start_time = time.time()
        self._seen_queries = set()
        self._stuck_counter = 0
        self.tool_call_count = 0
        self.total_tokens_used = 0
        self._force_tool_next_step = False

        # ----- 1. PLAN -----
        plan = self.planner.plan(instruction, has_image=has_image)
        adaptive_max_steps = self._compute_adaptive_budget(plan)
        logger.info("plan: type=%s, budget=%d steps", plan.task_type, adaptive_max_steps)

        # ----- 1B. DEEP PLAN (32B) -----
        plan_32b_section = ""
        deep_plan = None
        self._current_answer_type = None
        self._current_question = instruction
        if self.planner_32b:
            try:
                deep_plan = self.planner_32b.plan(instruction, has_image=has_image)
                if deep_plan.success:
                    plan_32b_section = self.planner_32b.format_as_prompt_section(deep_plan)
                    self._current_answer_type = deep_plan.expected_answer_type
            except Exception as exc:
                logger.warning("32B planner failed (non-fatal): %s", exc)

        # ----- 2. RECALL (strategy memory) -----
        strategy_block = ""
        if self.memory_store:
            strategy_block = self.memory_store.get_strategies(plan.task_type)

        # ----- 3. BUILD PROMPT -----
        system_prompt = self.prompt_builder.build(
            plan=plan,
            strategy_memory_block=strategy_block if strategy_block else None,
            plan_32b_section=plan_32b_section if plan_32b_section else None,
            expected_answer_type=self._current_answer_type,
            question=instruction,
            has_image=has_image,
        )

        # ----- 4. EXECUTE (ReAct loop) -----
        traj = Trajectory(task_id, output_dir=trajectory_dir)
        traj.write(Role.SYSTEM, system_prompt, step_id=0)

        user_content = self._build_user_content(instruction, image_b64, image_url)
        traj.write(Role.USER, user_content, step_id=0)

        final_answer = ""
        step = 0

        for step in range(1, adaptive_max_steps + 1):
            # Build context with compaction
            messages = self._build_context(traj, step, adaptive_max_steps)

            # Check force-answer conditions
            if self._should_force_answer(step, adaptive_max_steps):
                final_answer = self._force_answer(messages, traj)
                break

            # LLM call
            extra_body = {"enable_thinking": True}

            # Force tool usage if model tried to answer too early last step
            current_tool_choice = "required" if self._force_tool_next_step else "auto"
            self._force_tool_next_step = False

            try:
                kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=config.MAX_TOKENS,
                    temperature=0.7,
                    tools=TOOLS_SCHEMA,
                    tool_choice=current_tool_choice,
                )
                if extra_body:
                    kwargs["extra_body"] = extra_body
                response = self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                logger.error("LLM call failed at step %d: %s", step, exc)
                traj.write(Role.TOOL, f"[HARNESS ERROR] LLM call failed: {exc}", step_id=step)
                break

            if response.usage:
                self.total_tokens_used += response.usage.total_tokens

            # Sanitize response
            content, tool_calls, reasoning = self._sanitize_response(response)

            # Record assistant turn
            extra = {}
            if tool_calls:
                extra["tool_calls"] = [
                    tc.model_dump() if hasattr(tc, 'model_dump') else tc
                    for tc in tool_calls
                ]
            if reasoning:
                extra["reasoning_content"] = reasoning
            if response.usage:
                extra["total_tokens"] = response.usage.total_tokens

            traj.write(Role.ASSISTANT, content, step_id=step, extra=extra or None)

            # Exit condition: no tool calls and has meaningful content
            if not tool_calls and content.strip():
                # Minimum search effort: don't accept answer before 4 tool calls
                # But don't force more searches if we're running out of steps
                if self.tool_call_count < 4 and step < adaptive_max_steps - 1:
                    # Rewrite this step: replace the premature answer with a
                    # continuation signal so model doesn't see its own answer
                    traj.rewrite_last_assistant(
                        "I need to verify this with more searches before concluding.",
                        step_id=step,
                    )
                    # Inject a directive message that forces tool usage
                    remaining = 4 - self.tool_call_count
                    verify_msg = (
                        f"[System] You must perform at least {remaining} more search(es) "
                        "before giving your final answer. Use the search_text tool NOW "
                        "with a verification query. Do NOT output your answer yet — "
                        "call search_text with a query to cross-check your findings."
                    )
                    traj.write(Role.USER, verify_msg, step_id=step)
                    # Force next LLM call to use a tool
                    self._force_tool_next_step = True
                elif not self._looks_like_thinking(content, step):
                    final_answer = self._extract_answer_from_content(content)
                    break

            if not tool_calls and not content.strip():
                # Model produced no content and no tool calls — check reasoning
                if reasoning:
                    # In later steps or if reasoning contains strong answer signals, try extraction
                    if step >= adaptive_max_steps - 3 or self._reasoning_has_conclusion(reasoning):
                        answer_from_reasoning = self._extract_answer_from_reasoning(reasoning)
                        if answer_from_reasoning and self._validate_answer_type(answer_from_reasoning):
                            final_answer = answer_from_reasoning
                            break
                        # 正则失败 → 32B 反思式提取 (only in late steps to save tokens)
                        if step >= adaptive_max_steps - 2 and self.teacher:
                            reflected = self._teacher_reflect_on_reasoning(reasoning, instruction)
                            if reflected:
                                final_answer = reflected
                                break
                continue

            # Dispatch tools
            if tool_calls:
                self._dispatch_and_record(traj, tool_calls, step)

                # Mid-task 32B reflection: redirect search if stuck or at midpoint
                if self.reflector_32b and self._should_mid_reflect(step, adaptive_max_steps):
                    redirect_hint = self._mid_task_redirect(traj, instruction, step, adaptive_max_steps)
                    if redirect_hint:
                        traj.write(Role.USER, redirect_hint, step_id=step)

        # ----- FORCE ANSWER FALLBACK -----
        if not final_answer or final_answer.startswith("[HARNESS]"):
            messages = traj.to_messages()
            final_answer = self._force_answer(messages, traj)

        elapsed = time.time() - start_time

        # ----- 4B. VERIFY (32B) -----
        if self.verifier_32b and final_answer and not final_answer.startswith("[HARNESS]"):
            final_answer = self._verify_answer(traj, final_answer, instruction, step)

        # ----- 4C. FINAL CLEANUP -----
        # Ensure answer is valid
        if not final_answer or final_answer.startswith("[HARNESS]") or not self._is_valid_answer_candidate(final_answer):
            final_answer = self._extract_best_candidate_from_history(traj.to_messages(), traj)

        elapsed = time.time() - start_time

        result = {
            "task_id": task_id,
            "instruction": instruction,
            "answer": final_answer,
            "steps": step,
            "trajectory_path": str(traj.path),
            "elapsed_seconds": round(elapsed, 2),
            "total_tokens": self.total_tokens_used,
            "tool_calls": self.tool_call_count,
            "task_type": plan.task_type,
        }

        # ----- 5. REFLECT + 6. STORE -----
        self._reflect_and_store(traj, task_id, plan, result)

        return result

    # ===================================================================
    # Context Building
    # ===================================================================

    def _build_context(self, traj: Trajectory, step: int, max_steps: int) -> list[dict]:
        messages = traj.to_messages()

        # Apply compaction if needed
        if self.compactor.should_compact(messages, self.total_tokens_used):
            messages = self.compactor.compact(messages, self.total_tokens_used)
            traj.replace_messages(messages)

        # Add budget warning near end
        if step >= max_steps - 2:
            messages.append({"role": "user", "content": BUDGET_WARNING})

        return messages

    def _compute_adaptive_budget(self, plan: PlanResult) -> int:
        if not config.ADAPTIVE_BUDGET:
            return self.max_steps
        base = {
            "multi_constraint": 20,
            "visual_entity": 12,
            "comparison": 8,
            "general": 15,
        }
        return min(base.get(plan.task_type, 15), self.max_steps)

    # ===================================================================
    # Response Sanitization (Critical: fixes tool call explosion)
    # ===================================================================

    def _sanitize_response(self, response):
        choice = response.choices[0]
        msg = choice.message
        raw_content = msg.content or ""
        reasoning = getattr(msg, 'reasoning_content', '') or ""
        tool_calls = list(msg.tool_calls or [])

        # Strip <tool_call> XML from content (prevents replay bloat)
        content = re.sub(r'<tool_call>.*?</tool_call>', '', raw_content, flags=re.DOTALL)
        # Strip <think> blocks from content (reasoning is stored separately)
        if '</think>' in content:
            content = content.split('</think>', 1)[1]
        content = content.strip()

        # If no API tool_calls but found text tool calls, parse them
        if not tool_calls and '<tool_call>' in raw_content:
            text_calls = self._parse_text_tool_calls(raw_content)
            tool_calls = text_calls

        # Deduplicate within this step (same function + same args = duplicate)
        tool_calls = self._dedup_tool_calls(tool_calls)

        # Cap per-step
        tool_calls = tool_calls[:config.MAX_TOOLS_PER_STEP]

        return content, tool_calls, reasoning

    def _dedup_tool_calls(self, tool_calls) -> list:
        seen = set()
        result = []
        for tc in tool_calls:
            if hasattr(tc, 'function'):
                key = (tc.function.name, tc.function.arguments)
            elif isinstance(tc, dict):
                fn = tc.get("function", {})
                key = (fn.get("name", ""), fn.get("arguments", ""))
            else:
                result.append(tc)
                continue

            if key not in seen:
                seen.add(key)
                result.append(tc)
        return result

    # ===================================================================
    # Force Answer
    # ===================================================================

    def _should_force_answer(self, step: int, max_steps: int) -> bool:
        if step >= max_steps:
            return True
        if self.tool_call_count >= config.MAX_TOOL_CALLS_PER_TASK:
            logger.info("Tool budget exhausted (%d), forcing answer", self.tool_call_count)
            return True
        if self._stuck_counter >= config.STUCK_THRESHOLD:
            logger.info("Stuck detected (%d repetitions), forcing answer", self._stuck_counter)
            return True
        return False

    def _get_force_answer_prompt(self) -> str:
        """Generate type-aware force answer prompt."""
        answer_type = getattr(self, '_current_answer_type', None)
        if not answer_type or answer_type == "other":
            return FORCE_ANSWER_PROMPT

        type_descriptions = {
            "person_name": "a person's full name (e.g., 'Albert Einstein')",
            "movie_title": "a movie or show title",
            "organization": "a company or organization name",
            "number": "a number or quantity (e.g., '42,000')",
            "date": "a date (e.g., 'March 14, 2023')",
            "yes_no": "exactly Yes or No (one word)",
            "location": "a place name (e.g., 'Paris, France')",
            "scientific_name": "a scientific binomial name (e.g., 'Canis lupus')",
        }
        type_desc = type_descriptions.get(answer_type, answer_type)

        return (
            "[FINAL STEP] Your search budget is exhausted. "
            "Reply with ONLY the answer.\n"
            "Rules:\n"
            f"1. Your answer MUST be {type_desc}.\n"
            "2. Output ONLY the answer itself, nothing else.\n"
            "3. No explanations, no reasoning, no preambles.\n"
            "4. No section headers like 'Analyze the Request:' — ONLY the answer entity.\n"
            "5. Always give your best guess. Never say 'I cannot determine'.\n"
            "6. Maximum 5 words.\n\n"
            "Your answer:"
        )

    def _force_answer(self, messages: list[dict], traj: Optional[Trajectory] = None) -> str:
        messages = list(messages)
        messages.append({"role": "user", "content": self._get_force_answer_prompt()})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=2000,
                temperature=0.3,
                extra_body={"enable_thinking": True},
            )
            if response.usage:
                self.total_tokens_used += response.usage.total_tokens

            msg = response.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, 'reasoning_content', '') or ""

            if '</think>' in content:
                content = content.split('</think>', 1)[1]
            content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
            content = re.sub(r'<tool_call>.*', '', content, flags=re.DOTALL)
            content = content.strip()

            words = content.split()
            content_lower = f" {content.lower()} "
            if len(words) > 8 or (len(words) > 4 and any(
                v in content_lower for v in [
                    ' find ', ' search', ' try ', ' need ', ' check ', ' look ',
                    ' verify', ' determine', ' cannot', ' could not', ' unable',
                    ' giving', ' found ', ' based on', ' according to',
                    ' should ', ' would ', ' might ', ' about ',
                ]
            )):
                # Before discarding, try to extract an entity from it
                entity = self._extract_entity_from_verbose_content(content)
                if entity and self._is_valid_answer_candidate(entity) and self._validate_answer_type(entity):
                    return entity
                content = ""

            answer = self._extract_answer_from_content(content)
            if answer and self._validate_answer_type(answer):
                return answer

            # content 为空 → 从 reasoning 中提取
            if reasoning:
                answer = self._extract_answer_from_reasoning(reasoning)
                if answer and self._validate_answer_type(answer):
                    return answer
                # 正则提取失败 → 32B 反思式提取
                if self.teacher:
                    reflected = self._teacher_reflect_on_reasoning(
                        reasoning, getattr(self, '_current_question', '')
                    )
                    if reflected:
                        return reflected

        except Exception as exc:
            logger.error("Force answer LLM call failed: %s", exc)

        # Fallback: 用 32B 总结证据，再让 9B 回答（32B 不直接回答）
        if self.teacher and traj:
            answer = self._teacher_assisted_9b_answer(traj, messages)
            if answer:
                return answer

        # Last resort: scan trajectory for candidate entities
        return self._extract_best_candidate_from_history(messages, traj)

    def _extract_answer_from_reasoning(self, reasoning: str) -> str:
        """Extract a concise answer from reasoning_content (thinking mode output).

        Enhanced: multi-candidate ranking, more patterns, last-conclusion preference.
        """
        if not reasoning:
            return ""

        # Focus on the LAST portion of reasoning (where conclusions are)
        # But also scan earlier for explicit answer statements
        last_chunk = reasoning[-4000:] if len(reasoning) > 4000 else reasoning

        candidates_with_score: list[tuple[str, int, int]] = []  # (candidate, score, position)

        # Pattern group 1: Explicit answer declarations (highest priority)
        explicit_patterns = [
            (r'(?:my\s+)?(?:final\s+)?answer\s*(?:is|would be|should be|:)\s*[:\-]?\s*["\']?([^"\'\n,;]{2,80})["\']?', 10),
            (r'(?:the\s+answer\s+(?:is|to this|to the question))\s*[:\-]?\s*["\']?([^"\'\n,;]{2,80})["\']?', 10),
            (r'answer\s*[:=]\s*["\']?([^"\'\n,;]{2,80})["\']?', 9),
            (r'(?:I\'ll|I will|let me)\s+(?:go with|output|answer|give)\s+["\']?([^"\'\n,;]{2,80})["\']?', 8),
        ]
        for pat, score in explicit_patterns:
            for match in re.finditer(pat, last_chunk, re.IGNORECASE):
                candidate = match.group(1).strip().rstrip('.')
                pos = match.start()
                if self._is_valid_answer_candidate(candidate):
                    candidates_with_score.append((candidate, score, pos))

        # Pattern group 2: Conclusion indicators
        conclusion_patterns = [
            (r'(?:therefore|thus|so|hence|in conclusion)[,:]?\s+(?:the\s+answer\s+(?:is|would be)\s+)?["\']?([^"\'\n,;]{2,80})["\']?', 7),
            (r'(?:I\s+(?:believe|think|conclude|determine))\s+(?:the answer is\s+|it(?:\'s| is)\s+)?["\']?([^"\'\n,;]{2,80})["\']?', 7),
            (r'(?:this\s+(?:is|must be|has to be|should be))\s+["\']?([^"\'\n,;]{2,80})["\']?', 6),
            (r'(?:it\s+(?:is|was|must be|should be|appears to be))\s+["\']?([A-Z][^"\'\n,;]{1,80})["\']?', 5),
        ]
        for pat, score in conclusion_patterns:
            for match in re.finditer(pat, last_chunk, re.IGNORECASE):
                candidate = match.group(1).strip().rstrip('.')
                pos = match.start()
                if self._is_valid_answer_candidate(candidate):
                    candidates_with_score.append((candidate, score, pos))

        # Pattern group 3: Entity identification statements
        entity_patterns = [
            (r'(?:the\s+(?:person|player|actor|director|individual|character|company|team|movie|show|film|song|species|organization|name)\s+(?:is|was|must be|appears to be|seems to be))\s+["\']?([^"\'\n,;]{2,80})["\']?', 6),
            (r'(?:identified as|confirmed as|matches?:?)\s+["\']?([A-Z][^"\'\n,;]{1,80})["\']?', 6),
            (r'(?:correct answer|right answer).*?(?:is|:)\s*["\']?([^"\'\n,;]{2,80})["\']?', 8),
        ]
        for pat, score in entity_patterns:
            for match in re.finditer(pat, last_chunk, re.IGNORECASE):
                candidate = match.group(1).strip().rstrip('.')
                pos = match.start()
                if self._is_valid_answer_candidate(candidate):
                    candidates_with_score.append((candidate, score, pos))

        # Pattern group 4: Bolded entities (often the answer in structured reasoning)
        for match in re.finditer(r'\*\*([^*]{2,80})\*\*', last_chunk):
            candidate = match.group(1).strip()
            pos = match.start()
            if self._is_valid_answer_candidate(candidate):
                candidates_with_score.append((candidate, 4, pos))

        # Pattern group 5: Quoted entities
        for match in re.finditer(r'"([^"]{2,60})"', last_chunk):
            candidate = match.group(1).strip()
            pos = match.start()
            if self._is_valid_answer_candidate(candidate):
                # Higher score if near conclusion keywords
                context_before = last_chunk[max(0, pos-50):pos].lower()
                score = 5 if any(w in context_before for w in ['answer', 'conclude', 'therefore', 'final', 'output']) else 3
                candidates_with_score.append((candidate, score, pos))

        # Pattern group 6: Last short capitalized line (often the final conclusion)
        lines = last_chunk.strip().split('\n')
        for i, line in enumerate(reversed(lines)):
            line = line.strip().rstrip('.').strip()
            if not line:
                continue
            if 2 < len(line) < 60 and re.match(r'^[A-Z]', line) and line.count(' ') <= 5:
                if self._is_valid_answer_candidate(line):
                    # Position is at the very end
                    candidates_with_score.append((line, 3, len(last_chunk) - i))
                    break

        if not candidates_with_score:
            return ""

        # Rank candidates: prefer (1) higher score, (2) later position (more refined conclusion)
        # Normalize position to 0-1 scale for tiebreaking
        max_pos = max(pos for _, _, pos in candidates_with_score) or 1

        def rank_key(item):
            candidate, score, pos = item
            position_bonus = (pos / max_pos) * 2  # Later = up to +2 bonus
            type_bonus = 1 if self._validate_answer_type(candidate) else -3
            return score + position_bonus + type_bonus

        candidates_with_score.sort(key=rank_key, reverse=True)

        best = candidates_with_score[0][0]
        # Clean up common trailing noise
        best = re.sub(r'\s*\(.*?\)\s*$', '', best).strip()
        best = re.sub(r'\.$', '', best).strip()
        best = best.strip('*').strip()  # Remove bold markers

        if self._is_valid_answer_candidate(best):
            return best
        return ""

    def _is_valid_answer_candidate(self, candidate: str) -> bool:
        """Check if a candidate string looks like a real answer vs garbage/planning text."""
        if not candidate or len(candidate) <= 1 or len(candidate) >= 150:
            return False

        c_lower = candidate.lower().strip()

        # Reject planning/section headers (end with ":")
        if candidate.rstrip().endswith(':'):
            return False

        # Reject common prefixes that indicate non-answers
        bad_starts = (
            'that', 'the answer', 'based on', 'not', 'unclear',
            'i ', 'let me', 'search', 'analyze', 'deconstruct',
            'evaluate', 'identify', 'determine', 'verify', 'check',
            'step ', 'first', 'next', 'then', 'now',
        )
        if c_lower.startswith(bad_starts):
            return False

        # Reject planning/reasoning phrases
        bad_contains = [
            'constraint', 'search for', 'let me', 'i need', 'i should',
            'however', 'unfortunately', 'verify', 'check', 'try',
            'step', 'request', 'clues', 'information', 'evidence',
            'approach', 'strategy', 'requirement', 'the question',
        ]
        if any(w in c_lower for w in bad_contains):
            return False

        return True

    def _validate_answer_type(self, candidate: str) -> bool:
        """Validate candidate answer against expected answer type. Soft filter."""
        answer_type = getattr(self, '_current_answer_type', None)
        if not answer_type or answer_type == "other":
            return True

        c_lower = candidate.lower().strip()

        # Universal rejection: "True" / "False" are NEVER valid for non-yes_no types
        if c_lower in ("true", "false") and answer_type != "yes_no":
            return False

        if answer_type == "yes_no":
            return c_lower in ("yes", "no", "true", "false")

        if answer_type == "person_name":
            if c_lower in ("yes", "no", "true", "false"):
                return False
            if candidate.rstrip().endswith(':'):
                return False
            words = candidate.split()
            return 1 <= len(words) <= 8

        if answer_type in ("movie_title", "show_title"):
            if c_lower in ("yes", "no", "true", "false"):
                return False
            if candidate.rstrip().endswith(':'):
                return False
            return True

        if answer_type == "number":
            return bool(re.search(r'\d', candidate))

        if answer_type == "date":
            return bool(re.search(
                r'\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec',
                c_lower
            ))

        if answer_type == "scientific_name":
            if c_lower in ("yes", "no", "true", "false"):
                return False
            words = candidate.split()
            if len(words) == 2 and words[0][0].isupper() and words[1][0].islower():
                return True
            return len(words) <= 3 and words[0][0].isupper()

        if answer_type in ("organization", "location"):
            if c_lower in ("yes", "no", "true", "false"):
                return False
            if candidate.rstrip().endswith(':'):
                return False
            return True

        return True

    def _teacher_reflect_on_reasoning(self, reasoning: str, question: str) -> str:
        """32B 分析 9B 的 reasoning，提取 9B 自己得出的结论（反思式，合规）。"""
        if not self.teacher or not reasoning:
            return ""

        # Take the last 4000 chars but also include first 500 for context
        if len(reasoning) > 4500:
            reasoning_excerpt = reasoning[:500] + "\n...[middle omitted]...\n" + reasoning[-4000:]
        else:
            reasoning_excerpt = reasoning

        # Build answer type hint for the reflection
        answer_type = getattr(self, '_current_answer_type', None)
        type_hint = ""
        if answer_type and answer_type != "other":
            type_descriptions = {
                "person_name": "a person's name",
                "movie_title": "a movie/show title",
                "organization": "a company/organization name",
                "number": "a number",
                "date": "a date",
                "yes_no": "Yes or No",
                "location": "a place name",
                "scientific_name": "a scientific binomial name",
            }
            type_hint = f"\nExpected answer type: {type_descriptions.get(answer_type, answer_type)}"

        reflect_prompt = (
            "You are analyzing a search agent's internal reasoning to identify what answer "
            "it arrived at. This is a REFLECTION task — extract the agent's OWN conclusion, "
            "not your own answer.\n\n"
            f"Question the agent was trying to answer: {question[:500]}\n"
            f"{type_hint}\n\n"
            f"Agent's reasoning:\n{reasoning_excerpt}\n\n"
            "Instructions:\n"
            "1. Find the LAST/FINAL candidate the agent identified or leaned towards\n"
            "2. If the agent explored multiple candidates, pick the one it settled on LAST\n"
            "3. If the agent expressed uncertainty between candidates, pick the one with more evidence\n"
            "4. Extract ONLY the entity name / number / date / yes-no\n"
            "5. If the agent did NOT reach any conclusion at all, respond with NONE\n\n"
            "Output: just the extracted answer (max 5 words), or NONE."
        )

        try:
            resp = self.teacher.complete(
                [{"role": "user", "content": reflect_prompt}],
                max_tokens=100,
                temperature=0.1,
            )
            if not resp.success or not resp.content:
                return ""

            answer = resp.content.strip()
            if '</think>' in answer:
                answer = answer.split('</think>', 1)[1].strip()
            answer = answer.strip('"').strip("'").strip()

            if answer.upper() == "NONE" or len(answer) > 100:
                return ""

            if self._is_valid_answer_candidate(answer) and self._validate_answer_type(answer):
                logger.info("32B reflected answer from 9B reasoning: %s", answer[:60])
                return answer
        except Exception as exc:
            logger.warning("Teacher reasoning reflection failed: %s", exc)

        return ""

    def _teacher_assisted_9b_answer(self, traj: Trajectory, messages: list[dict]) -> str:
        """32B 总结证据形成 hint，注入 9B 对话让 9B 回答。32B 不直接给出答案。"""
        if not self.teacher:
            return ""

        entries = traj.read_all()
        question = ""
        for e in entries:
            if e.get("role") == "user":
                content = e.get("content", "")
                if isinstance(content, str) and len(content) > 20:
                    question = content[:500]
                    break
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            question = part.get("text", "")[:500]
                            break
                    break

        evidence = ""
        if self.verifier_32b:
            evidence = self.verifier_32b.extract_evidence_summary(entries)

        reasoning_highlights = []
        for e in reversed(entries):
            if e.get("role") == "assistant" and e.get("reasoning_content"):
                rc = e["reasoning_content"]
                for line in rc.split("\n"):
                    if any(kw in line.lower() for kw in [
                        "answer", "candidate", "found", "confirm",
                        "the person is", "the movie is", "the name is",
                        "conclude", "therefore", "must be"
                    ]):
                        reasoning_highlights.append(line.strip()[:150])
                if len(reasoning_highlights) >= 5:
                    break

        # 32B 的任务：总结搜索证据中的关键发现（不给答案）
        summarize_prompt = (
            "You are a research assistant. Summarize the KEY FINDINGS from this search evidence "
            "that are relevant to answering the question below. "
            "List the most important facts found. Do NOT give the answer directly — "
            "just organize the evidence clearly.\n\n"
            f"Question: {question}\n\n"
            f"Search Evidence:\n{evidence[:3000]}\n\n"
            f"Agent's Key Observations:\n" + "\n".join(reasoning_highlights[:5]) + "\n\n"
            "Key findings (bullet points, facts only):"
        )

        try:
            resp = self.teacher.complete(
                [{"role": "user", "content": summarize_prompt}],
                max_tokens=500,
                temperature=0.2,
            )
            if not resp.success or not resp.content:
                return ""

            evidence_summary = resp.content.strip()
            if '</think>' in evidence_summary:
                evidence_summary = evidence_summary.split('</think>', 1)[1].strip()
        except Exception as exc:
            logger.warning("Teacher evidence summary failed: %s", exc)
            return ""

        # 将 32B 的证据总结注入 9B 的对话，让 9B 回答
        enriched_messages = list(messages[-6:])  # 保留最近几条上下文
        enriched_messages.append({
            "role": "user",
            "content": (
                f"[Evidence Summary from search]\n{evidence_summary}\n\n"
                f"Based on the above evidence, answer this question with ONLY the entity "
                f"name/number/date/yes-no. Maximum 5 words. No explanation.\n\n"
                f"Question: {question}\n\nYour answer:"
            ),
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=enriched_messages,
                max_tokens=200,
                temperature=0.3,
                extra_body={"enable_thinking": True},
            )
            if response.usage:
                self.total_tokens_used += response.usage.total_tokens

            msg = response.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, 'reasoning_content', '') or ""

            if '</think>' in content:
                content = content.split('</think>', 1)[1]
            content = re.sub(r'<tool_call>.*', '', content, flags=re.DOTALL)
            content = content.strip()

            answer = self._extract_answer_from_content(content)
            if answer:
                logger.info("9B answered with 32B evidence hint: %s", answer[:60])
                return answer

            # 如果 content 仍为空，从 reasoning 提取
            if reasoning:
                answer = self._extract_answer_from_reasoning(reasoning)
                if answer:
                    logger.info("9B reasoning answer with 32B hint: %s", answer[:60])
                    return answer

        except Exception as exc:
            logger.warning("9B re-answer with hint failed: %s", exc)

        return ""

    def _extract_best_candidate_from_history(self, messages: list[dict], traj: Optional[Trajectory] = None) -> str:
        """Scan assistant messages and reasoning_content for bolded or quoted entities."""
        candidates = []

        # First, scan reasoning_content from trajectory entries (most likely to have info)
        if traj:
            for entry in reversed(traj.read_all()):
                if entry.get("role") != "assistant":
                    continue
                reasoning = entry.get("reasoning_content", "")
                if reasoning:
                    # Look for explicit answer statements in reasoning
                    answer_from_reasoning = self._extract_answer_from_reasoning(reasoning)
                    if answer_from_reasoning:
                        return answer_from_reasoning

                    # Collect entity candidates from reasoning
                    bold = re.findall(r'\*\*(.+?)\*\*', reasoning)
                    candidates.extend(b for b in bold if self._is_valid_answer_candidate(b))
                    quoted = re.findall(r'"([^"]{2,80})"', reasoning)
                    candidates.extend(q for q in quoted if self._is_valid_answer_candidate(q))

                if candidates:
                    break

        # Also scan message content
        if not candidates:
            for msg in reversed(messages):
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content", "")
                if not isinstance(content, str) or not content:
                    continue
                bold = re.findall(r'\*\*(.+?)\*\*', content)
                candidates.extend(b for b in bold if 2 < len(b) < 100)
                quoted = re.findall(r'"([^"]{2,80})"', content)
                candidates.extend(quoted)
                if candidates:
                    break

        # Also scan tool results for entity names (search results often contain the answer)
        if not candidates and traj:
            for entry in reversed(traj.read_all()):
                if entry.get("role") != "tool":
                    continue
                content = entry.get("content", "")
                if not isinstance(content, str):
                    continue
                # Look for title fields in search results
                titles = re.findall(r'"title":\s*"([^"]{3,80})"', content)
                candidates.extend(titles[:3])
                if candidates:
                    break

        if candidates:
            return candidates[0]
        return "[HARNESS] Could not extract answer"

    def _teacher_extract_answer(self, traj: Trajectory) -> str:
        """Deprecated — redirects to compliant method."""
        return ""

    # ===================================================================
    # Tool Dispatch
    # ===================================================================

    def _dispatch_and_record(self, traj: Trajectory, tool_calls, step: int):
        for tc in tool_calls:
            if self.tool_call_count >= config.MAX_TOOL_CALLS_PER_TASK:
                break

            if hasattr(tc, 'function'):
                fn_name = tc.function.name
                tc_id = tc.id
                try:
                    fn_args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}
            elif isinstance(tc, dict):
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                args_raw = fn.get("arguments", "{}")
                if isinstance(args_raw, str):
                    try:
                        fn_args = json.loads(args_raw)
                    except (json.JSONDecodeError, TypeError):
                        fn_args = {}
                else:
                    fn_args = args_raw if isinstance(args_raw, dict) else {}
            else:
                continue

            # Stuck detection for search_text
            if fn_name == "search_text":
                query = fn_args.get("query", "")
                if query in self._seen_queries:
                    self._stuck_counter += 1
                    tool_result = json.dumps({
                        "note": "Already searched this exact query. Try different keywords or answer now."
                    }, ensure_ascii=False)
                else:
                    self._seen_queries.add(query)
                    tool_result = self._execute_tool(fn_name, fn_args)
            else:
                tool_result = self._execute_tool(fn_name, fn_args)

            self.tool_call_count += 1

            # Truncate tool result
            if len(tool_result) > config.MAX_TOOL_RESULT_CHARS:
                tool_result = tool_result[:config.MAX_TOOL_RESULT_CHARS] + "\n...[truncated]"

            traj.write(
                Role.TOOL, tool_result, step_id=step,
                tool_call_id=tc_id,
                extra={"fn_name": fn_name, "fn_args": fn_args},
            )

    def _execute_tool(self, fn_name: str, fn_args: dict) -> str:
        if fn_name not in TOOL_FN_MAP:
            return f"[ERROR] Unknown tool: {fn_name}"

        last_error = None
        for attempt in range(config.TOOL_RETRY_MAX + 1):
            try:
                raw = TOOL_FN_MAP[fn_name](fn_args)
                if isinstance(raw, (dict, list)):
                    return json.dumps(raw, ensure_ascii=False)
                return str(raw)
            except Exception as exc:
                last_error = exc
                if attempt < config.TOOL_RETRY_MAX:
                    logger.warning("Tool %s failed (attempt %d): %s", fn_name, attempt + 1, exc)
                    import time as _t
                    _t.sleep(1 * (attempt + 1))

        return f"[ERROR] Tool '{fn_name}' failed: {last_error}"

    # ===================================================================
    # Verification (32B)
    # ===================================================================

    def _verify_answer(self, traj: Trajectory, answer: str, instruction: str, step: int) -> str:
        answer_type = getattr(self, '_current_answer_type', None)
        # "Simple answer" protection only for yes_no type questions
        is_yes_no_type = answer_type == "yes_no"
        is_simple_answer = (
            answer.lower().strip() in ("yes", "no", "true", "false") and is_yes_no_type
        )

        for attempt in range(config.TEACHER_MAX_VERIFY_ATTEMPTS):
            try:
                evidence = self.verifier_32b.extract_evidence_summary(traj.read_all())
                verification = self.verifier_32b.verify(
                    question=instruction,
                    candidate_answer=answer,
                    search_evidence=evidence,
                )
                if verification.verified:
                    logger.info("32B verified answer (confidence=%s)", verification.confidence)
                    return answer

                logger.info("32B rejected (attempt %d): %s", attempt + 1, verification.reason)

                # Conservative logic: simple answer + low confidence rejection → keep
                # BUT: if high-confidence with real contradictions, DO correct
                if verification.confidence == "high" and verification.constraints_failed:
                    real_contradictions = [
                        c for c in verification.constraints_failed
                        if "cannot verify" not in c.lower()
                        and "unknown" not in c.lower()
                        and "no evidence" not in c.lower()
                    ]
                    if real_contradictions:
                        logger.info("High-confidence rejection with contradictions, proceeding to correct")
                        corrected = self._teacher_correct_answer(
                            instruction, answer, evidence, verification.reason
                        )
                        if corrected and corrected.lower() != answer.lower() and self._is_valid_answer_candidate(corrected) and self._validate_answer_type(corrected):
                            logger.info("Corrected answer (high-confidence): %s → %s", answer[:30], corrected[:30])
                            return corrected
                        return answer

                # Simple answer + low/medium confidence → keep original
                if is_simple_answer and verification.confidence in ("low", "medium"):
                    logger.info("Keeping simple answer despite low-confidence rejection: %s", answer)
                    return answer

                # Only "cannot verify" failures (no contradictions) → keep
                if verification.constraints_failed:
                    real_contradictions = [
                        c for c in verification.constraints_failed
                        if "cannot verify" not in c.lower()
                        and "unknown" not in c.lower()
                        and "no evidence" not in c.lower()
                    ]
                    if not real_contradictions:
                        logger.info("Rejection based only on unverifiable constraints, keeping: %s", answer)
                        return answer

                if attempt >= config.TEACHER_MAX_VERIFY_ATTEMPTS - 1:
                    corrected = self._teacher_correct_answer(
                        instruction, answer, evidence, verification.reason
                    )
                    if corrected and self._is_valid_answer_candidate(corrected) and self._validate_answer_type(corrected):
                        logger.info("Corrected answer: %s → %s", answer[:30], corrected[:30])
                        return corrected
                    logger.info("Correction failed/invalid, keeping original: %s", answer[:30])
                    return answer

                if verification.suggestion:
                    hint_msg = (
                        f"[Verification] Previous answer '{answer}' may be wrong. "
                        f"Issue: {verification.reason}. "
                        f"Try: {verification.suggestion}. Give corrected answer."
                    )
                    traj.write(Role.USER, hint_msg, step_id=step + 1)

                    retry_answer = self._retry_with_hint(traj, step + 1)
                    if retry_answer:
                        answer = retry_answer
                        step += 3

            except Exception as exc:
                logger.warning("32B verifier failed: %s", exc)
                break

        return answer

    def _teacher_correct_answer(self, question: str, wrong_answer: str, evidence: str, reason: str) -> str:
        """32B 验证失败后，让 32B 给出纠正提示，再让 9B 回答（32B 不直接给答案）。"""
        if not self.teacher:
            return ""

        # 32B 生成纠正提示（指出错在哪里、正确方向是什么）
        hint_prompt = (
            "A search agent answered a question but was WRONG. Analyze why and provide "
            "a SPECIFIC HINT about what the correct answer should be, based on the evidence. "
            "Do NOT give the answer directly — just provide the key fact or reasoning "
            "that points to the correct answer.\n\n"
            f"Question: {question[:500]}\n\n"
            f"Wrong answer: {wrong_answer}\n"
            f"Why it's wrong: {reason}\n\n"
            f"Search evidence:\n{evidence[:2000]}\n\n"
            "Hint (one specific fact pointing to the right answer, no answer itself):"
        )

        try:
            resp = self.teacher.complete(
                [{"role": "user", "content": hint_prompt}],
                max_tokens=200,
                temperature=0.2,
            )
            if not resp.success or not resp.content:
                return ""

            hint = resp.content.strip()
            if '</think>' in hint:
                hint = hint.split('</think>', 1)[1].strip()
        except Exception as exc:
            logger.warning("Teacher correction hint failed: %s", exc)
            return ""

        # 将 hint 注入 9B 对话让 9B 辨析（允许保留原答案）
        correction_messages = [
            {"role": "user", "content": (
                f"A verifier flagged your answer '{wrong_answer}' with concern: {reason}\n"
                f"Key hint from analysis: {hint}\n\n"
                f"Question: {question[:300]}\n\n"
                f"Consider this feedback carefully. If you believe your original answer "
                f"'{wrong_answer}' is actually correct despite the concern, output it again. "
                f"Otherwise, provide a corrected answer.\n"
                f"ONLY the entity name/number/date/yes-no. Max 5 words:"
            )}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=correction_messages,
                max_tokens=200,
                temperature=0.3,
                extra_body={"enable_thinking": True},
            )
            if response.usage:
                self.total_tokens_used += response.usage.total_tokens

            msg = response.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, 'reasoning_content', '') or ""

            if '</think>' in content:
                content = content.split('</think>', 1)[1]
            content = content.strip()

            answer = self._extract_answer_from_content(content)
            if answer:
                # 9B 推回：保留原答案
                if answer.lower() == wrong_answer.lower():
                    logger.info("9B reaffirmed original answer: %s", answer[:30])
                    return wrong_answer
                logger.info("9B corrected answer with 32B hint: %s → %s", wrong_answer[:30], answer[:30])
                return answer

            if reasoning:
                answer = self._extract_answer_from_reasoning(reasoning)
                if answer:
                    if answer.lower() == wrong_answer.lower():
                        return wrong_answer
                    logger.info("9B corrected (from reasoning): %s", answer[:30])
                    return answer

        except Exception as exc:
            logger.warning("9B correction with hint failed: %s", exc)

        return ""

    def _extract_entity_from_text(self, text: str) -> str:
        """从文本中提取可能的实体名（通常在括号或引号中）。"""
        import re
        # 括号中的实体: (Some Entity), (ABC)
        parens = re.findall(r'\(([^)]{2,60})\)', text)
        for p in parens:
            if not any(w in p.lower() for w in ['e.g.', 'i.e.', 'note', 'see']):
                return p
        # 引号中的实体
        quoted = re.findall(r"['\"]([^'\"]{2,60})['\"]", text)
        for q in quoted:
            if not any(w in q.lower() for w in ['the ', 'a ', 'is ', 'was ']):
                return q
        # "is X" 或 "called X" 模式
        match = re.search(r'(?:is|called|named|known as)\s+([A-Z][^\s,;.]{1,50}(?:\s+[a-z]+)?)', text)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_entity_from_verbose_content(self, content: str) -> str:
        """Extract a candidate entity from verbose/rejected content that still contains the answer."""
        # Try bolded text
        bold = re.findall(r'\*\*([^*]{2,60})\*\*', content)
        for b in reversed(bold):
            if self._is_valid_answer_candidate(b):
                return b

        # Try "answer is X" pattern
        match = re.search(
            r'(?:answer|result|conclusion)\s*(?:is|:)\s*["\']?([^"\'\n,;.]{2,60})["\']?',
            content, re.IGNORECASE
        )
        if match:
            candidate = match.group(1).strip()
            if self._is_valid_answer_candidate(candidate):
                return candidate

        # Try first capitalized entity after common prefixes
        match = re.search(
            r'(?:Based on|According to|From|The answer is)\s+.*?([A-Z][A-Za-z\s\-\']{1,50}?)(?:\.|,|\s+(?:is|was|has|had|who))',
            content
        )
        if match:
            candidate = match.group(1).strip()
            if self._is_valid_answer_candidate(candidate):
                return candidate

        # Try quoted entities
        quoted = re.findall(r'"([^"]{2,60})"', content)
        for q in reversed(quoted):
            if self._is_valid_answer_candidate(q):
                return q

        return ""

    def _retry_with_hint(self, traj: Trajectory, start_step: int) -> Optional[str]:
        for step in range(start_step + 1, start_step + 4):
            messages = traj.to_messages()
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=config.MAX_TOKENS,
                    temperature=0.7,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    extra_body={"enable_thinking": True},
                )
            except Exception:
                return None

            if response.usage:
                self.total_tokens_used += response.usage.total_tokens

            content, tool_calls, reasoning = self._sanitize_response(response)
            extra = {}
            if tool_calls:
                extra["tool_calls"] = [
                    tc.model_dump() if hasattr(tc, 'model_dump') else tc
                    for tc in tool_calls
                ]
            traj.write(Role.ASSISTANT, content, step_id=step, extra=extra or None)

            if not tool_calls and content.strip():
                if not self._looks_like_thinking(content, step):
                    return self._extract_answer_from_content(content)

            if tool_calls:
                self._dispatch_and_record(traj, tool_calls, step)

        return None

    # ===================================================================
    # Reflection & Memory
    # ===================================================================

    def _reflect_and_store(self, traj: Trajectory, task_id: str, plan: PlanResult, result: dict):
        if not self.memory_store:
            return

        try:
            if self.reflector_32b:
                reflection = self.reflector_32b.reflect(
                    question=result.get("instruction", ""),
                    answer=result["answer"],
                    trajectory=traj.read_all(),
                    task_type=plan.task_type,
                )
                if reflection.success:
                    # Map likely_correct to success for MemoryStore compatibility
                    reflection.success = reflection.likely_correct
                    self.memory_store.store_episode(
                        task_id=task_id,
                        task_type=plan.task_type,
                        reflection_result=reflection,
                    )
                    return

            # Fallback to heuristic reflection
            reflection = self.reflector.reflect(
                trajectory=traj.read_all(),
                result=result,
                plan=plan,
            )
            self.memory_store.store_episode(
                task_id=task_id,
                task_type=plan.task_type,
                reflection_result=reflection,
            )
        except Exception as exc:
            logger.warning("Reflection failed (non-fatal): %s", exc)

    # ===================================================================
    # Helpers
    # ===================================================================

    def _build_user_content(self, instruction: str, image_b64: str = None, image_url: str = None):
        if not image_b64 and not image_url:
            return instruction

        parts = []
        if image_url:
            text = instruction + f"\nImage online URL: {image_url}"
        else:
            text = instruction
        parts.append({"type": "text", "text": text})

        if image_b64:
            if image_b64.startswith("/9j/"):
                mime = "image/jpeg"
            elif image_b64.startswith("iVBOR"):
                mime = "image/png"
            elif image_b64.startswith("R0lGOD"):
                mime = "image/gif"
            elif image_b64.startswith("UklGR"):
                mime = "image/webp"
            else:
                mime = "image/jpeg"
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{image_b64}"}
            })

        return parts

    def _parse_text_tool_calls(self, content: str) -> list[dict]:
        """Parse <tool_call> tags from Qwen native format."""
        tool_calls = []

        # Format 1: <tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>
        json_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        json_matches = re.findall(json_pattern, content, re.DOTALL)
        for match in json_matches[:config.MAX_TOOLS_PER_STEP]:
            try:
                data = json.loads(match)
                fn_name = data.get("name", "")
                fn_args = data.get("arguments", {})
                if isinstance(fn_args, str):
                    fn_args = json.loads(fn_args)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {"name": fn_name, "arguments": json.dumps(fn_args, ensure_ascii=False)},
                })
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Format 2: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
        pattern = r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>'
        matches = re.findall(pattern, content, re.DOTALL)
        for fn_name, params_block in matches[:config.MAX_TOOLS_PER_STEP]:
            args = {}
            param_pattern = r'<parameter=(\w+)>\s*(.*?)\s*</parameter>'
            param_matches = re.findall(param_pattern, params_block, re.DOTALL)
            for param_name, param_value in param_matches:
                param_value = param_value.strip()
                if param_value.lower() == 'true':
                    args[param_name] = True
                elif param_value.lower() == 'false':
                    args[param_name] = False
                else:
                    try:
                        args[param_name] = int(param_value)
                    except ValueError:
                        args[param_name] = param_value

            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "function": {"name": fn_name, "arguments": json.dumps(args, ensure_ascii=False)},
            })

        return tool_calls

    def _extract_answer_from_content(self, content: str) -> str:
        if not content:
            return ""

        # Remove think/tool blocks (handle unclosed tags too)
        if '</think>' in content:
            content = content.split('</think>', 1)[1]
        content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
        content = re.sub(r'<tool_call>.*', '', content, flags=re.DOTALL)
        content = re.sub(r'<function=\w+>.*', '', content, flags=re.DOTALL)
        content = content.strip()

        if not content:
            return ""

        # Reject confused/non-answer responses
        non_answer_patterns = [
            r'^(?:I notice|I don\'t see|I cannot|I\'m unable|Unfortunately)',
            r'^(?:Based on|Let me|I need to|I should|I\'ll|I was)',
            r'^(?:The context|The summary|The search|No (?:results|information|answer))',
            r'^(?:After|However|To (?:answer|find|determine)|My search)',
            r'^(?:Verification|I have (?:not|been)|I couldn\'t|I did not)',
            r'^(?:This is|Here (?:is|are)|According to my)',
            r'(?:mention|all (?:the )?criteria|they would have|this means they|might not be)',
            r'(?:search for|look for|try searching|find the)',
        ]
        for pat in non_answer_patterns:
            if re.match(pat, content, re.IGNORECASE):
                return ""

        # Reject if content is too long (likely reasoning dump)
        if len(content) > 150:
            # Try to extract a concise answer from it
            answer_patterns = [
                r'(?:final\s+answer|the\s+answer\s+is|answer:)\s*[:\-]?\s*["\']?(.+?)["\']?(?:\.|$|\n)',
                r'\*\*(.+?)\*\*',
            ]
            for pat in answer_patterns:
                match = re.search(pat, content, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    if 1 < len(candidate) < 150:
                        return candidate
            # If first line is short enough, use it
            first_line = content.split('\n')[0].strip()
            if 1 < len(first_line) <= 80 and not any(
                w in first_line.lower() for w in ['search', 'let me', 'i need', 'based on', 'however', 'unfortunately']
            ):
                return first_line
            return ""

        # Content is <= 150 chars — do a final sanity check
        # Reject if it contains typical reasoning markers
        reasoning_markers = [
            'constraint', 'verification', 'search for', 'try searching',
            'i found', 'let me check', 'my approach', 'reconsider',
        ]
        if any(m in content.lower() for m in reasoning_markers):
            # Still try to extract entity if there's one mentioned
            match = re.search(r'\*\*(.+?)\*\*', content)
            if match and 1 < len(match.group(1)) < 100:
                extracted = match.group(1)
                if self._is_valid_answer_candidate(extracted) and self._validate_answer_type(extracted):
                    return extracted
            return ""

        # Final validity + type check before returning
        cleaned = content.strip()
        if not self._is_valid_answer_candidate(cleaned):
            return ""
        if not self._validate_answer_type(cleaned):
            return ""
        return cleaned

    def _looks_like_thinking(self, content: str, step: int) -> bool:
        clean = content.strip()
        if '</think>' in clean:
            clean = clean.split('</think>', 1)[1].strip()
        if not clean:
            return True
        # In early steps, be very strict about what counts as an answer
        if step <= 5:
            thinking_patterns = [
                r'(?i)^(let me|I\'ll|I need to|I should|let\'s)',
                r'(?i)^(first,?\s+I|to answer this)',
                r'(?i)(mention|search for|find|look for|try)',
                r'^\d+\.\s',
            ]
            for pat in thinking_patterns:
                if re.search(pat, clean):
                    return True
            if len(clean) > 200:
                return True
            # If content looks like a search plan or partial reasoning
            if any(w in clean.lower() for w in [
                'criteria', 'constraint', 'all the', 'both', 'means they',
                'would have', 'they were', 'this might', 'could not'
            ]):
                return True
        return False

    def _reasoning_has_conclusion(self, reasoning: str) -> bool:
        """Check if reasoning contains strong signals that a conclusion was reached."""
        last_part = reasoning[-2000:] if len(reasoning) > 2000 else reasoning
        conclusion_signals = [
            r'(?:my\s+)?(?:final\s+)?answer\s*(?:is|would be|should be|:)',
            r'(?:therefore|thus|in conclusion|hence)\s+(?:the\s+)?(?:answer|it)',
            r'(?:I\'ll go with|I will output|let me answer)\s',
            r'(?:the\s+(?:answer|person|company|team|movie|name)\s+(?:is|must be))\s+[A-Z]',
            r'(?:I\s+(?:believe|conclude|determine)\s+(?:it|the answer|this)\s+(?:is|to be))',
        ]
        for pat in conclusion_signals:
            if re.search(pat, last_part, re.IGNORECASE):
                return True
        return False

    # ===================================================================
    # Mid-Task 32B Reflection (充分利用 32B)
    # ===================================================================

    def _should_mid_reflect(self, step: int, max_steps: int) -> bool:
        """Determine if we should ask 32B for a mid-task search direction check."""
        # Trigger at ~40% and ~70% of budget, but only if stuck or repeating
        progress = step / max_steps
        if progress < 0.35:
            return False
        # Only trigger at specific points (not every step)
        midpoint = max_steps * 4 // 10
        late_point = max_steps * 7 // 10
        if step != midpoint and step != late_point:
            return False
        # Don't reflect if making good progress (no stuck)
        if self._stuck_counter == 0 and step == midpoint:
            return False
        return True

    def _mid_task_redirect(self, traj: Trajectory, question: str, step: int, max_steps: int) -> str:
        """32B analyzes search progress and suggests a new search direction."""
        if not self.reflector_32b:
            return ""

        entries = traj.read_all()

        # Collect search queries used so far
        queries_used = []
        findings = []
        for entry in entries:
            if entry.get("role") == "tool":
                extra = entry.get("extra", {})
                if extra.get("fn_name") == "search_text":
                    queries_used.append(extra.get("fn_args", {}).get("query", ""))
                content = entry.get("content", "")
                if isinstance(content, str) and "title" in content:
                    # Extract key findings from search results
                    import json as _json
                    try:
                        results = _json.loads(content)
                        if isinstance(results, list):
                            for r in results[:2]:
                                if isinstance(r, dict):
                                    findings.append(f"- {r.get('title', '')}: {r.get('snippet', '')[:100]}")
                    except (ValueError, TypeError):
                        pass

        if not queries_used:
            return ""

        queries_summary = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(queries_used[-8:]))
        findings_summary = "\n".join(findings[-5:]) if findings else "(no useful findings yet)"

        redirect_prompt = (
            "A search agent is trying to answer a question but may be going in the wrong direction. "
            "Analyze its search history and suggest a BETTER search strategy. "
            "Do NOT provide the answer — only suggest what to search for next.\n\n"
            f"Question: {question[:500]}\n\n"
            f"Searches done so far:\n{queries_summary}\n\n"
            f"Key findings so far:\n{findings_summary}\n\n"
            f"Budget remaining: {max_steps - step} steps out of {max_steps}\n\n"
            "Based on the question structure, what search approach would be MORE EFFECTIVE? "
            "Focus on:\n"
            "1. What is the MOST DISTINCTIVE constraint that hasn't been searched yet?\n"
            "2. Is there a specific database or source to target?\n"
            "3. Should the agent decode a euphemistic description first?\n"
            "4. Suggest 2-3 specific search queries to try next.\n\n"
            "Keep your response under 150 words. Actionable suggestions only."
        )

        try:
            resp = self.teacher.complete(
                [{"role": "user", "content": redirect_prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            if not resp.success or not resp.content:
                return ""

            suggestion = resp.content.strip()
            if '</think>' in suggestion:
                suggestion = suggestion.split('</think>', 1)[1].strip()

            if len(suggestion) < 20:
                return ""

            logger.info("32B mid-task redirect at step %d: %s", step, suggestion[:80])
            return (
                f"[Search Direction Guidance] Your current search approach may not be optimal. "
                f"Consider this alternative strategy:\n{suggestion}\n\n"
                f"Try the suggested searches. Remember: output ONLY the final answer entity when done."
            )
        except Exception as exc:
            logger.warning("Mid-task redirect failed: %s", exc)
            return ""
