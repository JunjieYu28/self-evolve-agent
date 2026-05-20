"""
Reflector32B — 32B 反思模块
==============================

任务完成后用 32B 分析完整轨迹，提取高质量经验教训。
替代原有的启发式反思（modules/reflection.py），产出更精准的 lessons。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from modules.teacher_client import TeacherClient

logger = logging.getLogger("harness.reflector32b")

REFLECTOR_PROMPT = """You are an expert at analyzing search agent trajectories. Given a question and the agent's search trajectory, assess the quality of the answer and extract lessons.

## Instructions
1. Assess whether the agent likely found the correct answer (based on evidence quality)
2. Identify what went well and what went wrong in the search process
3. Extract reusable lessons for similar future questions
4. Suggest strategy improvements

## Output Format (JSON only)
```json
{
  "confidence": "high/medium/low",
  "likely_correct": true/false,
  "what_worked": ["effective strategy 1", ...],
  "what_failed": ["ineffective approach 1", ...],
  "root_cause": "main reason if answer is likely wrong, null if likely correct",
  "lessons": [
    "Reusable lesson 1 (actionable, specific)",
    "Reusable lesson 2"
  ],
  "strategy_suggestion": "One sentence: how to improve for this type of question"
}
```

## Guidelines
- Lessons should be SPECIFIC and ACTIONABLE (not generic like "search more carefully")
- Good lesson: "For questions about company financials, search the SEC EDGAR database directly"
- Bad lesson: "Be more thorough in searching"
- Focus on the SEARCH STRATEGY, not the answer content
- Assess confidence based on: how many constraints were verified, source credibility, consistency

IMPORTANT: Output ONLY the JSON. No other text."""


@dataclass
class Reflection32BResult:
    confidence: str = "low"
    likely_correct: bool = False
    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)
    root_cause: Optional[str] = None
    lessons: list[str] = field(default_factory=list)
    strategy_suggestion: str = ""
    success: bool = True
    source: str = "32b_reflector"


class Reflector32B:
    def __init__(self, teacher: TeacherClient):
        self.teacher = teacher

    def reflect(
        self,
        question: str,
        answer: str,
        trajectory: list[dict],
        task_type: str = "",
    ) -> Reflection32BResult:
        """
        Reflect on a completed task trajectory.

        Args:
            question: Original question
            answer: Final answer produced
            trajectory: Full trajectory entries
            task_type: Task classification from planner
        """
        traj_summary = self._summarize_trajectory(trajectory)

        user_msg = (
            f"## Question\n{question}\n\n"
            f"## Final Answer\n{answer}\n\n"
            f"## Task Type\n{task_type or 'unknown'}\n\n"
            f"## Search Trajectory\n{traj_summary}"
        )

        resp = self.teacher.complete(
            messages=[
                {"role": "system", "content": REFLECTOR_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )

        if not resp.success:
            logger.warning("Reflector32B call failed: %s", resp.error)
            return Reflection32BResult(success=False)

        parsed = self.teacher.parse_json_response(resp)
        if not parsed:
            logger.warning("Reflector32B response not parseable")
            return Reflection32BResult(success=False)

        return Reflection32BResult(
            confidence=parsed.get("confidence", "low"),
            likely_correct=parsed.get("likely_correct", False),
            what_worked=parsed.get("what_worked", []),
            what_failed=parsed.get("what_failed", []),
            root_cause=parsed.get("root_cause"),
            lessons=parsed.get("lessons", []),
            strategy_suggestion=parsed.get("strategy_suggestion", ""),
            success=True,
        )

    def _summarize_trajectory(self, trajectory: list[dict], max_chars: int = 4000) -> str:
        """Create a concise summary of the trajectory for reflection."""
        parts = []
        total = 0

        for entry in trajectory:
            role = entry.get("role", "")
            content = entry.get("content", "")

            if role == "system":
                continue

            if role == "assistant":
                # Show reasoning briefly
                if len(content) > 300:
                    content = content[:300] + "..."
                extra = entry.get("extra", {})
                tool_calls = extra.get("tool_calls", [])
                if tool_calls:
                    tc_summary = ", ".join(
                        f"{tc.get('function', {}).get('name', '?')}({tc.get('function', {}).get('arguments', '')[:50]})"
                        for tc in tool_calls[:3]
                    )
                    chunk = f"[Assistant] Calls: {tc_summary}"
                    if content:
                        chunk += f"\n  Text: {content[:150]}"
                else:
                    chunk = f"[Assistant] {content}"

            elif role == "tool":
                extra = entry.get("extra", {})
                fn_name = extra.get("fn_name", "tool")
                fn_args = extra.get("fn_args", {})
                query = fn_args.get("query", fn_args.get("url", ""))[:60]
                result_preview = content[:200] if content else "(empty)"
                chunk = f"[Tool:{fn_name}] q={query}\n  Result: {result_preview}"

            elif role == "user":
                chunk = f"[User] {content[:100]}"

            else:
                continue

            chunk += "\n"
            if total + len(chunk) > max_chars:
                parts.append("... (trajectory truncated)")
                break
            parts.append(chunk)
            total += len(chunk)

        return "\n".join(parts)
