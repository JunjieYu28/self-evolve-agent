"""
Planner32B — 32B 任务前规划器
==============================

用 32B 分析问题约束、制定搜索策略，注入 9B 的 system prompt。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from modules.teacher_client import TeacherClient

logger = logging.getLogger("harness.planner32b")

PLANNER_PROMPT = """You are an expert search strategist. Given a complex multi-constraint question, decompose it and plan an effective search strategy.

## Instructions
1. Identify ALL constraints/conditions in the question
2. Find the 1-2 MOST DISTINCTIVE clues (rare names, specific numbers, unusual combinations)
3. Plan 3-5 search queries in order of priority (most distinctive first)
4. Estimate difficulty
5. Determine the EXPECTED ANSWER TYPE — what form the final answer should take

## Output Format (JSON only)
```json
{
  "constraints": ["constraint 1", "constraint 2", ...],
  "most_distinctive_clue": "the rarest/most searchable piece of info",
  "search_plan": [
    "first search query (most distinctive)",
    "second search query (alternative angle)",
    "third search query (verification)"
  ],
  "difficulty": "easy/medium/hard",
  "tips": "one-line strategy advice for this specific question",
  "expected_answer_type": "person_name/movie_title/organization/number/date/yes_no/location/scientific_name/other"
}
```

## Guidelines for search queries
- Keep queries SHORT (4-6 words)
- Use EXACT PHRASES in quotes for specific names/numbers: e.g. "1.7 million" employed
- Start with the most unique constraint (not generic ones like "born in 1940s")
- Include at least one verification query that checks a different constraint
- For image questions, suggest what to look for in the image

## Answer type classification
- person_name: "Who is...", "name of the person", "full birth name", "first/last name"
- movie_title: "title of the movie/show/film", "name of the anime/series"
- organization: "which company", "name of the organization/team"
- number: "how many", "what percentage", "closing price", "population"
- date: "when did", "month and year", "what date"
- yes_no: "Is the X higher/lower?", "Did X do Y?", "True or false"
- location: "where", "which city/country"
- scientific_name: "scientific name", "genus and species", "binomial name"
- other: technology names, concepts, or anything else

IMPORTANT: Output ONLY the JSON. No other text."""


@dataclass
class Plan32BResult:
    constraints: list[str] = field(default_factory=list)
    most_distinctive_clue: str = ""
    search_plan: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    tips: str = ""
    expected_answer_type: str = "other"
    success: bool = True


class Planner32B:
    def __init__(self, teacher: TeacherClient):
        self.teacher = teacher

    def plan(
        self,
        question: str,
        has_image: bool = False,
        history_context: Optional[str] = None,
    ) -> Plan32BResult:
        """
        Generate a search plan for the given question.

        Args:
            question: The full question text
            has_image: Whether this question includes an image
            history_context: Optional context from memory (similar past successes)
        """
        user_parts = [f"## Question\n{question}"]

        if has_image:
            user_parts.append("\n[Note: This question includes an image. Plan for image-based search too.]")

        if history_context:
            user_parts.append(f"\n## Similar Past Questions (for reference)\n{history_context}")

        resp = self.teacher.complete(
            messages=[
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            temperature=0.3,
        )

        if not resp.success:
            logger.warning("Planner32B call failed: %s", resp.error)
            return Plan32BResult(success=False)

        parsed = self.teacher.parse_json_response(resp)
        if not parsed:
            logger.warning("Planner32B response not parseable")
            return Plan32BResult(success=False)

        return Plan32BResult(
            constraints=parsed.get("constraints", []),
            most_distinctive_clue=parsed.get("most_distinctive_clue", ""),
            search_plan=parsed.get("search_plan", []),
            difficulty=parsed.get("difficulty", "medium"),
            tips=parsed.get("tips", ""),
            expected_answer_type=parsed.get("expected_answer_type", "other"),
            success=True,
        )

    def format_as_prompt_section(self, plan: Plan32BResult) -> str:
        """Format the 32B plan as a prompt section to inject into 9B's system prompt."""
        if not plan.success or not plan.search_plan:
            return ""

        parts = ["\n\n## Pre-computed Search Plan (from analysis)"]

        if plan.constraints:
            parts.append(f"Constraints identified: {len(plan.constraints)}")
            for i, c in enumerate(plan.constraints[:8], 1):
                parts.append(f"  {i}. {c}")

        if plan.most_distinctive_clue:
            parts.append(f"\nMost distinctive clue: {plan.most_distinctive_clue}")

        if plan.search_plan:
            parts.append("\nRecommended search sequence:")
            for i, q in enumerate(plan.search_plan[:5], 1):
                parts.append(f"  {i}. {q}")

        if plan.tips:
            parts.append(f"\nStrategy tip: {plan.tips}")

        parts.append(f"\nEstimated difficulty: {plan.difficulty}")

        if plan.expected_answer_type and plan.expected_answer_type != "other":
            parts.append(f"\nExpected answer format: {plan.expected_answer_type}")

        return "\n".join(parts)
