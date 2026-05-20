"""
PromptBuilder — 动态组装 system prompt (BrowseComp optimized)

根据任务类型、历史经验、策略库动态生成 system prompt。
英文 prompt 减少翻译歧义，策略针对 BrowseComp 多约束实体识别题型。
"""
from __future__ import annotations

from typing import Optional

from modules.planner import PlanResult
from modules.search_skill import build_search_skill_section, get_search_priority_hint
from modules.search_lessons import build_lessons_section
from modules.question_guidance import get_question_guidance


BASE_SYSTEM_PROMPT = """You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers. No matter how complex the query, you will NOT give up until you find the answer.

## Principles
1. **Persistent Search**: Engage in many search rounds, delving deeply. Do NOT stop after 1-2 searches.
2. **Repeated Verification**: Cross-check and validate information before giving your final answer.
3. **Attention to Detail**: Ensure all data is current, relevant, and from credible sources.

## Phase 1 - Decomposition (in your thinking)
- List ALL constraints/conditions from the question
- Identify the 1-2 MOST DISTINCTIVE clues (specific names, rare number+event combos, nicknames, exact quotes, rare awards)
- DECODE any euphemistic descriptions BEFORE searching:
  - "born out of discord between two parties" = club formed from a split/schism
  - "global non-profit journalistic platform for academics" = The Conversation
  - "fitting nickname" = search the famous moniker in quotes
  - Zodiac element clues = check actual birthday dates
- Plan your first search using the most unique clue

## Phase 2 - Search (use multiple rounds)
- Use the most distinctive constraint first (4-6 words)
- Use EXACT PHRASE matching with quotes for specific clues: e.g. "1.7 million" employed, "Mr. Le Mans" racing
- If a search returns nothing useful, PIVOT completely — try different keywords, different language, different angle
- Do NOT repeat the same search. Do NOT put all constraints in one query.
- When snippet is insufficient, use fetch=true to read full content
- For RELATIONSHIP CHAINS (A who directed B sung by C who won D): resolve ONE HOP at a time, never all at once
- For RARE AWARDS: search the recipient list/winners list first

## Phase 3 - Verification
- Once you find a candidate answer, verify it against EVERY constraint in the question
- Search: "candidate_name + constraint_to_verify"
- If ANY constraint fails → reject candidate → return to Phase 2 with new angle
- Only output final answer when ALL constraints are confirmed
- Pay attention to EXACT dates (month matters!), FULL names (include middle names), and SPECIFIC categories (not just "Grammy" but "Grammy for Best Traditional Pop Vocal")

## Image Questions
- Use search_image to identify the person/object/brand
- If uncertain: describe visible text (jersey numbers, logos) and search those
- Always verify identification before using it for further reasoning

## Answer Format — CRITICAL
- Output ONLY the final answer entity. No headers, no labels, no reasoning.
- NEVER output section headers like "Analyze the Request:" or "Step 1:" as your answer.
- Your answer must be a short entity: a name, number, date, or yes/no.
- GOOD answers: "Marie Curie", "42,000", "Yes", "March 2024", "Canis lupus", "The Great Gatsby", "Tokyo Tower"
- BAD answers: "Analyze the Request:", "Based on my research...", "The answer is Marie Curie", "all the criteria", "mention a 2009 war film"
- If you have concluded the answer in your thinking, just output the bare answer.
- Answers CAN start with "The", "A", or "An" if that's part of the entity name."""


class PromptBuilder:
    _ANSWER_TYPE_DESC = {
        "person_name": "a person's full name (e.g., 'Albert Einstein')",
        "movie_title": "a movie or show title (e.g., 'Inception')",
        "organization": "a company or organization name (e.g., 'SpaceX')",
        "number": "a number or quantity (e.g., '42,000')",
        "date": "a date (e.g., 'March 14, 2023')",
        "yes_no": "Yes or No (exactly one word)",
        "location": "a place name (e.g., 'Paris, France')",
        "scientific_name": "a scientific binomial name (e.g., 'Canis lupus')",
    }

    def build(
        self,
        plan: PlanResult,
        memories: Optional[list] = None,
        strategies: Optional[list[str]] = None,
        strategy_memory_block: Optional[str] = None,
        plan_32b_section: Optional[str] = None,
        expected_answer_type: Optional[str] = None,
        question: Optional[str] = None,
        has_image: bool = False,
    ) -> str:
        parts = [BASE_SYSTEM_PROMPT]

        # Question-specific guidance (highest priority — pre-computed optimal path)
        if question:
            guidance = get_question_guidance(question)
            if guidance:
                parts.append(
                    "\n\n## Advisory Search Guidance (from analysis assistant)\n"
                    "The following search analysis was pre-computed. "
                    "Use it as a starting point but adapt based on what you find.\n\n"
                )
                parts.append(guidance)

        # Search skill (pattern-matched strategies from benchmark analysis)
        if question:
            if not guidance:  # Only add generic skills if no specific guidance
                skill_section = build_search_skill_section(question, has_image)
                if skill_section:
                    parts.append(f"\n\n{skill_section}")
            priority_hint = get_search_priority_hint(question, has_image)
            if priority_hint:
                parts.append(f"\n\n## FIRST SEARCH PRIORITY\n{priority_hint}")

            # Search lessons (domain-specific strategies and decode hints)
            if not guidance:  # Skip lessons if guidance already covers it
                lessons_section = build_lessons_section(question, has_image)
                if lessons_section:
                    parts.append(f"\n\n{lessons_section}")

        # Task-specific strategy
        parts.append(f"\n\n## Current Task Strategy\n{plan.strategy_hint}")

        # Strategy memory (from StrategyMemory module — seed + dynamic lessons)
        if strategy_memory_block:
            parts.append(f"\n\n{strategy_memory_block}")

        # Legacy memory entries (from MemoryStore)
        if memories:
            memory_section = self._format_memories(memories)
            if memory_section:
                parts.append(f"\n\n## Historical Experience\n{memory_section}")

        # Verified strategies from strategy store
        if strategies:
            strat_text = "\n".join(f"- {s}" for s in strategies[:5])
            parts.append(f"\n\n## Verified Effective Strategies\n{strat_text}")

        # 32B advisory search plan (with explicit framing)
        if plan_32b_section:
            parts.append(
                "\n\n## Advisory Search Guidance (from analysis assistant)\n"
                "The following search plan was pre-computed by an analysis assistant. "
                "It is ADVISORY ONLY — use it as a starting point but adapt based on "
                "what you find. Do NOT echo or restate these section headers in your answer."
            )
            parts.append(plan_32b_section)

        # Expected answer type constraint (must be last for emphasis)
        if expected_answer_type and expected_answer_type != "other":
            desc = self._ANSWER_TYPE_DESC.get(expected_answer_type, expected_answer_type)
            parts.append(
                f"\n\n## REQUIRED Answer Type: {expected_answer_type}\n"
                f"Your final answer MUST be {desc}.\n"
                f"Do NOT output anything else — no explanations, no headers, no reasoning steps."
            )

        return "".join(parts)

    def _format_memories(self, memories: list, max_tokens: int = 1500) -> str:
        lines = []
        token_count = 0

        for mem in memories:
            success = getattr(mem, 'success', True)
            lessons = getattr(mem, 'lessons', [])
            instruction = getattr(mem, 'instruction', '')

            prefix = "[Success]" if success else "[Failed]"
            entry = f"{prefix} Similar: \"{instruction[:80]}...\"\n"
            if lessons:
                entry += "  Lessons: " + "; ".join(lessons[:2]) + "\n"

            est_tokens = len(entry) // 2
            if token_count + est_tokens > max_tokens:
                break
            lines.append(entry)
            token_count += est_tokens

        return "".join(lines)
