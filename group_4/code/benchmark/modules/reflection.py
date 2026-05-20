"""
ReflectionModule — 反思模块 (Self-Assessment, No Ground Truth)
==============================================================

V2: 多维度分析 + 多样化 lesson 生成 + 准确的 success 判定
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from modules.planner import PlanResult

logger = logging.getLogger("harness.reflection")


@dataclass
class ReflectionResult:
    success: bool
    confidence: str  # "high" | "medium" | "low"
    root_cause: Optional[str] = None
    lessons: list[str] = field(default_factory=list)
    strategy_update: Optional[str] = None
    source: str = ""
    steps_taken: int = 0


class ReflectionModule:
    """Multi-dimensional self-assessment reflection."""

    def reflect(
        self,
        trajectory: list[dict],
        result: dict,
        plan: Optional[PlanResult] = None,
    ) -> ReflectionResult:
        answer = result.get("answer", "")
        steps = result.get("steps", 0)

        search_queries = self._extract_search_queries(trajectory)
        tool_results = self._extract_tool_results(trajectory)

        # Accurate success detection
        success = self._assess_success(answer, steps)
        confidence = "medium"
        root_cause = None
        lessons = []
        source = ""

        # Multi-dimensional analysis
        analysis = self._analyze_trajectory(search_queries, tool_results, answer, steps, plan)

        if not success:
            confidence = "low"
            # Determine primary failure mode
            if analysis["empty_answer"]:
                root_cause = "Failed to produce a final answer within step budget"
                source = "no_answer"
            elif analysis["stuck_count"] >= 3:
                root_cause = f"Got stuck repeating similar searches ({analysis['stuck_count']} repetitions)"
                source = "stuck_loop"
            elif analysis["all_errors"]:
                root_cause = "Tool calls returned errors, blocking progress"
                source = "tool_errors"
            elif steps >= (plan.estimated_steps * 1.5 if plan else 20):
                root_cause = "Exceeded expected step budget without converging"
                source = "timeout"
            else:
                root_cause = "Answer appears uncertain or too verbose"
                source = "low_quality_answer"
        else:
            if steps <= 4 and search_queries:
                confidence = "high"
                source = "quick_success"
            elif steps <= (plan.estimated_steps if plan else 10):
                confidence = "high"
                source = "efficient_success"
            else:
                confidence = "medium"
                source = "slow_success"

        # Generate diverse lessons based on specific trajectory patterns
        lessons = self._generate_lessons(
            analysis, search_queries, answer, success, source, plan
        )

        return ReflectionResult(
            success=success,
            confidence=confidence,
            root_cause=root_cause,
            lessons=lessons,
            strategy_update=None,
            source=source,
            steps_taken=steps,
        )

    def _assess_success(self, answer: str, steps: int) -> bool:
        if not answer or not answer.strip():
            return False
        if "[HARNESS]" in answer or "[ERROR]" in answer:
            return False
        lower = answer.lower().strip()
        if lower.startswith(("based on", "i ", "unfortunately", "i'm sorry", "i cannot")):
            return False
        if len(answer) > 300:
            return False
        return True

    def _analyze_trajectory(
        self, queries: list[str], tool_results: list[str],
        answer: str, steps: int, plan: Optional[PlanResult]
    ) -> dict:
        analysis = {
            "empty_answer": not answer or not answer.strip(),
            "total_searches": len(queries),
            "long_queries": sum(1 for q in queries if len(q.split()) > 8),
            "short_queries": sum(1 for q in queries if len(q.split()) <= 3),
            "stuck_count": self._count_stuck_episodes(queries),
            "all_errors": all("[ERROR]" in r or "[proxy-error]" in r for r in tool_results) if tool_results else False,
            "error_rate": sum(1 for r in tool_results if "[ERROR]" in r or "[proxy-error]" in r) / max(len(tool_results), 1),
            "unique_query_ratio": len(set(queries)) / max(len(queries), 1),
            "used_fetch": any("fetch" in q.lower() for q in queries),
            "query_diversity": self._query_diversity_score(queries),
        }
        return analysis

    def _generate_lessons(
        self, analysis: dict, queries: list[str],
        answer: str, success: bool, source: str,
        plan: Optional[PlanResult]
    ) -> list[str]:
        lessons = []
        task_type = plan.task_type if plan else "general"

        if not success:
            # Lesson from the specific failure mode
            if source == "no_answer":
                if analysis["stuck_count"] >= 2:
                    lessons.append(
                        f"For {task_type}: when stuck after 2 similar searches, immediately pivot to a completely different constraint or search the answer domain directly."
                    )
                elif analysis["long_queries"] > 2:
                    lessons.append(
                        f"For {task_type}: keep search queries to 4-6 words. Long queries ({analysis['long_queries']} found) reduce result relevance."
                    )
                else:
                    lessons.append(
                        f"For {task_type}: ensure every step makes progress toward an answer. If approaching step limit, commit to best candidate."
                    )

            elif source == "stuck_loop":
                if queries:
                    stuck_pattern = self._identify_stuck_pattern(queries)
                    if stuck_pattern:
                        lessons.append(
                            f"Avoid repeating searches around '{stuck_pattern}'. After 2 failures with similar keywords, try: (1) a different language, (2) searching from answer-side, or (3) a completely different constraint."
                        )

            elif source == "low_quality_answer":
                lessons.append(
                    "Output only the concise entity/name/number. Verbose answers indicate low confidence — verify more before answering."
                )

        else:
            # Learn from success patterns
            if source == "quick_success" and queries:
                first_q = queries[0][:50]
                lessons.append(
                    f"Effective first-query pattern for {task_type}: '{first_q}' — starting with the most distinctive constraint works."
                )
            elif source == "efficient_success" and analysis["query_diversity"] > 0.8:
                lessons.append(
                    f"For {task_type}: diverse search angles (ratio={analysis['query_diversity']:.1f}) led to success. Each search should explore a different facet."
                )

        # Cap at 2 lessons per task to keep memory diverse
        return lessons[:2]

    def _extract_search_queries(self, trajectory: list[dict]) -> list[str]:
        queries = []
        for entry in trajectory:
            if not isinstance(entry, dict):
                continue
            extra = entry.get("extra", {})
            fn_name = extra.get("fn_name", "") or entry.get("fn_name", "")
            fn_args = extra.get("fn_args") or entry.get("fn_args")
            if fn_name == "search_text" and fn_args:
                q = fn_args.get("query", "")
                if q:
                    queries.append(q)
        return queries

    def _extract_tool_results(self, trajectory: list[dict]) -> list[str]:
        results = []
        for entry in trajectory:
            if not isinstance(entry, dict):
                continue
            if entry.get("role") == "tool":
                content = entry.get("content", "")
                if isinstance(content, str):
                    results.append(content[:200])
        return results

    def _count_stuck_episodes(self, queries: list[str]) -> int:
        if len(queries) < 2:
            return 0
        stuck = 0
        for i in range(1, len(queries)):
            words_prev = set(queries[i-1].lower().split())
            words_curr = set(queries[i].lower().split())
            if not words_prev or not words_curr:
                continue
            overlap = len(words_prev & words_curr) / min(len(words_prev), len(words_curr))
            if overlap > 0.6:
                stuck += 1
        return stuck

    def _query_diversity_score(self, queries: list[str]) -> float:
        if len(queries) <= 1:
            return 1.0
        all_words = []
        for q in queries:
            all_words.extend(q.lower().split())
        word_counts = Counter(all_words)
        unique_ratio = len(word_counts) / max(len(all_words), 1)
        return round(unique_ratio, 2)

    def _identify_stuck_pattern(self, queries: list[str]) -> str:
        if len(queries) < 3:
            return ""
        word_counts = Counter()
        for q in queries:
            for w in q.lower().split():
                if len(w) > 3:
                    word_counts[w] += 1
        common = word_counts.most_common(3)
        if common and common[0][1] >= 3:
            return " ".join(w for w, c in common if c >= 3)
        return ""
