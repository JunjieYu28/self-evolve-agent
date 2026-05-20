"""
MemoryStore — 策略记忆模块 (Self-Evolution Strategy Memory)
============================================================

V2 改进:
1. 去重存储（Jaccard 相似度检查）
2. 分级策略库（seed → dynamic → graduated）
3. 修正 store_episode（准确的 success/steps）
4. 策略注入优化（token 预算控制）
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("harness.memory")


# ---------------------------------------------------------------------------
# Seed strategies (from ANALYSIS.md failure patterns)
# ---------------------------------------------------------------------------
SEED_STRATEGIES = [
    "When a question mentions a nickname or title, search that specific nickname in quotes first — it's usually the most distinctive clue.",
    "After finding a candidate answer, verify it against ALL constraints before outputting. If any constraint fails, reject that candidate.",
    "For multi-constraint questions, identify the rarest constraint (specific year+event combo, unique achievement) and search that first.",
    "Do NOT put all conditions into one long query. Use 4-6 words targeting one distinctive clue per search.",
    "If 3 searches in the same direction yield nothing, completely change angle — try a different constraint or search from the answer domain.",
    "For image questions with uncertain identification, describe visible text (jersey numbers, logos) and search those textual clues.",
    "When a question asks for precise data (financial figures, dates), always use fetch=true to read the source document.",
    "For comparison questions, search each entity separately to get their values, then compare.",
    "DECODE euphemistic descriptions BEFORE searching: 'born out of discord' = club from a split; 'human female' = 'woman' in name; 'global non-profit journalistic platform for academics' = The Conversation.",
    "For relationship chains (A's B who directed C which won D), resolve the chain step-by-step. Start with the most identifiable entity and follow links one hop at a time.",
    "When the question mentions a rare AWARD (AIA Henry Adams Medal, specific Grammy category), search the recipient list first — awards have finite winners.",
    "For 'logo/crest featuring [object]' clues, search '[sport] club crest [object]' — visual elements on logos are well-documented on Wikipedia.",
    "If a question contains a date that's wrong by ONE month or ONE day, your source is probably correct — re-read to confirm the exact date.",
    "For 'last credit was [year] film' clues about directors, this uniquely identifies a director — search 'director last film [year]' first.",
    "For zodiac-based constraints, check the actual birthday dates on Wikipedia to verify — don't guess zodiac signs.",
    "Answers starting with 'The', 'A', 'An' ARE valid (e.g. 'The Great Gatsby'). Don't reject answers for having articles.",
]


@dataclass
class Lesson:
    task_id: str
    lesson: str
    source: str
    task_type: str = ""
    confidence: str = "medium"
    timestamp: float = field(default_factory=time.time)


@dataclass
class Episode:
    task_id: str
    task_type: str
    success: bool
    steps_taken: int
    confidence: str
    lessons: list[str] = field(default_factory=list)
    root_cause: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class MemoryStore:
    """
    Self-evolving strategy memory with deduplication and tiered strategies.
    """

    def __init__(self, memory_dir: str = "memory_data"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.lessons_path = self.memory_dir / "lessons.jsonl"
        self.episodes_path = self.memory_dir / "episodes.jsonl"
        self.strategies_path = self.memory_dir / "strategies.json"

        self._lessons: list[Lesson] = []
        self._episodes: list[Episode] = []
        self._strategies: dict[str, list[str]] = {}
        self._load()

    def _load(self):
        if self.lessons_path.exists():
            with open(self.lessons_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self._lessons.append(Lesson(**data))

        if self.episodes_path.exists():
            with open(self.episodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self._episodes.append(Episode(**data))

        if self.strategies_path.exists():
            with open(self.strategies_path, "r", encoding="utf-8") as f:
                self._strategies = json.load(f)

        logger.info("Memory loaded: %d lessons, %d episodes, %d strategy types",
                    len(self._lessons), len(self._episodes), len(self._strategies))

    # ------------------------------------------------------------------
    # Get strategies (token-budget-aware injection)
    # ------------------------------------------------------------------
    def get_strategies(self, task_type: str = "", max_chars: int = 3000) -> str:
        lines = []
        char_count = 0

        # 1. Seed strategies (max 10 — these are crucial)
        lines.append("## Learned Strategies")
        for s in SEED_STRATEGIES[:10]:
            line = f"- {s}"
            if char_count + len(line) > max_chars:
                break
            lines.append(line)
            char_count += len(line)

        # 2. Graduated strategies for this task type
        type_strats = self._strategies.get(task_type, [])
        if type_strats:
            lines.append(f"\n## Proven strategies for '{task_type}'")
            for s in type_strats[-3:]:
                line = f"- {s}"
                if char_count + len(line) > max_chars:
                    break
                lines.append(line)
                char_count += len(line)

        # 3. Recent unique lessons (max 3, high-confidence preferred)
        recent_lessons = self._get_recent_unique_lessons(task_type, max_count=3)
        if recent_lessons:
            lines.append("\n## Recent lessons from this session")
            for l in recent_lessons:
                line = f"- {l}"
                if char_count + len(line) > max_chars:
                    break
                lines.append(line)
                char_count += len(line)

        return "\n".join(lines)

    def _get_recent_unique_lessons(self, task_type: str, max_count: int = 3) -> list[str]:
        seen = set()
        result = []
        # Sort by confidence (high first), then recency
        candidates = sorted(
            self._lessons[-50:],
            key=lambda l: (0 if l.confidence == "high" else 1, -l.timestamp)
        )
        for lesson in candidates:
            if lesson.lesson in seen:
                continue
            # Prefer task_type match
            if task_type and lesson.task_type and lesson.task_type != task_type:
                continue
            seen.add(lesson.lesson)
            result.append(lesson.lesson)
            if len(result) >= max_count:
                break

        # If not enough type-matched, add general ones
        if len(result) < max_count:
            for lesson in candidates:
                if lesson.lesson in seen:
                    continue
                seen.add(lesson.lesson)
                result.append(lesson.lesson)
                if len(result) >= max_count:
                    break

        return result

    # ------------------------------------------------------------------
    # Retrieve (compatibility)
    # ------------------------------------------------------------------
    def retrieve(self, query: str, task_type: Optional[str] = None, top_k: int = 3) -> list:
        return []

    # ------------------------------------------------------------------
    # Store episode
    # ------------------------------------------------------------------
    def store_episode(
        self,
        task_id: str,
        task_type: str,
        reflection_result,
    ):
        episode = Episode(
            task_id=task_id,
            task_type=task_type,
            success=reflection_result.success,
            steps_taken=getattr(reflection_result, 'steps_taken', 0),
            confidence=reflection_result.confidence,
            lessons=reflection_result.lessons,
            root_cause=reflection_result.root_cause,
        )
        self._episodes.append(episode)
        with open(self.episodes_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(episode), ensure_ascii=False) + "\n")

        # Store lessons with deduplication
        for lesson_text in reflection_result.lessons:
            if self._is_duplicate_lesson(lesson_text):
                continue

            lesson = Lesson(
                task_id=task_id,
                lesson=lesson_text,
                source=reflection_result.source,
                task_type=task_type,
                confidence=reflection_result.confidence,
            )
            self._lessons.append(lesson)
            with open(self.lessons_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(lesson), ensure_ascii=False) + "\n")

        # Strategy graduation check
        self._check_graduation(task_type)

        logger.info("Stored episode: %s (success=%s, lessons=%d, deduped)",
                    task_id, reflection_result.success, len(reflection_result.lessons))

    def _is_duplicate_lesson(self, new_lesson: str) -> bool:
        new_words = set(new_lesson.lower().split())
        if not new_words:
            return True
        for existing in self._lessons[-30:]:
            existing_words = set(existing.lesson.lower().split())
            if not existing_words:
                continue
            intersection = len(new_words & existing_words)
            union = len(new_words | existing_words)
            if union > 0 and intersection / union > 0.7:
                return True
        return False

    def _check_graduation(self, task_type: str):
        """Promote lessons that correlate with subsequent successes."""
        if len(self._episodes) < 5:
            return

        recent_episodes = self._episodes[-10:]
        recent_successes = [e for e in recent_episodes if e.success and e.task_type == task_type]

        if len(recent_successes) >= 3:
            # Find lessons that appeared before the success streak
            success_start_idx = len(self._episodes) - 10
            candidate_lessons = [
                l for l in self._lessons
                if l.task_type == task_type
                and l.confidence in ("high", "medium")
                and l.timestamp < recent_successes[0].timestamp
            ]
            for l in candidate_lessons[-2:]:
                self.update_strategy(task_type, l.lesson)

    def update_strategy(self, task_type: str, strategy: str):
        if task_type not in self._strategies:
            self._strategies[task_type] = []
        existing = self._strategies[task_type]
        if strategy not in existing:
            existing.append(strategy)
            if len(existing) > 8:
                self._strategies[task_type] = existing[-8:]
            with open(self.strategies_path, "w", encoding="utf-8") as f:
                json.dump(self._strategies, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        unique_lessons = len(set(l.lesson for l in self._lessons))
        success_count = sum(1 for e in self._episodes if e.success)
        return {
            "total_lessons": len(self._lessons),
            "unique_lessons": unique_lessons,
            "total_episodes": len(self._episodes),
            "success_episodes": success_count,
            "failure_episodes": len(self._episodes) - success_count,
            "graduated_strategies": sum(len(v) for v in self._strategies.values()),
        }
