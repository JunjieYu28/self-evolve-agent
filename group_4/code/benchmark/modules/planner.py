"""
Planner — 任务分类与策略推荐（适配 BrowseComp benchmark）
==========================================================

Benchmark 题型分析：
- Q1-Q50: 多约束实体识别（给出 5-15 个约束，找唯一实体）
- Q51-Q100: 图片→实体→属性（先识图再搜属性）

核心改进：
1. 识别 "multi_constraint" 为主要题型（而非泛化的 general）
2. estimated_steps 大幅提升（这类题通常需要 10-20 步）
3. 策略指导更具体（约束排序、验证循环）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PlanResult:
    task_type: str
    strategy_hint: str
    estimated_steps: int
    search_params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 分类规则
# ---------------------------------------------------------------------------
_COMPARISON_PATTERNS = [
    r"(?i)^(is|are|did|do|does|was|were)\s+.*\s+(higher|lower|more|less|greater|larger|smaller|bigger)",
    r"(?i)^(comparing|compare)\b",
    r"(?i)\b(higher than|lower than|more than|less than)\b.*\?$",
    r"(?i)^which\s+(had|has|is)\s+(a\s+)?(larger|smaller|higher|lower|bigger|more|fewer)",
]

_VISUAL_IDENTIFIER_PATTERNS = [
    r"(?i)\b(shown in the image|in the (picture|image|photo)|in this (picture|image|figure))\b",
    r"(?i)\b(the person|the player|the company|the product|the team)\s+in\s+(the\s+)?(image|picture|photo)\b",
]

_MULTI_CONSTRAINT_INDICATORS = [
    r"(?i)(born|founded|established|created|released)\s+(in|between|after|before)\s+",
    r"(?i)(between\s+\d{4}\s+and\s+\d{4})",
    r"(?i)(all of the (following|requirements|criteria))",
    r"(?i)(find\s+(this|the)\s+person|identify\s+the)",
    r"(?i)(A specific|A certain|An individual|A person|A company|A team|A property)",
    r"(?i)(who fits|that matches|who matches|that fits)\b",
]


# ---------------------------------------------------------------------------
# 策略模板
# ---------------------------------------------------------------------------
_STRATEGIES = {
    "multi_constraint": (
        "This is a multi-constraint entity identification problem. Strategy:\n"
        "1. List ALL constraints from the question\n"
        "2. Identify the 1-2 MOST DISTINCTIVE constraints (specific names, rare combinations, exact numbers)\n"
        "3. Search using those distinctive constraints first (4-6 words)\n"
        "4. When you find a candidate, VERIFY it against remaining constraints one by one\n"
        "5. If verification fails, reject the candidate and try a different search angle\n"
        "6. Only output the answer when ALL constraints are verified"
    ),
    "visual_entity": (
        "This is a visual question. Strategy:\n"
        "1. Use search_image to identify the person/object/brand in the image\n"
        "2. If image search is uncertain, describe visible text/logos/features and text-search them\n"
        "3. Once identified, search for the specific attribute asked in the question\n"
        "4. For time-sensitive questions (2024 events), include the year in your search"
    ),
    "comparison": (
        "This is a comparison question. Strategy:\n"
        "1. Search each entity separately to get their respective values\n"
        "2. Compare and answer directly\n"
        "3. For stock/financial data, use fetch=true to get precise numbers"
    ),
    "general": (
        "Search for the key entity/fact using the most distinctive keywords from the question. "
        "Verify your finding before answering."
    ),
}

_ESTIMATED_STEPS = {
    "multi_constraint": 15,
    "visual_entity": 8,
    "comparison": 6,
    "general": 10,
}

_SEARCH_PARAMS = {
    "multi_constraint": {"top_k": 3, "fetch": False, "max_chars": 800},
    "visual_entity": {"top_k": 3, "fetch": True, "max_chars": 800},
    "comparison": {"top_k": 3, "fetch": True, "max_chars": 800},
    "general": {"top_k": 3, "fetch": False, "max_chars": 800},
}


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class Planner:
    def classify(self, instruction: str, has_image: bool = False) -> str:
        if has_image:
            # Check if it's a comparison with image
            for pattern in _COMPARISON_PATTERNS:
                if re.search(pattern, instruction):
                    return "comparison"
            return "visual_entity"

        # Check comparison first
        for pattern in _COMPARISON_PATTERNS:
            if re.search(pattern, instruction):
                return "comparison"

        # Check multi-constraint (most Q1-Q50 questions)
        constraint_score = 0
        for pattern in _MULTI_CONSTRAINT_INDICATORS:
            if re.search(pattern, instruction):
                constraint_score += 1
        if constraint_score >= 2 or len(instruction) > 200:
            return "multi_constraint"

        return "general"

    def plan(self, instruction: str, has_image: bool = False) -> PlanResult:
        task_type = self.classify(instruction, has_image)
        return PlanResult(
            task_type=task_type,
            strategy_hint=_STRATEGIES[task_type],
            estimated_steps=_ESTIMATED_STEPS[task_type],
            search_params=_SEARCH_PARAMS[task_type],
        )
