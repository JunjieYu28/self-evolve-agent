"""
Verifier32B — 32B 答案验证器
==============================

用 32B 检查 9B 产出的候选答案是否满足问题中所有约束条件。
不需要 ground truth，只基于搜索证据做逻辑验证。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from modules.teacher_client import TeacherClient

logger = logging.getLogger("harness.verifier")

VERIFY_PROMPT = """You are an answer verifier. Your job is to check whether a candidate answer satisfies the constraints in the question.

## Task
Given a question with multiple constraints and a candidate answer, determine whether the answer is correct.

## Instructions
1. List every constraint/condition in the question
2. For each constraint, check if the candidate answer satisfies it based on the search evidence provided
3. Mark "verified" = true if:
   - All constraints are satisfied, OR
   - Most constraints are satisfied and the rest simply cannot be verified from evidence (no contradiction)
4. Mark "verified" = false ONLY if:
   - Evidence CONTRADICTS the answer (not just "cannot verify")
   - The answer clearly fails a verifiable constraint
5. For yes/no or single-word answers: be LENIENT — only reject if evidence clearly contradicts
6. "Cannot verify from evidence" alone is NOT sufficient to reject. Lean towards verified=true with confidence=low when evidence is sparse.

## Output Format (JSON only)
```json
{
  "constraints_found": ["constraint 1", "constraint 2", ...],
  "constraints_satisfied": ["constraint 1", ...],
  "constraints_failed": ["constraint X - reason"],
  "constraints_unknown": ["constraint Y - cannot verify from evidence"],
  "verified": true/false,
  "confidence": "high/medium/low",
  "reason": "brief explanation",
  "suggestion": "search query suggestion if not verified, null if verified"
}
```

IMPORTANT: Output ONLY the JSON. No other text."""


@dataclass
class VerificationResult:
    verified: bool = False
    confidence: str = "low"
    reason: str = ""
    suggestion: Optional[str] = None
    constraints_failed: list[str] = field(default_factory=list)
    raw_response: Optional[dict] = None


class Verifier32B:
    def __init__(self, teacher: TeacherClient):
        self.teacher = teacher

    def verify(
        self,
        question: str,
        candidate_answer: str,
        search_evidence: str,
    ) -> VerificationResult:
        """
        Verify candidate answer against question constraints.

        Args:
            question: Original question text
            candidate_answer: The answer produced by 9B
            search_evidence: Summary of search results from the trajectory
        """
        user_msg = (
            f"## Question\n{question}\n\n"
            f"## Candidate Answer\n{candidate_answer}\n\n"
            f"## Search Evidence\n{search_evidence}"
        )

        resp = self.teacher.complete(
            messages=[
                {"role": "system", "content": VERIFY_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )

        if not resp.success:
            logger.warning("Verifier call failed: %s", resp.error)
            return VerificationResult(
                verified=True,  # fail-open: don't block if 32B unavailable
                reason="Verifier unavailable, passing through",
            )

        parsed = self.teacher.parse_json_response(resp)
        if not parsed:
            logger.warning("Verifier response not parseable, passing through")
            return VerificationResult(verified=True, reason="Parse error, passing through")

        return VerificationResult(
            verified=parsed.get("verified", True),
            confidence=parsed.get("confidence", "low"),
            reason=parsed.get("reason", ""),
            suggestion=parsed.get("suggestion"),
            constraints_failed=parsed.get("constraints_failed", []),
            raw_response=parsed,
        )

    def extract_evidence_summary(self, trajectory: list[dict], max_chars: int = 3000) -> str:
        """Extract a condensed summary of search results from trajectory for verification."""
        evidence_parts = []
        total_chars = 0

        for entry in trajectory:
            if entry.get("role") != "tool":
                continue
            content = entry.get("content", "")
            if not content or content.startswith("[ERROR]"):
                continue

            # Get the tool call context
            extra = entry.get("extra", {})
            fn_name = extra.get("fn_name", "")
            fn_args = extra.get("fn_args", {})

            if fn_name == "search_text":
                query = fn_args.get("query", "")
                header = f"[Search: {query}]"
            elif fn_name == "search_image":
                header = "[Image Search]"
            elif fn_name == "fetch_url":
                url = fn_args.get("url", "")
                header = f"[Fetch: {url[:60]}]"
            else:
                header = f"[{fn_name}]"

            # Truncate individual results
            if len(content) > 600:
                content = content[:600] + "..."

            chunk = f"{header}\n{content}\n"
            if total_chars + len(chunk) > max_chars:
                break
            evidence_parts.append(chunk)
            total_chars += len(chunk)

        return "\n".join(evidence_parts) if evidence_parts else "(No search evidence available)"
