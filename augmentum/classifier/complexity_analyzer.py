"""Complexity analyzer — heuristic-based request complexity scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from augmentum.models.base import InternalChatRequest


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class ComplexityResult:
    level: ComplexityLevel
    confidence: float
    reason: str
    signals: list[str] = field(default_factory=list)


# --- Signal patterns ---

# Research / analytical trigger phrases
ANALYTICAL_TRIGGERS = [
    (re.compile(r"\b(?:analyze|analyse)\b", re.IGNORECASE), 0.20, "analyze"),
    (re.compile(r"\b(?:compare|contrast)\b", re.IGNORECASE), 0.20, "compare"),
    (re.compile(r"\b(?:research|investigate)\b", re.IGNORECASE), 0.25, "research"),
    (re.compile(r"\b(?:evaluate|assess)\b", re.IGNORECASE), 0.15, "evaluate"),
    (re.compile(r"\b(?:explain|describe)\s+(?:how|why|the\s+difference)", re.IGNORECASE), 0.15, "explain_deep"),
    (re.compile(r"\b(?:what are the (?:pros|cons|advantages|disadvantages|implications))", re.IGNORECASE), 0.20, "tradeoffs"),
    (re.compile(r"\b(?:step[\s-]by[\s-]step|systematically|methodically)\b", re.IGNORECASE), 0.15, "systematic"),
    (re.compile(r"\b(?:comprehensive|thorough|in[\s-]depth|detailed)\b", re.IGNORECASE), 0.15, "thorough"),
    (re.compile(r"\b(?:prove|derive|verify|validate)\b", re.IGNORECASE), 0.20, "verify"),
    (re.compile(r"\b(?:critique|review|audit)\b", re.IGNORECASE), 0.15, "critique"),
]

# Math / technical indicators
MATH_PATTERNS = [
    (re.compile(r"[=+\-*/^]{2,}|\\(?:frac|int|sum|prod|sqrt|lim)", re.IGNORECASE), 0.25, "math_notation"),
    (re.compile(r"\b(?:equation|formula|integral|derivative|matrix|vector)\b", re.IGNORECASE), 0.20, "math_term"),
    (re.compile(r"\b(?:calculate|compute|solve)\b", re.IGNORECASE), 0.15, "calculate"),
]

# Code indicators
CODE_PATTERNS = [
    (re.compile(r"```\w*\n"), 0.15, "code_block"),
    (re.compile(r"\b(?:function|class|def|import|const|let|var)\b"), 0.10, "code_keyword"),
    (re.compile(r"\b(?:debug|refactor|optimize|implement|architect)\b", re.IGNORECASE), 0.15, "dev_task"),
    (re.compile(r"\b(?:bug|error|exception|traceback|stack\s*trace)\b", re.IGNORECASE), 0.10, "debugging"),
]

# Multi-part / structured request indicators
STRUCTURE_PATTERNS = [
    (re.compile(r"(?:^|\n)\s*\d+[.)]\s+", re.MULTILINE), 0.15, "numbered_list"),
    (re.compile(r"(?:^|\n)\s*[-*]\s+", re.MULTILINE), 0.10, "bullet_list"),
    (re.compile(r"\b(?:first|second|third|finally|additionally|moreover)\b", re.IGNORECASE), 0.10, "sequence_words"),
]

# Simple / conversational indicators (reduce complexity score)
SIMPLE_PATTERNS = [
    (re.compile(r"^(?:hi|hello|hey|thanks|ok|yes|no|sure|great)\b", re.IGNORECASE), -0.30, "greeting"),
    (re.compile(r"^(?:what is|who is|when was|where is)\b", re.IGNORECASE), -0.10, "simple_question"),
    (re.compile(r"\?$"), -0.05, "single_question"),
]

ALL_SIGNALS = [
    *ANALYTICAL_TRIGGERS,
    *MATH_PATTERNS,
    *CODE_PATTERNS,
    *STRUCTURE_PATTERNS,
    *SIMPLE_PATTERNS,
]


class ComplexityAnalyzer:
    """Heuristic-based complexity analyzer — no model calls."""

    def analyze(self, request: InternalChatRequest) -> ComplexityResult:
        """Analyze request complexity from the last user message.

        Returns a complexity result with level and confidence.
        """
        user_content = self._extract_last_user_content(request)
        if not user_content:
            return ComplexityResult(
                level=ComplexityLevel.SIMPLE,
                confidence=1.0,
                reason="no user content",
            )

        score = 0.0
        matched_signals: list[str] = []

        for regex, weight, signal_name in ALL_SIGNALS:
            if regex.search(user_content):
                score += weight
                matched_signals.append(signal_name)

        # Length-based adjustments
        content_len = len(user_content)
        if content_len > 1000:
            score += 0.15
            matched_signals.append("long_message")
        elif content_len > 500:
            score += 0.10
            matched_signals.append("medium_message")
        elif content_len < 50:
            score -= 0.10
            matched_signals.append("short_message")

        # Multiple questions boost
        question_marks = user_content.count("?")
        if question_marks >= 3:
            score += 0.15
            matched_signals.append("multi_question")
        elif question_marks >= 2:
            score += 0.05
            matched_signals.append("dual_question")

        # Determine level
        if score >= 0.50:
            level = ComplexityLevel.COMPLEX
            confidence = min(score, 1.0)
            reason = f"complex ({score:.2f}): {', '.join(matched_signals[:5])}"
        elif score >= 0.25:
            level = ComplexityLevel.MODERATE
            confidence = score / 0.50  # Scale 0.25-0.50 → 0.5-1.0
            reason = f"moderate ({score:.2f}): {', '.join(matched_signals[:5])}"
        else:
            level = ComplexityLevel.SIMPLE
            confidence = 1.0 - max(score, 0.0) * 2  # Higher score = lower simple confidence
            reason = "simple" if not matched_signals else f"simple ({score:.2f})"

        return ComplexityResult(
            level=level,
            confidence=max(confidence, 0.0),
            reason=reason,
            signals=matched_signals,
        )

    def _extract_last_user_content(self, request: InternalChatRequest) -> str:
        """Get the content of the last user message."""
        for msg in reversed(request.messages):
            if msg.role == "user":
                return msg.content
        return ""
