"""Typed turn-intent classification for coder mode.

This module deliberately starts small. It provides a reusable internal
contract for turn intent without forcing a broad plan/act rewrite.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class TurnIntentKind(StrEnum):
    UNKNOWN = "unknown"
    INSPECT = "inspect"
    REVIEW = "review"
    IMPLEMENT = "implement"
    DEBUG = "debug"
    OPERATE = "operate"
    RESEARCH = "research"


@dataclass(frozen=True, slots=True)
class TurnIntent:
    kind: TurnIntentKind
    read_only_by_default: bool = False
    explicit_execution: bool = False
    reason: str = ""


_EXECUTION_RE = re.compile(
    r"\b("
    r"run|execute|launch|start|boot|spin\s+up|open|test|pytest|"
    r"build|compile|install|benchmark|profile|debug"
    r")\b",
    re.IGNORECASE,
)

_REVIEW_RE = re.compile(
    r"\b("
    r"review|audit|assess|security\s+review|code\s+review|"
    r"find\s+(?:bugs|issues|risks)|look\s+for\s+(?:bugs|issues|risks)"
    r")\b",
    re.IGNORECASE,
)

_INSPECT_PHRASES = (
    "walk me through",
    "talk me through",
    "what files",
    "which files",
    "what python files",
    "which python files",
    "what's in this project",
    "what is in this project",
    "what does this project have",
    "what python files it has",
)

_INSPECT_TOKENS = (
    "explain",
    "describe",
    "summarize",
    "summarise",
    "overview",
    "inventory",
)

_OPERATE_RE = re.compile(
    r"\b("
    r"serve|server|port\s+forward|forward\s+port|expose|tunnel|"
    r"localtunnel|ngrok|cloudflared|start\s+the\s+app|run\s+the\s+app"
    r")\b",
    re.IGNORECASE,
)

_DEBUG_RE = re.compile(
    r"\b("
    r"debug|investigate|root\s+cause|traceback|stack\s+trace|"
    r"failing|failure|regression|why\s+(?:is|does|did)"
    r")\b",
    re.IGNORECASE,
)

_RESEARCH_RE = re.compile(
    r"\b("
    r"latest|recent|current|release\s+notes|look\s+up|search\s+the\s+web|"
    r"browse|research|docs|documentation"
    r")\b",
    re.IGNORECASE,
)

_IMPLEMENT_RE = re.compile(
    r"\b("
    r"add|create|implement|fix|change|modify|refactor|update|build"
    r")\b",
    re.IGNORECASE,
)


def explicitly_requests_execution(text: str) -> bool:
    """True when the user clearly asked to run, build, or mutate runtime state."""
    if not text:
        return False
    return bool(_EXECUTION_RE.search(text.strip()))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def classify_turn_intent(*, latest_text: str, goal_text: str = "") -> TurnIntent:
    """Classify the turn into a coarse controller intent.

    `goal_text` can be an earlier non-continuation user objective, allowing
    the classifier to preserve intent across "continue" turns.
    """
    latest = _normalize(latest_text)
    goal = _normalize(goal_text)
    subject = goal or latest
    explicit = explicitly_requests_execution(latest) or explicitly_requests_execution(goal)

    if any(phrase in subject for phrase in _INSPECT_PHRASES) or any(
        token in subject for token in _INSPECT_TOKENS
    ):
        return TurnIntent(
            kind=TurnIntentKind.INSPECT,
            read_only_by_default=True,
            explicit_execution=explicit,
            reason="inspect_phrase",
        )

    if _REVIEW_RE.search(subject):
        return TurnIntent(
            kind=TurnIntentKind.REVIEW,
            read_only_by_default=True,
            explicit_execution=explicit,
            reason="review_phrase",
        )

    if _OPERATE_RE.search(subject):
        return TurnIntent(
            kind=TurnIntentKind.OPERATE,
            explicit_execution=explicit,
            reason="operate_phrase",
        )

    if _DEBUG_RE.search(subject):
        return TurnIntent(
            kind=TurnIntentKind.DEBUG,
            explicit_execution=explicit,
            reason="debug_phrase",
        )

    if _RESEARCH_RE.search(subject):
        return TurnIntent(
            kind=TurnIntentKind.RESEARCH,
            explicit_execution=explicit,
            reason="research_phrase",
        )

    if _IMPLEMENT_RE.search(subject):
        return TurnIntent(
            kind=TurnIntentKind.IMPLEMENT,
            explicit_execution=explicit,
            reason="implement_phrase",
        )

    return TurnIntent(
        kind=TurnIntentKind.UNKNOWN,
        explicit_execution=explicit,
        reason="fallback",
    )


# ---------------------------------------------------------------------------
# Tier classification — Phase 1 of the coder foundation.
# Orthogonal to intent kind: INSPECT can be REFLEX (one-line lookup) or
# SURGICAL (small file walk); IMPLEMENT can be REFLEX (add an import) or
# PROJECT (build a CLI from scratch). Tier governs which execution path
# the handler routes through and what iteration/budget caps apply.
#
# Heuristic-only by design. The whole point of REFLEX is to avoid paying
# for any unnecessary LLM call — including the classifier itself. When
# signals are weak the classifier defaults to COMPOSED (current behavior),
# which keeps the existing hybrid loop in charge. Misclassification cost
# is bounded: the act-loop's existing termination guards catch under-
# tiered work; over-tiered work just wastes compute.
# ---------------------------------------------------------------------------


class Tier(StrEnum):
    REFLEX = "reflex"        # single-shot, ~500 tok, <2s
    SURGICAL = "surgical"    # 1-3 files, plan + short act, ~3K tok
    COMPOSED = "composed"    # multi-file, full hybrid loop (default)
    PROJECT = "project"      # from-scratch / large redesign


@dataclass(frozen=True, slots=True)
class TierClassification:
    tier: Tier
    reason: str
    signals: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TierLimit:
    max_iterations: int
    max_tokens: int


TIER_LIMITS: dict[Tier, TierLimit] = {
    Tier.REFLEX:   TierLimit(max_iterations=2,  max_tokens=2_000),
    Tier.SURGICAL: TierLimit(max_iterations=8,  max_tokens=10_000),
    Tier.COMPOSED: TierLimit(max_iterations=25, max_tokens=40_000),
    Tier.PROJECT:  TierLimit(max_iterations=60, max_tokens=120_000),
}


# --- PROJECT signals: from-scratch / migration / build-a-thing ---
# Allows an optional adjective between article and noun ("a new service")
# and up to 3 words between "rewrite" and the in/to/from connector.
_PROJECT_RE = re.compile(
    r"\b("
    r"(?:build|create|scaffold|make|set\s+up)\s+(?:a|an|the)(?:\s+\w+){0,2}\s+"
    r"(?:cli|app|application|service|server|website|web\s+app|"
    r"dashboard|game|tool|library|package|module|project)|"
    r"from\s+scratch|"
    r"start(?:ing)?\s+(?:from|with)\s+(?:nothing|empty|scratch)|"
    r"migrate\s+(?:from|to)|migration|"
    r"port\s+(?:from|to)|"
    r"rewrite(?:\s+\w+){1,4}\s+(?:in|to|from)"
    r")\b",
    re.IGNORECASE,
)

# --- COMPOSED signals: multi-file / cross-cutting refactor language ---
_COMPOSED_RE = re.compile(
    r"\b("
    r"refactor|restructure|reorganize|reorganise|consolidate|"
    r"extract\s+(?:into|to|a)|deduplicate|de-duplicate|"
    r"across\s+(?:the\s+)?(?:codebase|project|files|modules)|"
    r"throughout|every(?:where|\s+file)|"
    r"(?:all|both|each|several|multiple)\s+(?:files?|modules?|tests?|callers?)|"
    r"split\s+(?:up|into)|merge\s+(?:into|together)"
    r")\b",
    re.IGNORECASE,
)

# --- REFLEX signals: very specific small-edit verbs paired with one target ---
_REFLEX_VERB_RE = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+)?"
    r"(?:add|fix|rename|delete|remove|import|inline|change|update)\b",
    re.IGNORECASE,
)

# Phrases that disqualify REFLEX even when the verb matches (multi-step
# or scope-broadening language).
_NON_REFLEX_PHRASES = (
    " then ",
    " after that ",
    " and also ",
    " and then ",
    " everywhere",
    " across ",
    " throughout",
    " all the ",
    " every ",
    " refactor",
)

# A REFLEX message has to be terse AND single-clause. Multi-clause
# messages (commas, sub-clauses) almost always carry implicit scope
# beyond the headline verb.
_REFLEX_MAX_CHARS = 100


def classify_tier(
    *,
    latest_text: str,
    goal_text: str = "",
    workspace_file_count: int | None = None,
) -> TierClassification:
    """Classify the user's request into an execution tier.

    Heuristic-only. Returns ``Tier.COMPOSED`` (current hybrid behavior)
    when signals are weak — over-tiering is cheaper to recover from than
    under-tiering for a default-cautious foundation.

    ``workspace_file_count`` is an optional hint. Empty / nearly-empty
    workspaces lean PROJECT for build-a-thing prompts.
    """
    latest = _normalize(latest_text)
    goal = _normalize(goal_text)
    subject = goal or latest
    signals: list[str] = []

    # Empty workspace + creation verbs = PROJECT regardless of phrasing.
    empty_ws = workspace_file_count is not None and workspace_file_count <= 1
    if empty_ws and re.search(r"\b(build|create|scaffold|make)\b", subject):
        signals.append("empty_workspace_with_creation_verb")
        return TierClassification(
            tier=Tier.PROJECT,
            reason="empty_workspace_with_creation_verb",
            signals=tuple(signals),
        )

    # Explicit PROJECT phrases — strongest signal.
    if _PROJECT_RE.search(subject):
        signals.append("project_phrase")
        return TierClassification(
            tier=Tier.PROJECT,
            reason="project_phrase",
            signals=tuple(signals),
        )

    # COMPOSED phrases — multi-file or cross-cutting work.
    if _COMPOSED_RE.search(subject):
        signals.append("composed_phrase")
        return TierClassification(
            tier=Tier.COMPOSED,
            reason="composed_phrase",
            signals=tuple(signals),
        )

    # REFLEX requires four things together: terse, single small verb,
    # single-clause (no commas), and no scope-broadening language. Any
    # one missing → at least SURGICAL.
    is_terse = len(latest_text or "") <= _REFLEX_MAX_CHARS
    has_reflex_verb = bool(_REFLEX_VERB_RE.match(latest_text or ""))
    has_non_reflex_phrase = any(p in f" {latest} " for p in _NON_REFLEX_PHRASES)
    is_single_clause = "," not in (latest_text or "")

    if is_terse and has_reflex_verb and is_single_clause and not has_non_reflex_phrase:
        signals.append("terse_reflex_verb")
        return TierClassification(
            tier=Tier.REFLEX,
            reason="terse_single_action",
            signals=tuple(signals),
        )

    # If we have a single concrete action verb but the message isn't
    # terse enough for REFLEX, treat as SURGICAL.
    if has_reflex_verb or _IMPLEMENT_RE.search(subject) or _DEBUG_RE.search(subject):
        signals.append("single_action_verb")
        return TierClassification(
            tier=Tier.SURGICAL,
            reason="single_action_verb",
            signals=tuple(signals),
        )

    # Default: COMPOSED (current hybrid loop).
    return TierClassification(
        tier=Tier.COMPOSED,
        reason="default",
        signals=tuple(signals),
    )
