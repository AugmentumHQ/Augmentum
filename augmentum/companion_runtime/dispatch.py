"""LLM-free dispatch with LLM tie-breaker.

The dispatcher decides which subagent should handle an :class:`Intent`.
Sprint 3's commitment: the LLM is **not** in the critical path for
unambiguous decisions. Heuristic utility scoring picks the winner;
the LLM is consulted only when the top two candidates are within
``TIE_BREAKER_THRESHOLD`` utility of each other.

Design rationale (sprint-plan §5):
- We have ~12 lexical/contextual features. Picking the top of 7
  subagents on those features is a comparison, not a generation —
  no creativity needed, no LLM warranted.
- The tie-breaker is bounded: hard 800ms cap, single 200-token
  rationale prompt, falls back to deterministic top-1 on timeout.
- The legacy UI mode-toggle survives as one feature among the 12
  (``UserModeHintFeature``). Worst-case regression is bounded: we
  pick exactly what the mode-toggle would have picked.

Validation point: ``passthrough/orchestrator.py::SSOSOrchestrator``
already proved heuristic-primary + LLM-tie-breaker works for a
related decision (which tool to pre-execute). We generalize that
pattern.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime, Intent
    from augmentum.companion_runtime.subagents.base import SubagentBase

log = get_logger(__name__)

# Tunables. These are intentional defaults, not magic numbers — each
# has a justification in the sprint plan.
TIE_BREAKER_THRESHOLD: float = 0.05      # plan §5 — top-1 wins outright above this margin
TIE_BREAKER_TIMEOUT_S: float = 0.8       # plan §11 risk — hard cap to keep p99 sane
TIE_BREAKER_MAX_RATIONALE_TOKENS: int = 200
TIE_BREAKER_MAX_CANDIDATES: int = 2      # only ever break ties between top 2

# Feature weights. The user_mode_hint weight starts high (0.35) so the
# legacy UI mode-toggle dominates by default; telemetry will let us
# tune it down once we have real-world distributions.
FEATURE_WEIGHTS: dict[str, float] = {
    "lexical_name": 0.18,
    "lexical_description": 0.10,
    "state_affinity": 0.08,
    "role_affinity": 0.08,
    "focus_affinity": 0.06,
    "recency": 0.05,
    "user_mode_hint": 0.35,
    "persona_kernel_affinity": 0.05,
    "intent_source": 0.05,
}


# ── Data shapes ──────────────────────────────────────────────────────

@dataclass(slots=True)
class FeatureScore:
    """A single feature's contribution to a candidate's utility."""
    name: str
    raw: float                # in [0, 1]
    weighted: float           # raw * FEATURE_WEIGHTS[name]


@dataclass(slots=True)
class Candidate:
    """A subagent considered by the dispatcher."""
    name: str
    subagent: SubagentBase
    features: list[FeatureScore] = field(default_factory=list)
    utility: float = 0.0      # sum of weighted feature scores
    explanation: str = ""     # human-readable; built lazily


@dataclass(slots=True)
class DispatchDecision:
    """The dispatcher's final pick + audit trail."""
    winner: Candidate | None
    ranked: list[Candidate]
    used_tiebreaker: bool = False
    tiebreaker_rationale: str = ""
    decision_ms: float = 0.0
    abstained: bool = False        # True if no candidate cleared the floor


# ── Feature extractors ───────────────────────────────────────────────

_LEX_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_LEX_TOKEN_RE.findall(text.lower()))


def _lexical_overlap(intent_text: str, target_text: str) -> float:
    """Jaccard overlap between intent tokens and target tokens.

    Used for both ``lexical_name`` and ``lexical_description`` features.
    Returns 0.0 on empty inputs to avoid div-by-zero biasing toward
    candidates whose names happen to be empty (impossible in practice
    given SubagentBase.__init_subclass__ enforces non-empty name).
    """
    a = _tokenize(intent_text)
    b = _tokenize(target_text)
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union)


def _affinity_score(active: str, affinity: tuple[str, ...]) -> float:
    """1.0 if active is in the affinity tuple, 0.4 if affinity is
    empty (neutral), 0.0 otherwise. Empty-as-neutral matters because
    most adapters won't declare every axis."""
    if not affinity:
        return 0.4
    return 1.0 if active in affinity else 0.0


def _recency_score(last_used_at: float, now: float, *, half_life_s: float = 600.0) -> float:
    """Exponential decay since last successful invocation. ``half_life_s``
    of 10 minutes means: a subagent invoked 10 minutes ago scores 0.5,
    20 minutes ago 0.25. Encourages re-using a recent winner without
    locking the dispatcher into one choice.
    """
    if last_used_at <= 0:
        return 0.0
    dt = max(0.0, now - last_used_at)
    return 0.5 ** (dt / half_life_s)


def _intent_source_compat(source: str, subagent: SubagentBase) -> float:
    """Compatibility of intent source with subagent. Tick-sourced
    intents (Sprint 4a) shouldn't route to user-facing modes by
    default; voice-sourced intents prefer fast modes. Sprint 3 ships
    a coarse table — Sprint 4a will refine.
    """
    if source == "tick":
        # Autonomous tick intents prefer journal/narrative/build paths
        # over conversation modes; agentic/coder are too heavyweight
        # for the tick budget.
        if subagent.name in ("narrative", "build"):
            return 1.0
        if subagent.name in ("agentic", "coder"):
            return 0.1
        return 0.5
    if source == "user_voice":
        # Voice prefers low-latency replies — penalize streaming heavies.
        if subagent.name in ("passthrough", "analytical"):
            return 1.0
        if subagent.name in ("agentic", "coder", "bug_finder"):
            return 0.2
        return 0.6
    return 0.5  # unknown/default source — neutral


# ── Recency cache (in-memory, per-process) ───────────────────────────

class _RecencyTracker:
    """Tracks last successful invocation timestamp per subagent.

    In-process state only — survives within a single FastAPI process
    but resets on restart. Persistence across restarts isn't worth the
    schema; recency is a soft signal and a cold start is just "no
    recent winner," which is fine.
    """

    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    def mark(self, name: str) -> None:
        self._last[name] = time.time()

    def get(self, name: str) -> float:
        return self._last.get(name, 0.0)


_RECENCY = _RecencyTracker()


def mark_invocation_success(subagent_name: str) -> None:
    """Public hook for runtime.submit_intent to record a successful
    invocation — used by future scoring rounds."""
    _RECENCY.mark(subagent_name)


# ── Scoring ──────────────────────────────────────────────────────────

def _score_candidate(
    candidate: Candidate,
    intent: Intent,
    *,
    runtime: CompanionRuntime,
    user_mode_hint: str = "",
    now: float | None = None,
) -> None:
    """Fill ``candidate.features`` and ``candidate.utility`` in place."""
    now = now if now is not None else time.time()
    sub = candidate.subagent
    feats: list[FeatureScore] = []

    raw_name = _lexical_overlap(intent.text, sub.name)
    raw_desc = _lexical_overlap(intent.text, sub.description)
    feats.append(FeatureScore(
        "lexical_name", raw_name, raw_name * FEATURE_WEIGHTS["lexical_name"],
    ))
    feats.append(FeatureScore(
        "lexical_description", raw_desc,
        raw_desc * FEATURE_WEIGHTS["lexical_description"],
    ))

    state_snap = runtime.state.snapshot()
    raw_state = _affinity_score(state_snap.get("state", ""), sub.state_affinity)
    raw_role = _affinity_score(state_snap.get("role_dominant", ""), sub.role_affinity)
    focus = state_snap.get("focus", {})
    focus_kind = focus.get("kind", "") if isinstance(focus, dict) else ""
    raw_focus = _affinity_score(focus_kind, sub.focus_affinity)
    feats.append(FeatureScore("state_affinity", raw_state, raw_state * FEATURE_WEIGHTS["state_affinity"]))
    feats.append(FeatureScore("role_affinity", raw_role, raw_role * FEATURE_WEIGHTS["role_affinity"]))
    feats.append(FeatureScore("focus_affinity", raw_focus, raw_focus * FEATURE_WEIGHTS["focus_affinity"]))

    raw_recency = _recency_score(_RECENCY.get(sub.name), now)
    feats.append(FeatureScore("recency", raw_recency, raw_recency * FEATURE_WEIGHTS["recency"]))

    raw_hint = 1.0 if user_mode_hint and user_mode_hint == sub.name else 0.0
    feats.append(FeatureScore("user_mode_hint", raw_hint, raw_hint * FEATURE_WEIGHTS["user_mode_hint"]))

    # Persona kernel affinity uses the identity's drift score as a
    # liveness signal — high drift (Becca diverged) penalizes
    # narrative/personal modes that depend on stable voice; low drift
    # leaves things neutral.
    drift = getattr(runtime.identity, "drift_score", 0.0) or 0.0
    if sub.name in ("narrative",) and drift > 0.10:
        raw_persona = max(0.0, 1.0 - drift / 0.15)
    else:
        raw_persona = 0.5
    feats.append(FeatureScore(
        "persona_kernel_affinity", raw_persona,
        raw_persona * FEATURE_WEIGHTS["persona_kernel_affinity"],
    ))

    raw_source = _intent_source_compat(intent.source, sub)
    feats.append(FeatureScore(
        "intent_source", raw_source, raw_source * FEATURE_WEIGHTS["intent_source"],
    ))

    candidate.features = feats
    candidate.utility = sum(f.weighted for f in feats)


def rank(
    intent: Intent,
    *,
    runtime: CompanionRuntime,
    user_mode_hint: str = "",
) -> list[Candidate]:
    """Rank all available subagents for the given intent.

    Returns candidates sorted by utility descending. Empty list if the
    subagent registry is inactive (flag off) or no adapters registered.
    """
    from augmentum.companion_runtime.subagents.registry import SubagentRegistry
    subs = SubagentRegistry.available()
    if not subs:
        return []
    now = time.time()
    candidates = [Candidate(name=s.name, subagent=s) for s in subs]
    for c in candidates:
        _score_candidate(c, intent, runtime=runtime, user_mode_hint=user_mode_hint, now=now)
    candidates.sort(key=lambda c: c.utility, reverse=True)
    return candidates


# ── Tie-breaker ──────────────────────────────────────────────────────

def _build_tiebreaker_prompt(intent: Intent, top: list[Candidate]) -> str:
    """Single-shot prompt for the tie-breaker LLM."""
    lines = [
        "You are picking which assistant should handle a user request.",
        f"User request: {intent.text!r}",
        "Choices:",
    ]
    for i, c in enumerate(top, start=1):
        lines.append(f"  {i}. {c.name} — {c.subagent.description}")
    lines.append(
        "Reply with ONLY the number of the best choice, then a one-sentence "
        f"rationale, total at most {TIE_BREAKER_MAX_RATIONALE_TOKENS} tokens."
    )
    return "\n".join(lines)


async def _consult_tiebreaker(
    intent: Intent,
    top: list[Candidate],
    runtime: CompanionRuntime,
) -> tuple[Candidate, str]:
    """Single LLM call to break a tie. On any failure or timeout,
    returns ``(top[0], "")`` so the caller falls back deterministically.
    """
    # Tie-breaker runs at utility tier (cheap, fast; this is decision
    # support, not user-facing prose).
    try:
        from augmentum.companion_runtime import tiers
        backend, model_name = await tiers.utility(runtime)
    except Exception:
        return top[0], "no backend for tie-breaker"

    prompt = _build_tiebreaker_prompt(intent, top)

    async def _call() -> str:
        try:
            from augmentum.models.base import InternalChatRequest
        except Exception:
            return ""
        req = InternalChatRequest(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=TIE_BREAKER_MAX_RATIONALE_TOKENS,
        )
        try:
            resp = await backend.chat(req) if hasattr(backend, "chat") else None
        except Exception as exc:
            log.warning("tiebreaker_llm_failed", error=str(exc))
            return ""
        if resp is None:
            return ""
        from augmentum.models.base import response_text
        return response_text(resp)

    try:
        text = await asyncio.wait_for(_call(), timeout=TIE_BREAKER_TIMEOUT_S)
    except asyncio.TimeoutError:
        # Working-as-designed — the tiebreaker LLM is fast-or-skip, and a
        # timeout cleanly falls back to top-1. Was firing as ``warning`` at
        # ~40/day; debug keeps the signal available without flooding the
        # warning floor for a normal control-flow path.
        log.debug(
            "tiebreaker_timeout",
            elapsed_s=TIE_BREAKER_TIMEOUT_S,
            falling_back_to=top[0].name,
        )
        return top[0], "timeout"

    # Parse: leading digit picks the choice. Anything we can't parse →
    # top-1 deterministically.
    text = text.strip()
    if not text:
        return top[0], "empty"
    first_char = text[0]
    if first_char.isdigit():
        idx = int(first_char) - 1
        if 0 <= idx < len(top):
            return top[idx], text
    return top[0], text or "unparseable"


# ── Public entry ─────────────────────────────────────────────────────

async def decide(
    intent: Intent,
    *,
    runtime: CompanionRuntime,
    user_mode_hint: str = "",
) -> DispatchDecision:
    """Pick a subagent for the intent.

    The complete public surface of dispatch. ``runtime.submit_intent``
    calls this; nobody else should.
    """
    t0 = time.monotonic()
    ranked = rank(intent, runtime=runtime, user_mode_hint=user_mode_hint)
    if not ranked:
        return DispatchDecision(
            winner=None, ranked=[], abstained=True,
            decision_ms=(time.monotonic() - t0) * 1000.0,
        )

    # Sprint 4b — apply DPO-style preference deltas from the skill
    # archive. No-op when the flag is off, so Sprint 3 behavior is
    # preserved. Re-sort after applying.
    try:
        from augmentum.companion_runtime import dpo_retrieval
        deltas = await dpo_retrieval.preference_delta(
            runtime, intent, [c.name for c in ranked],
        )
    except Exception:
        log.warning("dpo_preference_failed", exc_info=True)
        deltas = {}
    if deltas:
        for c in ranked:
            d = deltas.get(c.name, 0.0)
            if d:
                c.utility += d
                c.features.append(FeatureScore(
                    "dpo_preference", d / 1.0 if d else 0.0, d,
                ))
        ranked.sort(key=lambda c: c.utility, reverse=True)

    if len(ranked) == 1:
        return DispatchDecision(
            winner=ranked[0], ranked=ranked,
            decision_ms=(time.monotonic() - t0) * 1000.0,
        )

    top_two = ranked[:TIE_BREAKER_MAX_CANDIDATES]
    margin = top_two[0].utility - top_two[1].utility
    if margin >= TIE_BREAKER_THRESHOLD:
        return DispatchDecision(
            winner=top_two[0], ranked=ranked,
            decision_ms=(time.monotonic() - t0) * 1000.0,
        )

    log.info(
        "dispatch_tiebreaker_consulted",
        top1=top_two[0].name, u1=round(top_two[0].utility, 3),
        top2=top_two[1].name, u2=round(top_two[1].utility, 3),
        margin=round(margin, 3),
    )
    chosen, rationale = await _consult_tiebreaker(intent, top_two, runtime)
    return DispatchDecision(
        winner=chosen, ranked=ranked,
        used_tiebreaker=True,
        tiebreaker_rationale=rationale,
        decision_ms=(time.monotonic() - t0) * 1000.0,
    )


__all__ = [
    "Candidate",
    "DispatchDecision",
    "FeatureScore",
    "FEATURE_WEIGHTS",
    "TIE_BREAKER_THRESHOLD",
    "TIE_BREAKER_TIMEOUT_S",
    "decide",
    "mark_invocation_success",
    "rank",
]
