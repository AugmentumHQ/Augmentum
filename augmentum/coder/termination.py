"""Termination Quality Gate (TQG) — Phase 3.6 of the coder foundation.

When ``_act_hybrid`` reaches an iteration where the model emitted no tool
calls and a ``finish_reason`` of ``stop``, the harness must decide
between three outcomes:

* **Accept the stop** — the turn is genuinely complete or the prose is a
  legitimate substantive answer.
* **Nudge the model** — the stop is premature and a single re-prompt
  could recover the turn (model bailed on partial info, model produced
  one short sentence under an action request, etc).
* **Hard-stop** — already nudged once; further pushing would loop.

Pre-3.6 the decision lived inline in ``phase_act._act_hybrid`` as a
single ``if`` checking ``len(prose) >= 40``. That conflates three
semantically distinct things — *did the model produce an answer*, *did
the model do work this turn*, *did the user demand completion* — into
one length comparison. The 40-char bar is the length of an excuse, not
an answer; observed bailout that slipped through:

    "I read the file but the middle was elided."   (47 chars)

This module separates the three signals into independent primitives
(``UserDemand``, ``ProseKind``, ``intent_is_action``) and composes them
in :func:`evaluate_termination`. Heuristic-only by design — no phrase
lists, no model calls. The decision must be reproducible without
network and explainable from a printed structure.

Design contract
---------------

* **Pure.** No filesystem, no async, no logging side effects. Caller
  feeds in a ``TerminationContext`` and gets back a ``TerminationVerdict``
  it can act on.
* **Phrase-agnostic.** Length, sentence count, and structural signals
  only. A regression that adds a phrase list to ProseKind classification
  is a code-smell — those lists rot and the model talks around them.
* **Defaults to acceptance.** When in doubt (UNKNOWN intent, ambiguous
  prose), prefer accepting the stop over an infinite-nudge loop. The
  pre-3.6 behaviour was over-accept; this module corrects it without
  swinging to over-nudge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from augmentum.modes.coder.intent import TurnIntentKind


# ---------------------------------------------------------------------------
# Primitive 1: user demand classification
# ---------------------------------------------------------------------------


class UserDemand(StrEnum):
    """How the user framed the request, w.r.t. completion expectations.

    * ``PASSIVE`` — analysis-style or open-ended ask. A substantive prose
      answer is a fine endpoint even with zero writes.
    * ``ACTIVE`` — the user expects work. Zero writes + short prose is
      suspect; a substantive answer can still satisfy.
    * ``INSISTENT`` — the user explicitly demanded completion (``until
      finished``, ``don't stop``). Bar for accepting a stop rises.
    * ``UNKNOWN`` — no signal either way. Defer to ``intent_kind``.
    """
    PASSIVE = "passive"
    ACTIVE = "active"
    INSISTENT = "insistent"
    UNKNOWN = "unknown"


# Generic patterns that capture the *shape* of insistence rather than
# specific phrases. Each catches a family — e.g. ``until\s+(?:absolutely|
# truly|completely|fully)`` covers "until absolutely finished",
# "until truly done", "until fully complete". The point is the
# adverb-of-totality plus a completion verb, not any one phrasing.
#
# All patterns are intentionally broad: a false positive (treating an
# active request as insistent) only raises the bar by one tier, while a
# false negative (missing real insistence) is the bug we shipped to
# production. Asymmetric cost → bias toward broad capture.
_INSISTENT_PATTERNS = (
    # "until absolutely/fully/truly/completely finished/done/complete"
    re.compile(
        r"\b(?:until|till)\s+(?:absolutely|truly|completely|fully|"
        r"the\s+(?:task|job|work)\s+is)\b",
        re.IGNORECASE,
    ),
    # "don't stop until X" / "don't stop"
    re.compile(r"\bdon'?t\s+stop\b", re.IGNORECASE),
    # "keep going until X"
    re.compile(r"\bkeep\s+going\s+until\b", re.IGNORECASE),
    # "go all the way" / "all the way through"
    re.compile(r"\ball\s+the\s+way\b", re.IGNORECASE),
    # "fully implement" / "fully complete" — intensifier on completion
    re.compile(
        r"\b(?:fully|completely|absolutely)\s+"
        r"(?:complete|implement|finish|deliver|done)\b",
        re.IGNORECASE,
    ),
    # "finish the whole thing" / "the entire X"
    re.compile(
        r"\b(?:finish|complete)\s+(?:the\s+)?"
        r"(?:whole|entire|full)\b",
        re.IGNORECASE,
    ),
)

# A small set of "open-ended question / read-only investigation" markers.
# These tilt PASSIVE — a substantive prose answer is a legitimate end
# state without any writes.
_PASSIVE_PATTERNS = (
    # "explain how X works" / "tell me about Y"
    re.compile(
        r"\b(?:explain|tell\s+me\s+about|walk\s+me\s+through|"
        r"talk\s+me\s+through|describe|summarize)\b",
        re.IGNORECASE,
    ),
    # "what is X?" / "where is Y?" / "why does Z?"
    re.compile(r"^\s*(?:what|where|why|how)\b.*\?$", re.IGNORECASE),
)


def classify_user_demand(text: str) -> UserDemand:
    """Classify the user's message by completion-expectation shape.

    Order matters: insistence is the strongest signal and supersedes
    everything else. Passive markers only fire when no insistence is
    detected. ``ACTIVE`` is the common-case fallback when the user
    issued a verb-style request without explicit insistence framing.
    Empty / whitespace input → ``UNKNOWN``.
    """
    raw = (text or "").strip()
    if not raw:
        return UserDemand.UNKNOWN

    for pat in _INSISTENT_PATTERNS:
        if pat.search(raw):
            return UserDemand.INSISTENT

    for pat in _PASSIVE_PATTERNS:
        if pat.search(raw):
            return UserDemand.PASSIVE

    return UserDemand.UNKNOWN


# ---------------------------------------------------------------------------
# Primitive 2: prose-kind classification
# ---------------------------------------------------------------------------


class ProseKind(StrEnum):
    """How the model's stop-prose presents.

    * ``EMPTY`` — < 20 chars or whitespace. The model said nothing
      useful.
    * ``BAILOUT`` — short and single-sentence. The shape of an excuse:
      "I would need…", "I read the file but the middle was elided.",
      "Let me know if you want me to continue." — too brief to be a
      real answer, structured to look like one.
    * ``SUBSTANTIVE`` — long enough or multi-sentence enough to be a
      real answer. Pre-3.6 the only check was ``>= 40 chars``; the
      revised threshold uses *both* length and sentence count so a
      single 47-char sentence can no longer pass as substantive.
    """
    EMPTY = "empty"
    BAILOUT = "bailout"
    SUBSTANTIVE = "substantive"


# Length / structure thresholds. The pre-3.6 bar
# (``_HYBRID_MEANINGFUL_ANSWER_CHARS = 40``) was a flat length check.
# We replace it with a two-axis gate (length AND sentence count) that
# discriminates a one-line excuse from a real summary at similar
# character counts. Tuned so:
#
# * "I read the file but the middle was elided." (47 chars, 1 sentence)
#   → BAILOUT (was SUBSTANTIVE pre-3.6 — the headline bug)
# * "Done. Added helper, ran tests, all pass." (41 chars, 4 sentences)
#   → SUBSTANTIVE (legitimate compact completion summary)
# * "I would need more information to continue." (42 chars, 1 sentence)
#   → BAILOUT (the bail-shaped pattern — verbose-looking but evasive)
# * "I'll add the helper next." (25 chars, 1 sentence) → BAILOUT
# * Any prose >= 200 chars → SUBSTANTIVE (legitimate single-paragraph
#   answer — length floor alone justifies it)
#
# The headline insight: at this character range, sentence count is the
# stronger discriminator than length. A 41-char message saying four
# things is an answer; a 47-char message saying one thing is a bail.
_PROSE_EMPTY_MAX = 20
_PROSE_SUBSTANTIVE_MIN_LONG = 200       # over this, single sentence is fine
_PROSE_SUBSTANTIVE_MIN_CHARS = 30       # in the multi-sentence band
_PROSE_SUBSTANTIVE_MIN_SENTENCES = 2    # short prose needs >= 2 sentences

# Sentence terminator regex — counts ``.``, ``!``, ``?`` followed by
# whitespace or end-of-string. Multi-punctuation runs (``...``, ``!!``)
# count as one. Inline ``.`` inside numbers ("v1.2") is a known false
# positive but cheap; for the 65-char regime we're already deep in
# bailout territory and one extra inflated sentence count won't tip
# something out of BAILOUT.
_SENTENCE_END_RE = re.compile(r"[.!?]+(?:\s+|$)")


def classify_prose(text: str) -> ProseKind:
    """Classify the model's clean-text content at stop time.

    See :data:`_PROSE_EMPTY_MAX` etc. for the threshold rationale.
    """
    raw = (text or "").strip()
    if len(raw) < _PROSE_EMPTY_MAX:
        return ProseKind.EMPTY

    # Long-form single-paragraph answers are legitimate substantive
    # prose even when they're one sentence — the length floor is doing
    # the work there, not the sentence count.
    if len(raw) >= _PROSE_SUBSTANTIVE_MIN_LONG:
        return ProseKind.SUBSTANTIVE

    sentence_count = len(_SENTENCE_END_RE.findall(raw))
    if (
        len(raw) >= _PROSE_SUBSTANTIVE_MIN_CHARS
        and sentence_count >= _PROSE_SUBSTANTIVE_MIN_SENTENCES
    ):
        return ProseKind.SUBSTANTIVE

    # Either too short, or short-and-single-sentence — both bail.
    return ProseKind.BAILOUT


# ---------------------------------------------------------------------------
# Primitive 3: intent → "is this an action request?"
# ---------------------------------------------------------------------------


# Intents that genuinely produce filesystem state changes. Zero writes
# under these intents at stop time is a strong signal something is
# wrong. INSPECT / RESEARCH / REVIEW are read-only by design — a
# substantive prose answer with zero writes is the *expected* outcome
# for those.
_ACTION_INTENTS: frozenset[TurnIntentKind] = frozenset({
    TurnIntentKind.IMPLEMENT,
    TurnIntentKind.DEBUG,
    TurnIntentKind.OPERATE,
})


def intent_is_action(kind: TurnIntentKind) -> bool:
    """True iff the turn was classified as wanting filesystem changes.

    UNKNOWN is treated as action — biased toward not letting the model
    quietly skip work on ambiguous requests. The cost of a false-action
    classification is one extra nudge; the cost of a false-passive is
    silent task abandonment (the bug we're fixing).
    """
    return kind == TurnIntentKind.UNKNOWN or kind in _ACTION_INTENTS


# ---------------------------------------------------------------------------
# Composition: TerminationContext / TerminationVerdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TerminationContext:
    """Inputs the gate needs to decide on a stop.

    Built by the act loop at the moment the model emits no tool calls
    and finish_reason=stop. Pure data — the gate doesn't read any other
    state.
    """
    user_text: str
    intent_kind: TurnIntentKind
    clean_prose: str
    total_writes: int
    had_recent_progress: bool
    continuation_nudged: bool


@dataclass(frozen=True, slots=True)
class TerminationVerdict:
    """The gate's answer.

    ``accept_stop`` is the primary boolean. ``reason`` is a short tag
    suitable for telemetry and chat_egress meta chunks. ``nudge_kind``
    is "" when accepting; otherwise names the nudge variant the act
    loop should send.
    """
    accept_stop: bool
    reason: str
    nudge_kind: str = ""

    def explain(self) -> str:
        """One-line human-readable explanation for traces / debug logs.

        The ``reason`` tag is for telemetry; this is for the developer
        reading a session log trying to figure out why the loop did
        what it did.
        """
        if self.accept_stop:
            return _ACCEPT_EXPLANATIONS.get(
                self.reason, f"Accepted stop ({self.reason}).",
            )
        return _NUDGE_EXPLANATIONS.get(
            self.reason, f"Nudged: {self.reason}.",
        )


# Reason tags. Stable strings — exercised by chat_egress validators
# and by tests that pin the gate's decision-tree shape.
REASON_RECENT_PROGRESS = "recent_progress"
REASON_ALREADY_NUDGED = "already_nudged"
REASON_SUBSTANTIVE_PASSIVE = "substantive_under_passive"
REASON_SUBSTANTIVE_NON_ACTION = "substantive_under_non_action_intent"
REASON_SUBSTANTIVE_ACTIVE = "substantive_under_active"
REASON_NUDGE_INSISTENT = "nudge_insistent_zero_writes"
REASON_NUDGE_BAILOUT = "nudge_bailout_under_action"
REASON_NUDGE_EMPTY = "nudge_empty_prose"

NUDGE_NO_PROGRESS = "no_progress_action_turn"
NUDGE_INSISTENCE = "user_demanded_completion"
NUDGE_BAILOUT = "bailout_short_prose"


# Lookup tables for :meth:`TerminationVerdict.explain`. Module-private —
# callers should not couple to specific message phrasings; they should
# use the stable ``reason`` tag for any structured branching.
_ACCEPT_EXPLANATIONS: dict[str, str] = {
    REASON_ALREADY_NUDGED:
        "Accepted stop: already nudged once this turn; bound respected.",
    REASON_RECENT_PROGRESS:
        "Accepted stop: writes occurred recently; prose is a wrap-up.",
    REASON_SUBSTANTIVE_PASSIVE:
        "Accepted stop: substantive prose under a passive (analysis) request.",
    REASON_SUBSTANTIVE_NON_ACTION:
        "Accepted stop: substantive prose under a non-action intent "
        "(inspect/review/research).",
    REASON_SUBSTANTIVE_ACTIVE:
        "Accepted stop: substantive prose under an action request "
        "with zero writes — model articulated a genuine answer.",
}

_NUDGE_EXPLANATIONS: dict[str, str] = {
    REASON_NUDGE_INSISTENT:
        "Nudged: user explicitly demanded completion (until-finished, "
        "don't-stop, fully-complete) and the model produced zero writes.",
    REASON_NUDGE_BAILOUT:
        "Nudged: prose is short and single-sentence — has the shape "
        "of a bail-out, not an answer.",
    REASON_NUDGE_EMPTY:
        "Nudged: model produced no prose worth showing the user.",
}


def evaluate_termination(ctx: TerminationContext) -> TerminationVerdict:
    """Decide whether to accept the model's stop or nudge once more.

    Decision tree (top-to-bottom; first match wins):

    1. **Already nudged once** → accept (avoid infinite-nudge loops).
       Pre-3.6 behaviour preserved — one nudge is the bound.
    2. **Recent progress** → accept. Writes happened in the lookback
       window; the prose is a wrap-up, not a bail.
    3. **INSISTENT user demand + zero writes** → nudge regardless of
       prose. The user explicitly asked for completion; a one-paragraph
       summary is not completion. (This is the gate's headline use.)
    4. **Action intent + zero writes**:
       * empty prose → nudge (no answer at all)
       * bailout prose → nudge (short, single-sentence excuse)
       * substantive prose → accept (genuine multi-sentence answer)
    5. **Non-action intent (INSPECT/REVIEW/RESEARCH)** → substantive
       prose accepts; bailout/empty nudges. Read-only intents legitimately
       produce only prose, but the prose still has to be a real answer.

    UNKNOWN intent is treated as action (see :func:`intent_is_action`).
    """
    # Rule 1: bound the nudge depth.
    if ctx.continuation_nudged:
        return TerminationVerdict(
            accept_stop=True, reason=REASON_ALREADY_NUDGED,
        )

    # Rule 2: recent writes mean the model did work; prose is a wrap.
    if ctx.had_recent_progress:
        return TerminationVerdict(
            accept_stop=True, reason=REASON_RECENT_PROGRESS,
        )

    demand = classify_user_demand(ctx.user_text)
    prose = classify_prose(ctx.clean_prose)
    is_action = intent_is_action(ctx.intent_kind)

    # Rule 3: user explicitly demanded completion. Zero writes is
    # automatic-fail regardless of prose quality — the user asked for
    # a finished result, not a status report.
    if demand == UserDemand.INSISTENT and ctx.total_writes == 0:
        return TerminationVerdict(
            accept_stop=False,
            reason=REASON_NUDGE_INSISTENT,
            nudge_kind=NUDGE_INSISTENCE,
        )

    # Rule 4: action intent, no writes — prose has to be substantive
    # for the stop to be acceptable.
    if is_action and ctx.total_writes == 0:
        if prose == ProseKind.EMPTY:
            return TerminationVerdict(
                accept_stop=False,
                reason=REASON_NUDGE_EMPTY,
                nudge_kind=NUDGE_NO_PROGRESS,
            )
        if prose == ProseKind.BAILOUT:
            return TerminationVerdict(
                accept_stop=False,
                reason=REASON_NUDGE_BAILOUT,
                nudge_kind=NUDGE_BAILOUT,
            )
        # SUBSTANTIVE under ACTIVE / UNKNOWN demand — accept. The model
        # gave a real answer even though no writes happened (e.g.,
        # genuine "blocked because" explanation).
        return TerminationVerdict(
            accept_stop=True, reason=REASON_SUBSTANTIVE_ACTIVE,
        )

    # Rule 5: non-action intent (INSPECT / RESEARCH / REVIEW). These
    # legitimately produce prose-only outcomes — but the prose must
    # still be a real answer.
    if prose == ProseKind.SUBSTANTIVE:
        if demand == UserDemand.PASSIVE:
            return TerminationVerdict(
                accept_stop=True, reason=REASON_SUBSTANTIVE_PASSIVE,
            )
        return TerminationVerdict(
            accept_stop=True, reason=REASON_SUBSTANTIVE_NON_ACTION,
        )

    # Bailout or empty under non-action intent — still nudge once.
    if prose == ProseKind.EMPTY:
        return TerminationVerdict(
            accept_stop=False,
            reason=REASON_NUDGE_EMPTY,
            nudge_kind=NUDGE_NO_PROGRESS,
        )
    return TerminationVerdict(
        accept_stop=False,
        reason=REASON_NUDGE_BAILOUT,
        nudge_kind=NUDGE_BAILOUT,
    )
