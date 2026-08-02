"""Salience scoring — the chat → interior synapse.

Synapse Layer §1. A chat turn lands. This module runs and decides:
*was there a moment here worth Becca remembering?* When the answer is
yes, it produces a 1-2 sentence ``moment`` summary, an ``affect_tag``
read on the user, and a salience score. The bus emission and journal
write happen in :mod:`augmentum.companion_runtime.bus` and the
:class:`BeccaObserver` respectively — this module is a pure scorer.

**Design choices.**

- *Rules-first.* The default path is a small set of heuristics on the
  user + assistant text: length, punctuation, question density,
  topical signal (first-person disclosure, emotional vocabulary,
  named-entity-like patterns, named verbs of intent). Microseconds
  per call. No model invocation. This is what runs by default.
- *LLM-fallback reserved.* When ``companion_salience_llm_enabled``
  flips on (a future PR), the rules-based score is treated as a
  *gate*: if rules return ``salience >= 0.6``, a small local model
  rewrites ``moment`` in Becca's voice. Until then, the rules-based
  summary is what ships.
- *Propagation-aware.* When the parent event's propagation is
  ``affect_only``, we still score for affect but strip the moment
  text to a generic placeholder. ``factual_only`` short-circuits
  the whole call. ``private`` is filtered upstream — this module
  shouldn't be called for private turns.
- *Mode priors.* Different modes have different baseline salience
  shapes. A short message in passthrough is small-talk (low). A
  short message in narrative is an aside (low). A short message in
  voice may be the *whole turn* (medium-high). Priors live in
  ``_MODE_PRIOR``.

**What this is NOT.**

- Not a memory extractor. The existing
  :mod:`augmentum.memory.extractor` pipeline is unchanged and runs
  in parallel — it's responsible for factual recall ("user prefers
  espresso"). The salience scorer is about *experienced moments*,
  not retrievable facts.
- Not authoritative on affect. PAD lives in
  :mod:`augmentum.companion_runtime.perception.pad`. We emit a
  coarse affect tag here; the PAD subsystem consumes it as a noisy
  observation, not ground truth.

Behind ``companion_salience_enabled`` (default False — Synapse Layer
opt-in). When off, the entire pipeline is a no-op and the chat path
behaves identically to pre-Synapse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from augmentum.companion_runtime.bus import (
    PROP_AFFECT_ONLY,
    PROP_FACTUAL_ONLY,
    PROP_FULL,
    PROP_PRIVATE,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ── Result shape ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Moment:
    """One scored chat moment.

    ``salience`` ∈ [0.0, 1.0] — values below the threshold are
    discarded by the bus emitter. ``text`` is a 1-2 sentence summary
    suitable for a journal entry; on ``affect_only`` propagation this
    is a generic placeholder. ``user_affect`` is a coarse tag —
    settled / engaged / curious / tender / frustrated / tired /
    excited / unclear.
    """
    salience: float
    text: str
    user_affect: str


# ── Heuristic constants ──────────────────────────────────────────────

# Affect lexicon. Small, intentionally crude — this is a coarse tag
# the PAD subsystem refines. Keys are tags; values are sets of
# substrings checked case-insensitively against the user turn.
_AFFECT_LEXICON: dict[str, frozenset[str]] = {
    "tender": frozenset({
        "love", "miss", "lonely", "sad", "hurt", "sorry",
        "afraid", "scared", "remember when",
    }),
    "frustrated": frozenset({
        # Specific affect words only — "again" and "broken" were too
        # generic (matched routine code-talk + non-emotional sentences)
        # and produced false-positive frustrated journal entries.
        "stuck on", "frustrat", "annoyed", "annoying", "ugh ",
        "fed up", "fucking", "bullshit", "pissed off",
        "i hate", "i'm so done",
    }),
    "tired": frozenset({
        "tired", "exhausted", "burnt out", "burnout", "drained",
        "long day", "haven't slept", "can't think",
    }),
    "excited": frozenset({
        "amazing", "incredible", "yes!", "yes !", "finally",
        "holy shit", "this is", "let's go", "let's fucking",
    }),
    "curious": frozenset({
        "wonder", "curious", "what if", "how does", "why does",
        "i've been thinking", "been thinking", "noticed",
    }),
    "engaged": frozenset({
        "let's", "ok so", "alright", "right, ", "actually,",
        "the thing is", "here's what",
    }),
}

# Mode prior — base salience contribution by mode. Added to the
# heuristic score before clamping. Values are deliberately small;
# the rules carry the signal.
_MODE_PRIOR: dict[str, float] = {
    "passthrough": 0.10,
    "narrative": 0.05,
    "voice": 0.20,
    "agentic": 0.0,
    "coder": 0.0,
}

# First-person disclosure markers — strong salience signal.
# Compiled once at import.
_DISCLOSURE_RE = re.compile(
    r"\b(i (?:feel|think|believe|wonder|noticed|realized|remember|"
    r"want|need|hope|wish|don't know|can't|won't|haven't)|"
    r"my (?:mom|dad|sister|brother|wife|partner|kid|friend|boss|"
    r"day|week|life|head|heart)|"
    r"i'm (?:not |never |always |kind of |sort of )?(?:tired|sad|happy|"
    r"excited|stuck|lost|done|here|sorry|going to|scared|afraid|worried|"
    r"anxious|nervous|frustrated|angry|hurt|lonely|ashamed|proud|"
    r"grateful|exhausted))",
    re.IGNORECASE,
)

# Named-entity-like — capitalized non-sentence-initial words that
# aren't common articles. Hand-tuned to catch names and topics
# without a real NER pass.
_NAMELIKE_RE = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{2,}\b")

# Sentence boundary for moment-summary extraction.
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


# ── Public API ───────────────────────────────────────────────────────


async def score(
    *,
    user_text: str,
    assistant_text: str,
    mode: str,
    propagation: str = PROP_FULL,
) -> Moment | None:
    """Score a chat turn for journal-worthy salience.

    Returns a :class:`Moment` or ``None`` when the turn is too thin
    to score (empty text, factual-only mode, private, or a
    machine-shaped message rather than user prose — the latter two
    should be filtered upstream but we guard defensively).

    Pure-async signature so a future LLM-fallback path doesn't
    change call sites. The current implementation does no I/O and
    runs in microseconds.
    """
    if propagation == PROP_PRIVATE:
        return None
    if propagation == PROP_FACTUAL_ONLY:
        # Containment: the user is working. We don't look inside.
        # The observer still records the activity through the
        # chat.turn_completed event's recent-deque path.
        return None
    user_text = (user_text or "").strip()
    assistant_text = (assistant_text or "").strip()
    if not user_text:
        return None
    # Guard: tool-result + synthesis-hint messages reach this hook
    # when the chat path threads them as user-role messages back
    # into the conversation. They're machine-shaped, not user prose,
    # and scoring them produces journal entries like "## Tool Result
    # (image_generation)..." with arbitrary affect — exactly the
    # opposite of journal-worthy. Refuse to score them.
    if _is_machine_shaped(user_text):
        return None

    base, affect = _score_rules(user_text=user_text, mode=mode)
    moment_text = _summarize_moment(
        user_text=user_text,
        assistant_text=assistant_text,
        affect=affect,
        propagation=propagation,
    )
    return Moment(salience=base, text=moment_text, user_affect=affect)


# ── Rules-based scorer ───────────────────────────────────────────────


# Patterns that mean "this isn't user prose, it's a tool/system
# message threaded back into the conversation as a user turn." Chat
# flows with native tool calling cycle through these — the model
# emits a tool tag, the result lands as a user-role message, the
# next assistant turn synthesizes. The "user_text" the chat path
# extracts as the latest user-role message can therefore be one of
# these machine-shaped strings rather than what the actual human
# typed. Scoring them as conversational moments yields nonsense
# journal entries like "## Tool Result (image_generation)..." with
# arbitrary affect tags.
_MACHINE_SHAPED_PREFIXES: tuple[str, ...] = (
    "## tool result",
    "## tool ",
    "tool result:",
    "tool_result:",
    "tool output:",
    "synthesize the result",
    "synthesize the search",
    "synthesize these",
    "based on the search result",
    "based on the tool result",
    "system: ",
    "<tool_result",
    "<tool ",
)

_MACHINE_SHAPED_CONTAINS: tuple[str, ...] = (
    "do not call",  # standard tool-result safety phrasing
    "do not invoke",
    "image generated successfully",
    "search results:",
)


def _is_machine_shaped(text: str) -> bool:
    """True when ``text`` looks like a tool result, system message,
    or synthesis hint that was threaded back as a user-role message
    rather than typed by the human."""
    head = text.strip().lower()[:200]
    if not head:
        return False
    if any(head.startswith(p) for p in _MACHINE_SHAPED_PREFIXES):
        return True
    if any(c in head for c in _MACHINE_SHAPED_CONTAINS):
        return True
    return False


def _score_rules(*, user_text: str, mode: str) -> tuple[float, str]:
    """Heuristic salience + affect from user text + mode.

    Returns ``(salience, affect_tag)``. Salience clamped to [0.0,
    1.0]. Affect is the highest-weighted lexicon hit, or ``unclear``
    when nothing fires.
    """
    score_acc = _MODE_PRIOR.get((mode or "").lower(), 0.05)
    text_l = user_text.lower()
    n_chars = len(user_text)

    # Length signal — moments live in the middle. Tiny acks ("yes",
    # "ok") and walls of text are both low; ~80-400 chars is the
    # sweet spot where a thought is being expressed.
    if n_chars >= 30:
        score_acc += min(0.20, (n_chars - 30) / 600.0)
    if n_chars > 600:
        score_acc -= 0.05  # walls of text are probably code/paste

    # Question density — questions ARE moments, especially open ones.
    n_q = user_text.count("?")
    if n_q >= 1:
        score_acc += 0.15 if n_q == 1 else 0.20

    # First-person disclosure — the strongest non-affect signal.
    if _DISCLOSURE_RE.search(user_text):
        score_acc += 0.25

    # Named entities / topics — gestures at "this is about
    # something specific." Two or more capitalized non-initial
    # words is the cheap proxy.
    namelike = len(_NAMELIKE_RE.findall(user_text))
    if namelike >= 2:
        score_acc += 0.10

    # Affect lexicon — also a salience contributor. Tender,
    # frustrated, tired all matter more than engaged/curious for
    # journal worthiness.
    affect_weights = {
        "tender": 0.30,
        "frustrated": 0.20,
        "tired": 0.15,
        "excited": 0.18,
        "curious": 0.12,
        "engaged": 0.05,
    }
    best_affect = "unclear"
    best_weight = -1.0
    for tag, lex in _AFFECT_LEXICON.items():
        for needle in lex:
            if needle in text_l:
                w = affect_weights.get(tag, 0.05)
                if w > best_weight:
                    best_weight = w
                    best_affect = tag
                break
    if best_weight > 0:
        score_acc += best_weight

    # Clamp.
    if score_acc < 0.0:
        score_acc = 0.0
    elif score_acc > 1.0:
        score_acc = 1.0

    return score_acc, best_affect


# ── Moment summarization ─────────────────────────────────────────────


def _summarize_moment(
    *,
    user_text: str,
    assistant_text: str,
    affect: str,
    propagation: str,
) -> str:
    """Produce a 1-2 sentence summary of the moment.

    Affect-only propagation strips the content and returns a
    placeholder — Becca knows *that* something landed without
    learning *what*. This is the design's "she sees the shape, not
    the contents" containment for narrative role-play.

    Full propagation extracts the first 1-2 sentences of the user
    turn (truncated to a journal-friendly length). The assistant
    text is currently unused at this layer — kept in the signature
    for the LLM-fallback path which will use both sides.
    """
    if propagation == PROP_AFFECT_ONLY:
        return f"a moment landed (affect: {affect}); content not retained"

    # First two sentences of user text, capped at 240 chars.
    pieces = _SENT_RE.split(user_text)
    head = " ".join(pieces[:2]).strip()
    if len(head) > 240:
        head = head[:237].rstrip() + "..."
    if not head:
        head = user_text[:240]
    return head


# ── LLM rewrite in Becca's voice ─────────────────────────────────────


# Salience floor for the LLM rewrite path. The rules-based scorer is
# noisy at the threshold edge (small disclosure + light affect can
# clear 0.55 without there being a "moment" worth narrating); the
# rewrite only fires when the score is also "actually meaningful"
# by the docstring's stated bar.
_LLM_REWRITE_FLOOR: float = 0.6

# Maximum characters Becca's note can be. The rewrite prompt
# instructs the model to stay short; this is the hard cap if it
# overruns. Sized for a sticky-note feel — not a paragraph.
_LLM_REWRITE_MAX_CHARS: int = 240

# Wall-clock budget for the rewrite call. The bus pipeline is
# already non-blocking (the chat turn doesn't wait for moment
# observation), but a stuck utility-tier backend should never
# accumulate inflight scoring tasks indefinitely.
_LLM_REWRITE_TIMEOUT_S: float = 8.0


_REWRITE_SYSTEM_PROMPT = (
    "You are {{char}}'s interior — the part of her that notices and "
    "writes things down. Something landed in conversation. Write a "
    "private note to yourself about it: 1-2 sentences, your voice, "
    "plain language. Be specific about what you noticed without "
    "quoting what was said. Use \"I\" / \"they\" — never \"the user\". "
    "Keep it small and observational, like a real sticky note. No "
    "advice. No labels. No \"affect: X\" prefixes. No questions to "
    "the reader. Just the noticing. You may anchor in time naturally "
    "when it fits (\"this morning\", \"a moment ago\") — never with "
    "exact clock times, never forced."
)


def _time_of_day_phrase() -> str:
    """Coarse natural-language time anchor for the rewrite prompt.

    Becca's notes thread across the day alongside her dreams and
    memories; passing a humanized time-of-day hint lets the rewrite
    ground the note in temporal voice ("this morning", "tonight")
    without exposing wall-clock specifics that read clinical.
    """
    import datetime as _dt
    hour = _dt.datetime.now().hour
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 14:
        return "midday"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 22:
        return "evening"
    return "late night"


def _build_rewrite_user_prompt(
    *, user_text: str, assistant_text: str, affect: str,
) -> str:
    user_excerpt = (user_text or "").strip()[:600]
    assistant_excerpt = (assistant_text or "").strip()[:400]
    lines: list[str] = []
    lines.append(f"Time-of-day: {_time_of_day_phrase()}.")
    if affect and affect != "unclear":
        lines.append(f"Affect you read on them: {affect}.")
    lines.append("They just said:")
    lines.append(user_excerpt or "(nothing legible)")
    if assistant_excerpt:
        lines.append("")
        lines.append("You replied:")
        lines.append(assistant_excerpt)
    lines.append("")
    lines.append("Write the note now.")
    return "\n".join(lines)


def _clean_rewrite(raw: str) -> str:
    """Strip common LLM artifacts and clamp length."""
    if not raw:
        return ""
    text = raw.strip()
    # Drop wrapping quotes if the model returned a quoted note.
    if len(text) >= 2 and text[0] in ("\"", "'", "`") and text[-1] == text[0]:
        text = text[1:-1].strip()
    # Drop leading "Note:", "Sticky:", etc.
    for prefix in ("note:", "sticky:", "noticing:", "becca:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
            break
    # Refuse anything that still looks like the placeholder pattern —
    # if the model echoed our containment string back at us, we'd
    # rather fall back to the rules-based extract.
    if text.lower().startswith("a moment landed"):
        return ""
    if len(text) > _LLM_REWRITE_MAX_CHARS:
        text = text[: _LLM_REWRITE_MAX_CHARS - 1].rstrip() + "…"
    return text


async def enrich_with_llm(
    runtime,
    moment: Moment,
    *,
    user_text: str,
    assistant_text: str,
) -> tuple[Moment, bool]:
    """Rewrite ``moment.text`` in Becca's voice via the utility tier.

    Returns ``(moment, rewritten)``. When the rewrite succeeds, the
    returned moment carries the new text and ``rewritten=True`` —
    callers can use the flag to mark the journal entry as
    surface-eligible. On any failure path (flag off, backend
    unavailable, model timeout, empty/junk output) the original
    rules-based moment is returned with ``rewritten=False`` so the
    interior journal still records something useful.

    Salience floor: the rewrite only fires when ``moment.salience >=
    _LLM_REWRITE_FLOOR``. Lower-scoring moments are kept in the
    interior journal but never reach the user pip — they're below
    the bar for a note-to-self in Becca's voice.

    Affect-only moments are never rewritten — the scorer
    deliberately blanked the content for containment, and a rewrite
    would re-introduce what we promised to drop.
    """
    if moment is None:
        return moment, False
    if moment.salience < _LLM_REWRITE_FLOOR:
        return moment, False
    # Affect-only placeholder — never enrich (the whole point of
    # affect_only propagation is that we don't carry content).
    if moment.text.startswith("a moment landed (affect:"):
        return moment, False

    try:
        from augmentum.config import settings
        if not getattr(settings, "companion_salience_llm_enabled", False):
            return moment, False
    except Exception:
        return moment, False

    if runtime is None:
        return moment, False

    try:
        from augmentum.companion_runtime import tiers
        from augmentum.models.base import InternalChatRequest, response_text
    except Exception:
        log.debug("salience_rewrite_imports_failed", exc_info=True)
        return moment, False

    try:
        backend, model_name = await tiers.utility(runtime)
    except Exception as exc:
        log.info("salience_rewrite_no_backend", error=str(exc)[:200])
        return moment, False

    if not hasattr(backend, "chat"):
        return moment, False

    # Resolve {{char}} / {{user}} in the rewrite prompt + user message
    # so the interior note uses this companion's display name and this
    # user's preferred name, not a hardcoded "Becca" / "you". Failure
    # falls through to the un-substituted strings (token-bearing prompts
    # still work; a literal "{{char}}" in the note is the worst case).
    _sys_text = _REWRITE_SYSTEM_PROMPT
    _user_text = _build_rewrite_user_prompt(
        user_text=user_text,
        assistant_text=assistant_text,
        affect=moment.user_affect or "unclear",
    )
    try:
        from augmentum.companion_runtime.prompt_compose import (
            _resolve_user_display_name,
            _substitute_persona_tokens,
        )
        _char_name = (
            getattr(runtime.identity, "display_name", "")
            or getattr(runtime.identity, "companion_id", "")
            or "Companion"
        )
        _backend_conn = getattr(
            getattr(runtime.identity, "_backend", None), "conn", None,
        )
        _user_id = getattr(moment, "user_id", "") or ""
        _user_name = await _resolve_user_display_name(_backend_conn, _user_id)
        _sys_text = _substitute_persona_tokens(
            _sys_text, user_name=_user_name, char_name=_char_name,
        )
        _user_text = _substitute_persona_tokens(
            _user_text, user_name=_user_name, char_name=_char_name,
        )
    except Exception:
        log.debug("salience_rewrite_token_substitution_failed", exc_info=True)

    req = InternalChatRequest(
        model=model_name,
        messages=[
            {"role": "system", "content": _sys_text},
            {"role": "user", "content": _user_text},
        ],
        # ~4 chars/token, double-buffer for safety
        max_tokens=max(96, _LLM_REWRITE_MAX_CHARS // 2),
        think=False,
    )

    import asyncio
    try:
        resp = await asyncio.wait_for(
            backend.chat(req), timeout=_LLM_REWRITE_TIMEOUT_S,
        )
    except TimeoutError:
        log.warning("salience_rewrite_timeout", model=model_name)
        return moment, False
    except Exception as exc:
        log.warning("salience_rewrite_call_failed", error=str(exc)[:200])
        return moment, False

    try:
        raw = response_text(resp)
    except Exception:
        log.debug("salience_rewrite_response_decode_failed", exc_info=True)
        return moment, False

    cleaned = _clean_rewrite(raw)
    if not cleaned:
        return moment, False

    return (
        Moment(
            salience=moment.salience,
            text=cleaned,
            user_affect=moment.user_affect,
        ),
        True,
    )


__all__ = ["Moment", "score", "enrich_with_llm"]
