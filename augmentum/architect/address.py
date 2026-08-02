"""Address classifier — decide whether a transcribed utterance was
directed at the companion, without depending on a specific wake-word.

In always-listening mode the mic is continuously open and the server
transcribes every utterance the speaker-verified user produces. Most
of those utterances are NOT directed at Becca — they're conversation
with another person, self-talk while reading, or background mumbling.

This classifier separates the two with structural heuristics that
don't rely on a name match (so the companion is renameable / multi-
persona-safe, and natural addressing like "can you play some jazz"
still fires without anyone saying "Becca").

Five signals, combined with a confidence score the dispatcher uses
to decide between (a) routing to the architect, (b) treating as
ambient observation, or (c) ignoring outright.

The classifier is pure — no I/O, no global state — so it's trivially
testable and cheap to evolve. The companion's address policy is one
function call away from any future tier-2 / tier-3 classifier
(embedding similarity, tiny LLM zero-shot) that lands on top.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# How long after Becca's last TTS we treat a fresh utterance as a
# "continuation" rather than a fresh address. Tuned to typical follow-up
# rhythm — long enough for the user to react, short enough that random
# background speech doesn't get conversationally promoted.
_CONTINUATION_WINDOW_S = 30.0


# Default decision thresholds. The dispatcher uses these; callers can
# pass overrides for stricter / more lenient modes.
DEFAULT_ADDRESS_THRESHOLD = 0.85
DEFAULT_AMBIENT_FLOOR = 0.40  # below this, treat as background noise


# ---------------------------------------------------------------------------
# Patterns — all anchored, all name-free, all case-insensitive.
# Verb lists deliberately overlap with the architect's primitive trigger
# verbs so commands that already fire a primitive register here too.
# ---------------------------------------------------------------------------

# Imperative verb stems — matched in any inflection (generate /
# generating / generated). Direct commands that, when said with no
# other context, overwhelmingly target an assistant.
#
# Stems are matched with a tolerant suffix pattern so we cover
# present + gerund + past tense without enumerating every form:
#   play(s|ed|ing)?   show(s|ed|ing|n)?   generat(e|es|ed|ing)
# Stems ending in 'e' get the e-dropping form for gerund/past
# (generate → generating, create → created). The matcher handles
# both shapes via the suffix alternation below.
_IMPERATIVE_STEMS = (
    "play", "put", "show", "tell", "find", "search", "make", "draw",
    "generat", "creat", "open", "clos", "stop", "paus", "resum",
    "read", "remind", "set", "queue", "skip", "next", "previous",
    "repeat", "note", "sav", "remember", "forget", "send", "call",
    "start", "cancel", "help", "describ", "explain", "list", "giv",
)
_IMPERATIVE_VERB_GROUP = (
    rf"(?:{'|'.join(_IMPERATIVE_STEMS)})"
    r"(?:e|es|ed|ing|s|n)?"
)

# Discourse preamble — any combination of polite / hesitation / meta
# tokens that lead INTO an imperative verb. Allows up to 4 such tokens
# so "Hey there, try" + verb works ("Hey", "there", ",", "try"
# semantically; the regex collapses them as discourse). Keeps the
# verb match anchored at start-of-utterance from the user's
# perspective, just tolerant of natural lead-in.
_DISCOURSE_TOKEN = (
    r"(?:hey|please|alright|okay|ok|so|listen|sorry|um+|uh+|"
    r"there|then|just|maybe|actually|wait|now|try(?:ing)?|"
    r"let'?s|let\s+me|let\s+me\s+see\s+if|"
    r"can\s+you|could\s+you|would\s+you|will\s+you|"
    r"do\s+you|did\s+you|are\s+you|will\s+you)"
)
_IMPERATIVE_START = re.compile(
    rf"^\s*(?:{_DISCOURSE_TOKEN}[,\s]+){{0,4}}{_IMPERATIVE_VERB_GROUP}\b",
    re.IGNORECASE,
)

# Conversational lead-in — 0-3 short tokens (greetings, hedges,
# vocative names) before the canonical addressing anchor. The
# 12-char-or-less cap rules out long content nouns from forming a
# false lead-in. Used by _SECOND_PERSON_QUESTION and
# _WH_QUESTION_OPENER below so natural openers like "Hey there,",
# "Hi Becca,", "Ok so,", "Hello love," all flow through. Strong-
# ambient patterns (_SELF_TALK_PREFIX, _THIRD_PERSON_NARRATION)
# are checked first in is_addressed() so they take precedence over
# this permissive lead-in.
_LEAD_IN = r"(?:[a-z']{1,12}[,\s]+){0,3}"

# "Can you X?", "Could you Y?", "Do you know Z?" — 2nd-person question
# directed at the listener. The lead-in absorbs any greeting +
# optional vocative ("Hey there,", "Hi Becca,", "Yo,") before the
# modal verb.
_SECOND_PERSON_QUESTION = re.compile(
    rf"^\s*{_LEAD_IN}"
    r"(?:can|could|would|will|do|did|are|is|should|have|has)\s+you\b",
    re.IGNORECASE,
)

# WH-question containing "you" anywhere in the same clause —
# "What are some things you can do", "How do you handle X",
# "Where can I find that file". Catches natural addressing where the
# WH word comes first and "you" appears later in the sentence (which
# my anchored _SECOND_PERSON_QUESTION misses). The "same clause"
# constraint ([^.!?] between WH and "you") avoids cross-sentence
# false positives — STT often joins a closing "see you there." with a
# subsequent "What are..." question, and we still want to match the
# question half on its own.
_WH_QUESTION_WITH_YOU = re.compile(
    r"\b(?:what|how|where|when|why|which|who)\b[^.!?]*?\byou\b",
    re.IGNORECASE,
)

# Pure WH-question opener — "what time is it", "what's the weather",
# "how's the traffic". No "you" required, but the structure is
# clearly a request for information. Lower confidence than the
# variants with explicit "you", but still above the ambient floor.
# Same lead-in as _SECOND_PERSON_QUESTION so "Hey there, what time
# is it" / "Hi, what's the weather" land too.
_WH_QUESTION_OPENER = re.compile(
    rf"^\s*{_LEAD_IN}"
    r"(?:what(?:'?s|\s+is|\s+are|\s+was|\s+were|\s+time|\s+about)|"
    r"how(?:'?s|\s+is|\s+are|\s+do|\s+does|\s+can|\s+should|\s+about|\s+much|\s+many|\s+long|\s+far)|"
    r"where(?:'?s|\s+is|\s+are|\s+can|\s+do|\s+did)|"
    r"when(?:'?s|\s+is|\s+are|\s+can|\s+do|\s+did|\s+should)|"
    r"why(?:'?s|\s+is|\s+are|\s+do|\s+did|\s+should)|"
    r"which|who(?:'?s|\s+is|\s+are))\b",
    re.IGNORECASE,
)

# "Tell me X", "Show me Y", "Find me Z" — explicit request-for-help
# pattern. Matches anywhere in the utterance, not just the start.
_DIRECT_REQUEST = re.compile(
    r"\b(?:tell|show|find|play|read|remind|give|send|"
    r"help|explain|describe)\s+me\b",
    re.IGNORECASE,
)

# Bare continuation tokens — only count as "addressed" when Becca
# spoke recently. Outside the continuation window these are pure
# self-talk.
_CONTINUATION_PATTERN = re.compile(
    r"^\s*(?:again|louder|softer|next|previous|stop|pause|"
    r"the\s+(?:other|next|last)|no(?:[,]|\s+(?:the|wait))|"
    r"yes(?:[,]|\s)|maybe\s+(?:not|so)|sorry,?)\b",
    re.IGNORECASE,
)

# Self-talk markers — 1st-person reflection, hesitation noises,
# musing. These down-rate even if a request marker happens to also
# match (e.g. "I should remind me to do that" is musing, not a
# request). Anchored at start to avoid false negatives on legitimate
# addresses that happen to contain "I" mid-sentence.
_SELF_TALK_PREFIX = re.compile(
    r"^\s*(?:i\s+(?:think|should|was|am|wonder|don'?t|do|need|"
    r"want|guess|hope|feel|like|gotta|have\s+to|need\s+to)|"
    r"hmm|huh|um+|uh+|let\s+me\s+(?:think|see|check)|"
    r"maybe|probably|actually,?|wait,?)\b",
    re.IGNORECASE,
)

# Embedded address override — when a self-talk-looking prefix wraps an
# actual delegation marker ("let me see if you can X", "I wonder if
# you could Y"), the embedded 2nd-person request signal flips the
# classification back to addressed. Keeps the conversational hedge
# without losing the request.
_EMBEDDED_ADDRESS = re.compile(
    r"\b(?:if\s+)?you\s+(?:can|could|would|should|might)\b",
    re.IGNORECASE,
)

# 3rd-person conversational markers — "she said X", "they did Y",
# "what did John mean by". Down-rates because the utterance is most
# likely directed at another human, not the assistant.
# 3rd-person narration about a PERSON other than the user/companion.
# Deliberately excludes "it": "it" is a topic pronoun, not a person —
# "it was about the narrative chat", "it's raining", "it was a long
# day" are conversational sharing/continuations directed AT the
# companion, not narration about someone else. Including "it" mis-drops
# them as ambient on both the Tier-1 and fallback paths (a coherent
# one-on-one turn → goal=drop, "she ignored me", 2026-06-13). Keep the
# genuine third-person persons (he/she/they/named) where the down-rate
# is right — those usually ARE directed at another human.
_THIRD_PERSON_NARRATION = re.compile(
    r"^\s*(?:he|she|they|john|mary)\s+(?:said|did|was|is|are|"
    r"were|told|asked|wanted|thinks)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AddressDecision:
    """Result of running the classifier on an utterance.

    ``addressed`` is True when the dispatcher should route this to the
    architect. ``confidence`` is the strongest matching signal's score
    (0.0-1.0). ``signal`` names the cue that fired so logs / tests can
    introspect ("imperative_start", "continuation", etc.).
    """

    addressed: bool
    confidence: float
    signal: str


def is_addressed(
    transcript: str,
    *,
    last_tts_ended_at: float | None = None,
    now: float | None = None,
    address_threshold: float = DEFAULT_ADDRESS_THRESHOLD,
) -> AddressDecision:
    """Decide whether ``transcript`` was directed at the companion.

    Pure function — no global state, no I/O. The companion runtime
    calls this once per finalized STT utterance in always-listening
    mode; everything below the threshold falls into the ambient sink.

    Time-of-day is irrelevant here; ``last_tts_ended_at`` + ``now`` are
    wall-clock seconds (e.g. ``time.time()``) used only to detect
    whether the user is responding to something Becca just said.

    The decision rule (highest-priority match wins):

      1. Self-talk / hesitation marker → ambient (low confidence)
      2. 3rd-person narration → ambient (low confidence)
      3. Imperative verb start → addressed (very high confidence)
      4. 2nd-person question form → addressed (high confidence)
      5. Direct request marker ("tell me", "show me") → addressed
      6. Continuation token (within window) → addressed
      7. Default → ambient (no signal fired)
    """
    text = (transcript or "").strip()
    if not text:
        return AddressDecision(False, 0.0, "empty")

    if _SELF_TALK_PREFIX.match(text):
        # Embedded delegation override — "let me see if you can play X"
        # looks like self-talk but the "you can" reveals a request.
        if _EMBEDDED_ADDRESS.search(text):
            return AddressDecision(True, 0.88, "self_talk_with_delegation")
        return AddressDecision(False, 0.10, "self_talk")
    if _THIRD_PERSON_NARRATION.match(text):
        return AddressDecision(False, 0.15, "third_person")

    is_continuation = (
        last_tts_ended_at is not None
        and now is not None
        and (now - last_tts_ended_at) < _CONTINUATION_WINDOW_S
    )

    if _IMPERATIVE_START.match(text):
        conf = 0.99 if is_continuation else 0.95
        return AddressDecision(conf >= address_threshold, conf, "imperative_start")

    if _SECOND_PERSON_QUESTION.match(text):
        return AddressDecision(True, 0.92, "second_person_question")

    if _DIRECT_REQUEST.search(text):
        return AddressDecision(True, 0.90, "direct_request")

    # WH-question that mentions "you" later in the sentence —
    # "What are some things you can do", "How do you do X". Strong
    # addressing signal even without start-anchored 2nd-person form.
    if _WH_QUESTION_WITH_YOU.search(text):
        return AddressDecision(True, 0.90, "wh_question_with_you")

    # Bare WH-question opener — "what time is it", "how's the
    # weather". No explicit "you" but the structure is a request.
    # Lower confidence; tunable via threshold.
    if _WH_QUESTION_OPENER.match(text):
        return AddressDecision(0.85 >= address_threshold, 0.85, "wh_question_opener")

    if is_continuation and _CONTINUATION_PATTERN.match(text):
        return AddressDecision(True, 0.85, "continuation")

    return AddressDecision(False, 0.30, "no_signal")
