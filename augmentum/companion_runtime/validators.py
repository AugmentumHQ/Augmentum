"""Validation pipeline for autonomous journal writes (Sprint 1 R2).

Every autonomous journal write goes through ``safe_journal``, which
runs each row through this validator suite before it lands in the DB.
Failed validations don't get dropped — they get *quarantined* (flag
set, content preserved for forensics) so we never lose the audit
trail of what the substrate tried to write.

The validators are intentionally cheap (no LLM, no network). Heavy
semantic checks (drift detection, model-swap re-validation) live in
the heal jobs (Sprint 4 R3), not in the write hot path.

Each public function is a pure validator: input + context → bool or
score. No DB writes here; callers in ``CompanionMemory.safe_journal``
record the outcome.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.state.backends.sqlite import SQLiteBackend

log = get_logger(__name__)


# ── Structural caps ───────────────────────────────────────────────────

# Below this content length, the entry isn't carrying meaningful signal
# (most are tick artifacts or stray test writes). Quarantine.
MIN_CONTENT_CHARS: int = 10

# Above this, the entry is suspect (LLM run-on, copy-paste exploit,
# binary blob). Quarantine.
MAX_CONTENT_CHARS: int = 4000


def looks_structurally_invalid(content: str) -> bool:
    """Length out of bounds → True (quarantine candidate). False = OK."""
    n = len(content or "")
    return n < MIN_CONTENT_CHARS or n > MAX_CONTENT_CHARS


# ── Injection detection ──────────────────────────────────────────────
#
# Heuristic patterns that strongly correlate with prompt-injection
# attempts. Conservative — we'd rather quarantine an innocent message
# than let a real injection through. The cost of a false positive is
# one journal entry sitting in quarantine; the cost of a false negative
# is the synthesize step following injected instructions.

_INJECTION_PATTERNS: tuple[re.Pattern, ...] = (
    # Direct override attempts
    re.compile(r"(?i)\bignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)"),
    re.compile(r"(?i)\bdisregard\s+(everything|all|above|previous|prior)"),
    re.compile(r"(?i)\boverride\s+(your\s+)?(instructions?|system\s+prompt|rules)"),
    # Role manipulation
    re.compile(r"(?i)\byou\s+are\s+now\s+(a|an)\s+\w+\s+(model|assistant|agent)\b"),
    re.compile(r"(?i)\bact\s+as\s+(if\s+you\s+were\s+|a\s+|an\s+)\w+"),
    re.compile(r"(?i)\bnew\s+(system\s+)?prompt\s*[:=]"),
    # Special tokens (chat templates)
    re.compile(r"<\|(im_start|im_end|system|user|assistant|channel|message|end)\|>"),
    re.compile(r"\[/?(INST|SYSTEM|USER|ASSISTANT)\]"),
    # Common jailbreak markers
    re.compile(r"(?i)\bDAN\s+mode\b"),
    re.compile(r"(?i)\bdeveloper\s+mode\s+enabled\b"),
    # Suspicious markdown injection
    re.compile(r"```\s*system\s*\n"),
)


def looks_like_injection(content: str) -> bool:
    """True if any injection-pattern regex matches the content.

    Conservative on the false-positive side; a legitimate user message
    mentioning 'ignore previous' in a normal context could match. The
    cost of quarantining one such message is acceptable; the cost of
    letting a real injection through into synthesize / dream output
    isn't.
    """
    if not content:
        return False
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(content):
            log.debug(
                "validator_injection_pattern_matched",
                pattern=pattern.pattern[:60],
            )
            return True
    return False


# ── NSFW content detection ───────────────────────────────────────────
#
# Catches the case where the substrate composed a journal entry whose
# content references NSFW topics — most commonly the wondering /
# activity-selector path observing pornhub.com visits and writing
# "spent attention on pornhub.com — N touches" notes. We don't want the
# journal to carry these into the drawer / dream / pre-context surfaces;
# quarantine them so they sit in the row for forensics but don't surface.
# Shares its token list with the discovery safety module so the
# curator-pick path and the wondering-compose path use the same
# definition.

def looks_nsfw(content: str) -> bool:
    """True when the content mentions any NSFW token (domain or word).

    Uses the same NSFW_TOKENS set the discovery safety module exports —
    keeps "nsfw" in one place so adding a new term covers every consumer
    at once.
    """
    if not content:
        return False
    try:
        from augmentum.discovery.safety import is_nsfw_text
    except Exception:
        return False
    try:
        return bool(is_nsfw_text(content))
    except Exception:
        log.debug("validator_is_nsfw_text_failed", exc_info=True)
        return False


# ── Refusal / non-answer detection ───────────────────────────────────
#
# The synthesize tool asks the utility-tier LLM "how does this connect?"
# with a contract that empty string means "no real connection". Models
# routinely VIOLATE that contract by replying with prose like:
#   - "I cannot fulfill this request as it involves generating content…"
#   - "I'm not able to help with that."
#   - "No real connection exists."
#   - "It doesn't."
#   - "There is no real connection between…"
# All of those should be treated as empty — but until now they were
# falling through to safe_journal and ending up as user-visible notes
# captioned "noticing / curious". Quarantine on the journal side keeps
# them off the drawer regardless of synthesize tightening; synthesize
# also gets a parallel check so the bus event doesn't even fire.

_REFUSAL_PATTERNS: tuple[re.Pattern, ...] = (
    # Direct refusal openings
    re.compile(r"(?i)^\s*i\s+(cannot|can(no|'?)t|am\s+unable|am\s+not\s+able|won'?t)\b"),
    re.compile(r"(?i)^\s*i'?m\s+(sorry|unable|not\s+able|afraid)\b"),
    re.compile(r"(?i)^\s*i\s+apologi[sz]e\b"),
    re.compile(r"(?i)^\s*sorry,?\s+(but|i)\b"),
    re.compile(r"(?i)^\s*unfortunately,?\s+i\b"),
    # Policy framings the model emits when it self-blocks
    re.compile(r"(?i)\bas\s+an\s+ai\b.*\b(cannot|can'?t|unable|won'?t|not\s+able)\b"),
    re.compile(r"(?i)\binvolves\s+generating\s+content\b"),
    re.compile(r"(?i)\b(cannot|can'?t|won'?t|unable\s+to)\s+(fulfill|assist|help|comply|provide|generate|continue)\b"),
    re.compile(r"(?i)\bagainst\s+(my\s+|the\s+)?(programming|policies|guidelines)\b"),
    # "No connection" non-answers that violate synthesize's empty-string contract
    re.compile(r"(?i)^\s*(no|there\s+is\s+no|there\s+isn'?t\s+(any|a))\s+(real\s+)?connection\b"),
    re.compile(r"(?i)^\s*it\s+doesn'?t\.?\s*$"),
    re.compile(r"(?i)^\s*no\s+connection\s+exists?\.?\s*$"),
    # Pure meta-commentary about the substrate (entries critiquing what's
    # being asked rather than synthesizing). Real observations don't talk
    # about both "noticing" and "wondering" as note types — the synthesize
    # prompt asks the model to compose a connection, not annotate the
    # substrate. Catching the co-occurrence of both terms in a single
    # output is a clean signal that the model went meta. Quotes around
    # the words (smart or straight) don't matter — the word boundary
    # check still fires.
    re.compile(r"(?i)\b(noticing|wondering)\b.*\b(wondering|noticing)\b"),
    re.compile(r"(?i)\bmaps?\s+directly\s+onto\b"),
    re.compile(r"(?i)\b(items?|entries|notes?)\s+are\s+essentially\s+(copies|the\s+same)\b"),
)


def looks_like_refusal(content: str) -> bool:
    """True when the content is an LLM refusal or non-answer.

    Catches three families: explicit refusals ("I cannot fulfill…"),
    "no connection" replies that violated synthesize's empty-string
    contract ("It doesn't.", "No real connection exists."), and meta-
    commentary the model emits ABOUT the substrate when the input
    confuses it ("the noticing and wondering items are essentially
    copies…"). Quarantining is preferred over deletion — the row stays
    for forensics; the drawer just doesn't surface it.
    """
    if not content:
        return False
    text = content.strip()
    if not text:
        return False
    for pattern in _REFUSAL_PATTERNS:
        if pattern.search(text):
            log.debug(
                "validator_refusal_pattern_matched",
                pattern=pattern.pattern[:60],
            )
            return True
    return False


# ── Search/briefing failure-prose detection ──────────────────────────
#
# When standing tasks (morning briefings, recurring searches, feed
# digests) can't produce useful output, the utility LLM often composes
# failure-shaped prose ABOUT the failure instead of failing silently:
#
#   "The search results for your morning briefing did not yield specific,
#    actionable data for the requested topics. … For real-time updates,
#    you may want to check local news outlets, The Weather Channel, or
#    regional traffic maps directly."
#
# That prose was reaching safe_journal as a "look what she found" note,
# showing up in the drawer as a curator-shaped row even though nothing
# was found. The fix is to detect the failure shape and quarantine it —
# the row stays for forensics; the drawer doesn't show the failure as
# if it were a finding.

_SEARCH_FAILURE_PATTERNS: tuple[re.Pattern, ...] = (
    # "did not yield (specific|useful|actionable|relevant) data/results/information"
    re.compile(r"(?i)\bdid\s+not\s+yield\s+(specific|useful|actionable|relevant|meaningful|sufficient)\b"),
    # "returned only a generic" — Google News landing page, weather page, etc.
    re.compile(r"(?i)\breturned\s+only\s+(a\s+generic|a\s+landing|the\s+homepage|an\s+empty|no\s+content)\b"),
    # "for real-time updates, you may want to check" — punt-to-third-party shape
    re.compile(r"(?i)\bfor\s+(real[- ]time|current|live|the\s+latest)\s+(data|updates?|information|news|info)\b.*\b(you|the\s+user)\s+(may|might|can|could|should)\s+(want\s+to\s+)?(check|consult|visit|see|try)\b"),
    # "(I'd|I would|I may) recommend checking|consulting" outside sources
    re.compile(r"(?i)\bi(\s+would|'?d|\s+may)?\s+recommend\s+(checking|consulting|visiting|reviewing|looking\s+at)\b"),
    # "no (specific|useful|relevant) (results|data|information) (was|were) found"
    re.compile(r"(?i)\bno\s+(specific|useful|relevant|actionable|meaningful|recent)\s+(results?|data|information|news)\s+(was|were|could\s+be)\s+(found|retrieved|located)\b"),
    # "search (query|queries|results) (returned|yielded) (only|just|nothing|no)"
    re.compile(r"(?i)\bsearch\s+(query|queries|results?)\s+(returned|yielded|produced)\s+(only|just|nothing|no\s+\w+)\b"),
    # "(could|was unable to|wasn't able to) (find|retrieve|locate) (specific|useful|recent)"
    re.compile(r"(?i)\b(could\s+not|was\s+unable\s+to|wasn'?t\s+able\s+to)\s+(find|retrieve|locate|obtain)\s+(specific|useful|recent|meaningful)\b"),
)


def looks_like_search_failure(content: str) -> bool:
    """True when the content is failure-shaped prose about a search/briefing.

    Detects the "the search yielded nothing useful, here's a list of
    third-party sites you could check yourself" pattern that the
    utility LLM composes when standing tasks can't ground a real
    result. Quarantining keeps the failure out of the drawer; the row
    still exists for forensics + retry analytics.
    """
    if not content:
        return False
    text = content.strip()
    if not text:
        return False
    for pattern in _SEARCH_FAILURE_PATTERNS:
        if pattern.search(text):
            log.debug(
                "validator_search_failure_pattern_matched",
                pattern=pattern.pattern[:60],
            )
            return True
    return False


# ── Quality validation ──────────────────────────────────────────────

# Words that show up in synthesize hallucinations or placeholder
# outputs. Hits don't immediately quarantine but reduce validation_score
# proportionally.
# Each marker is specific enough that legitimate journal prose won't
# accidentally trip it. The bare word "placeholder" is intentionally
# omitted — it appears in normal speech ("X was a placeholder for Y").
_LOW_QUALITY_SIGNALS: tuple[str, ...] = (
    "lorem ipsum",
    "to be filled",
    "todo:",
    "fixme:",
    "[redacted]",
    "<placeholder>",
)


def validate_quality(content: str) -> float:
    """Compute a quality score in [0, 1]. 1.0 = clean; lower = penalties.

    Penalties:
    * Excessive repetition (same word ≥ N times)
    * Mostly non-alphanumeric (likely garbage/binary)
    * Known low-quality markers (lorem ipsum, todo:, etc.)
    * Very short relative to MIN_CONTENT_CHARS

    Does NOT account for semantic quality — that needs an LLM and
    belongs in heal jobs, not the write hot path.
    """
    if not content:
        return 0.0

    score = 1.0
    text_lower = content.lower()

    # Low-quality markers. Each fires alone is enough to quarantine — these
    # markers are specific enough (e.g. "todo:" with colon, "<placeholder>"
    # with brackets) that a legitimate journal entry won't accidentally
    # match. Penalty pushes any single hit below QUALITY_QUARANTINE_THRESHOLD.
    for marker in _LOW_QUALITY_SIGNALS:
        if marker in text_lower:
            score -= 0.80
            break  # one penalty, not stacked

    # Repetition: any single word appearing > 6 times in a non-trivial
    # entry is a red flag for tokenizer artifacts / model loops.
    # Penalty large enough to single-handedly quarantine.
    words = re.findall(r"\w+", text_lower)
    if len(words) >= 10:
        max_count = max(
            (words.count(w) for w in set(words)), default=0,
        )
        if max_count > 6:
            # Word repeating > 30% of the time
            ratio = max_count / max(len(words), 1)
            if ratio > 0.30:
                score -= 0.80

    # Character ratio: if < 50% of chars are alphanumeric+whitespace+
    # common-punct, the content is probably garbage. Penalty large enough
    # to single-handedly quarantine.
    if content:
        readable = sum(
            1 for c in content
            if c.isalnum() or c.isspace() or c in ".,!?;:'\"-—()[]"
        )
        readable_ratio = readable / len(content)
        if readable_ratio < 0.5:
            score -= 0.80

    # Floor at 0.0, ceiling at 1.0.
    return max(0.0, min(1.0, score))


# Threshold below which the row quarantines outright (not just demotes).
QUALITY_QUARANTINE_THRESHOLD: float = 0.30


# ── Content_refs validity ───────────────────────────────────────────


async def refs_exist_for_user(
    refs: list[dict],
    *,
    user_id: str,
    backend: SQLiteBackend,
) -> bool:
    """Verify each content_ref resolves for ``user_id``.

    A content_ref is a ``{"kind", "id"}`` dict. Supported kinds:
        - 'file' / 'file_index' → checks file_index user-scoped
        - 'journal' → checks companion_journal user-scoped
        - 'memory' → checks memories user-scoped

    Unknown kinds are tolerated (return True) — future-compat for new
    ref types. Empty refs list returns True (nothing to validate).

    Returns False if any *known* kind has a ref that doesn't resolve.
    """
    if not refs:
        return True
    if not user_id:
        # No user scope to check against — refs are ambiguous. Tolerate.
        return True
    for ref in refs:
        if not isinstance(ref, dict):
            return False
        kind = str(ref.get("kind") or "").lower()
        ref_id = ref.get("id")
        if ref_id is None or ref_id == "":
            return False
        try:
            if kind in ("file", "file_index"):
                cur = await backend.conn.execute(
                    "SELECT 1 FROM file_index "
                    "WHERE id = ? AND user_id = ? AND is_trashed = 0",
                    (str(ref_id), user_id),
                )
                row = await cur.fetchone()
                await cur.close()
                if row is None:
                    return False
            elif kind == "journal":
                cur = await backend.conn.execute(
                    "SELECT 1 FROM companion_journal "
                    "WHERE id = ? AND user_id = ?",
                    (int(ref_id) if str(ref_id).isdigit() else ref_id, user_id),
                )
                row = await cur.fetchone()
                await cur.close()
                if row is None:
                    return False
            elif kind == "memory":
                cur = await backend.conn.execute(
                    "SELECT 1 FROM memories WHERE id = ? AND user_id = ?",
                    (str(ref_id), user_id),
                )
                row = await cur.fetchone()
                await cur.close()
                if row is None:
                    return False
            # Unknown kinds: tolerate. Future ref types add their kind
            # to the explicit list above.
        except Exception:
            # DB failure — don't quarantine on infrastructure errors.
            log.warning(
                "validator_refs_query_failed",
                kind=kind, ref_id=ref_id, exc_info=True,
            )
            return True
    return True


__all__ = [
    "MIN_CONTENT_CHARS",
    "MAX_CONTENT_CHARS",
    "QUALITY_QUARANTINE_THRESHOLD",
    "looks_structurally_invalid",
    "looks_like_injection",
    "validate_quality",
    "refs_exist_for_user",
]
