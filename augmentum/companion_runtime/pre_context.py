"""Pre-context injection — make the relationship feel continuous.

Sprint 3 Piece 10, Aletheia × Augmentum arc.

When a chat session starts (the user's first message), scan unsurfaced
notes for content_ref overlap with the user's message. If a strong
match exists, inject ONE note as a system note before generation.
This is what makes "Pull it together" feel continuous: her reply
already knows.

Constraints:

* **One injection per session.** Re-entering an existing session does
  NOT re-inject. Continuity preserves.
* **Strong match only.** Either a content_ref appears verbatim in the
  message (rare but cheapest) or ≥2 of the note's keywords overlap
  with extracted message keywords (cheap heuristic, no embedding call).
* **Surface-tagging.** The injection is wrapped in a clear marker so
  downstream readers can distinguish injected context from user input.

Used by chat handlers at session-start hook.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# Tunable knobs — settings keys default to these.
DEFAULT_MIN_KEYWORD_OVERLAP: int = 2
DEFAULT_MAX_NOTES_TO_SCAN: int = 10


async def maybe_inject_notes_context(
    runtime: CompanionRuntime,
    *,
    user_id: str,
    first_message: str,
    session_id: str = "",
) -> str | None:
    """Return injectable system-note text or None.

    Returns None when no strong match exists, when the kill switch is
    off, or when an injection has already been recorded for this
    session (when session_id is provided and tracking is wired).

    Callers (chat handlers) wrap the returned text in their preferred
    system-message envelope and prepend it to the message list before
    inference.
    """
    if not user_id or not first_message:
        return None

    from augmentum.config import settings
    if not getattr(settings, "companion_pre_context_enabled", False):
        return None

    # Sprint 5 — presence_mode gate. Pre-context injection is the
    # 'engaged' tier; gentle + silent never inject.
    from augmentum.companion_runtime import presence_mode as _pm
    if not _pm.pre_context_allowed():
        return None

    backend = runtime.backend
    min_overlap = int(
        getattr(settings, "companion_pre_context_min_keyword_overlap",
                DEFAULT_MIN_KEYWORD_OVERLAP),
    )
    max_scan = int(
        getattr(settings, "companion_pre_context_max_notes_scan",
                DEFAULT_MAX_NOTES_TO_SCAN),
    )

    # Pull recent unsurfaced notes. Index `idx_cj_quiet_share_ready`
    # (mig 178) makes this O(eligible-notes).
    try:
        cur = await backend.conn.execute(
            """
            SELECT id, content, content_refs
            FROM companion_journal
            WHERE companion_id = ?
              AND user_id = ?
              AND quiet_share_ready = 1
              AND surfaced_at IS NULL
              AND COALESCE(quarantined, 0) = 0
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (runtime.companion_id, user_id, max_scan),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception:
        log.warning("pre_context_query_failed", exc_info=True)
        return None

    if not rows:
        return None

    message_lower = first_message.lower()
    message_keywords = _extract_message_keywords(first_message)

    best_match: tuple[int, str, list, int] | None = None  # (note_id, content, refs, score)

    for row in rows:
        note_id, content, refs_json = row
        try:
            refs = json.loads(refs_json or "[]")
        except (json.JSONDecodeError, TypeError):
            refs = []

        # Score the match. Two signals:
        # 1. Direct id mention in message (rare, but the cleanest hit)
        # 2. Keyword overlap with note content

        # Signal 1: any ref id appears in message?
        id_hit = False
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            ref_id = str(ref.get("id") or "")
            if ref_id and len(ref_id) >= 4 and ref_id.lower() in message_lower:
                id_hit = True
                break

        # Signal 2: keyword overlap
        note_keywords = _extract_message_keywords(content)
        overlap = len(message_keywords & note_keywords)

        # Either signal alone qualifies if strong enough; both combined
        # is stronger.
        score = (overlap or 0) + (5 if id_hit else 0)
        # Threshold: keyword overlap must reach min_overlap, OR id_hit
        if id_hit or overlap >= min_overlap:
            if best_match is None or score > best_match[3]:
                best_match = (note_id, content, refs, score)

    if best_match is None:
        return None

    note_id, content, refs, score = best_match
    log.info(
        "pre_context_injection",
        user_id=user_id, note_id=note_id, score=score,
        session_id=session_id,
    )
    return _format_injection(note_id, content, refs)


def _extract_message_keywords(text: str) -> set[str]:
    """Lowercase non-stopword tokens ≥4 chars. Same logic as the topical
    aggregator's keyword extractor; centralizing it here so callers
    don't have to import the aggregator."""
    import re
    if not text:
        return set()
    stopwords = {
        "the", "a", "an", "and", "or", "but", "of", "on", "in", "with",
        "to", "for", "is", "this", "that", "from", "by", "at", "as",
        "if", "it", "we", "you", "are", "was", "were", "been", "have",
        "has", "had", "will", "would", "could", "should", "may", "can",
        "did", "https", "http", "www", "com", "org", "what", "where",
        "when", "how", "why", "who", "your", "their", "there", "here",
    }
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower())
    return {t for t in tokens if t not in stopwords}


def _format_injection(note_id: int, content: str, refs: list) -> str:
    """Wrap the note as a system-readable context block.

    Format chosen for easy LLM parsing while marked clearly as
    Becca-provided context, not user input.
    """
    ref_names = []
    for ref in refs[:3]:
        if isinstance(ref, dict):
            rid = ref.get("id")
            kind = ref.get("kind")
            if rid and kind:
                ref_names.append(f"{kind}:{rid}")
    refs_line = (
        f"\n(related: {', '.join(ref_names)})" if ref_names else ""
    )
    return (
        f"[becca's note — written earlier, not yet seen by user]\n"
        f"{content}{refs_line}\n"
        f"[end note]\n\n"
        f"You may reference this naturally if it's relevant; "
        f"otherwise treat it as background only."
    )


__all__ = [
    "maybe_inject_notes_context",
    "DEFAULT_MIN_KEYWORD_OVERLAP",
    "DEFAULT_MAX_NOTES_TO_SCAN",
]
