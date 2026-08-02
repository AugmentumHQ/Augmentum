"""Per-turn search-result deduplication shared across every tool loop.

The web / image / youtube search tools each dedup *within a single call*, but
nothing remembered results across the multiple tool-call ROUNDS of one user turn
— so a chatty model (DeepSeek especially) would get the same image/video/page
back 4-5 times across rounds, see them re-listed in its context, and search
again. This module gives every loop (passthrough chat, companion native loop,
UARF analytical auto-search) one shared, per-turn memory of what was already
returned, so each round only surfaces genuinely NEW results.

Mechanism: a ``contextvars.ContextVar`` holding a ``TurnSearchDedup``. A loop
sets it at the start of a turn (and resets it after); the search tools, running
in the same async context, consult it inside ``execute()`` to skip
already-shown results (and backfill with fresh ones). No signature threading —
tools opt in by reading the contextvar, and a tool called outside any loop
(contextvar unset) behaves exactly as before.

The same ``total_new`` counter also powers the loops' productivity guard: a
round that calls a search tool but adds zero new results is "spinning" and the
loop can stop early instead of burning rounds on repeats.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field

# Tool names whose results participate in cross-round dedup + the productivity
# guard. (``web`` is the chain/agentic web tool; ``web_search`` the direct one.)
SEARCH_TOOL_NAMES: frozenset[str] = frozenset(
    {"web_search", "web", "image_search", "youtube"}
)


def is_search_round(calls) -> bool:
    """True iff this round did genuine SEARCH work — the signal the loops'
    productivity guard uses to detect spinning.

    A ``youtube`` call carrying a URL/video-id is a transcript FETCH, not a
    search: it legitimately surfaces no new *search* result yet is real progress
    (the model is reading a video it already found). Counting it as a search
    would let the "two rounds with nothing new → stop" guard guillotine a model
    that reads transcripts one per round. Every other SEARCH_TOOL_NAMES call
    counts as a search.
    """
    for call in calls:
        name = call[0]
        args = call[1] if len(call) > 1 else {}
        if name not in SEARCH_TOOL_NAMES:
            continue
        if name == "youtube":
            from augmentum.tools.youtube import _extract_video_id
            if _extract_video_id(str((args or {}).get("query") or "")):
                continue  # direct transcript fetch — not a search
        return True
    return False


def _norm(url: str) -> str:
    """Cheap URL identity key — strip whitespace + a trailing slash. Paths can be
    case-sensitive so we do NOT lowercase the whole thing."""
    return (url or "").strip().rstrip("/")


@dataclass
class TurnSearchDedup:
    """Per-turn memory of search results already returned to the model.

    ``mark`` registers a result as shown and returns True iff it's new this turn
    — used by web/youtube where returning == showing. ``seen`` is a check-only
    peek for image_search, which registers (``mark``) only after a result is
    actually downloaded + displayed, so a failed download stays retryable.
    """

    _buckets: dict[str, set[str]] = field(default_factory=lambda: {
        "web": set(), "image": set(), "video": set(),
    })
    total_new: int = 0
    total_dup: int = 0
    _round_start_new: int = 0

    @staticmethod
    def _key(kind: str, key: str) -> str:
        # Videos are keyed by extracted id (already canonical); urls normalized.
        return (key or "").strip() if kind == "video" else _norm(key)

    def seen(self, kind: str, key: str) -> bool:
        k = self._key(kind, key)
        return bool(k) and k in self._buckets.get(kind, set())

    def mark(self, kind: str, key: str) -> bool:
        """Register ``key`` as shown; return True iff it was new this turn."""
        k = self._key(kind, key)
        if not k:
            return False
        bucket = self._buckets.setdefault(kind, set())
        if k in bucket:
            self.total_dup += 1
            return False
        bucket.add(k)
        self.total_new += 1
        return True

    # ── productivity-guard round accounting ───────────────────────────
    def begin_round(self) -> None:
        self._round_start_new = self.total_new

    def round_new_count(self) -> int:
        return self.total_new - self._round_start_new


_turn_dedup: contextvars.ContextVar[TurnSearchDedup | None] = contextvars.ContextVar(
    "turn_search_dedup", default=None,
)


def get_turn_dedup() -> TurnSearchDedup | None:
    """Return the active per-turn dedup, or None when not inside a tool loop."""
    return _turn_dedup.get()


def set_turn_dedup(dedup: TurnSearchDedup | None) -> contextvars.Token:
    """Install a per-turn dedup for the current async context; returns a reset token."""
    return _turn_dedup.set(dedup)


def reset_turn_dedup(token: contextvars.Token) -> None:
    """Restore the prior dedup (call in a ``finally`` after the turn)."""
    try:
        _turn_dedup.reset(token)
    except (ValueError, LookupError):
        # Token from a different context (e.g. the set happened in a parent
        # task) — clearing to None is the safe fallback.
        _turn_dedup.set(None)
