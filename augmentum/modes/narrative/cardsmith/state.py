"""In-memory state for active Cardsmith sessions, backed by a write-through
SQLite cache.

Each Cardsmith conversation gets a session_id that the frontend supplies on
every /turn call. The session holds the conversation log, the accumulated
field emissions (latest scalar wins; arrays append), and metadata. Sessions
are evicted from the OrderedDict by simple LRU + TTL.

The OrderedDict is the read path for hot sessions. A companion
``cardsmith_sessions`` SQLite table (migration 185) is the durable backing
store — the routes file owns the SQL and calls ``serialize_for_disk()`` /
``session_from_row()`` here for the JSON marshalling.

Rehydration: when ``get_session`` is called with a session_id that isn't in
memory (after a server restart, LRU eviction, or just a different worker),
the routes layer queries the disk row and calls ``register_session`` to put
the parsed session back into the OrderedDict. From there it behaves exactly
like a hot session — including TTL eviction on next access if it's stale.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

# Field paths that are scalar (latest emission wins).
_SCALAR_PATHS: frozenset[str] = frozenset({
    "name",
    "description",
    "personality",
    "scenario",
    "greeting",
    "examples",
    "visualTraits",
    "imageStyle",
    "voice",
    "systemPrompt",
    "postHistoryInstructions",
    "depthPrompt",
    "depthPromptDepth",
    "creatorNotes",
    "backgroundImage",
    # Description paragraph slots — composed into `description` at save time
    # by output_mapper.build_character_payload. The model emits each slot
    # exactly once when its question is answered, instead of re-emitting a
    # growing `description` field across turns (which wasted tokens).
    "desc_physical",
    "desc_personality",
    "desc_depth",
    # Ensemble-specific scalars
    "group_dynamic",
    "generation_mode",
})

# Field paths that are arrays (each emission appends). The "[]" suffix on the
# emitted path is what flags array semantics; this set documents which ones
# the output mapper consumes.
_ARRAY_PATHS: frozenset[str] = frozenset({
    "tags",
    "alternateGreetings",
    "lorebook",
    "regex_scripts",
    "avatar_prompt",
    # Ensemble-specific arrays
    "members",
    "relationships",
    # Agentic control — not persisted on the saved card, consumed by routes
    "fetch_targets",
})


# ── Eviction tuning ────────────────────────────────────────────────────────
_MAX_SESSIONS = 256
_TTL_SECONDS = 60 * 60 * 4  # 4 hours of idle


@dataclass
class CardsmithSession:
    """One in-flight Cardsmith conversation."""

    session_id: str
    user_id: str
    card_type: str  # "single" | "ensemble" | "world_rpg"
    source: str  # "describe" | "wiki" | "blank"
    created_at: float
    last_active_at: float

    # Conversation log fed to the LLM each turn. The system prompt is NOT
    # stored here — it's prepended at request time by the route so prompt
    # edits take effect for the next turn.
    messages: list[dict[str, str]] = field(default_factory=list)

    # Accumulated field state. Scalars overwrite; arrays append.
    fields: dict[str, Any] = field(default_factory=dict)

    # Free-form metadata bucket — wiki source URL, source classification,
    # initial prompt seed, etc.
    meta: dict[str, Any] = field(default_factory=dict)

    # Set true once the model emits [CARDSMITH_DONE] (or the user hits the
    # explicit Save button which forces finalize).
    finalized: bool = False

    # Per-session lock — held by the /turn route for the duration of a
    # streaming reply so two concurrent turns can't interleave message
    # appends or field commits. Created lazily so we don't depend on a
    # running event loop at session creation time.
    _lock: asyncio.Lock | None = field(default=None, repr=False, compare=False)

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def append_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.last_active_at = time.time()

    def append_assistant(self, content: str) -> None:
        """Record the visible portion of the assistant reply (tags stripped)."""
        if content:
            self.messages.append({"role": "assistant", "content": content})
        self.last_active_at = time.time()

    def commit_field(self, path: str, value: str) -> None:
        """Apply a single ``<set path="...">`` emission to accumulated state.

        Scalar paths overwrite; ``foo[]`` paths append to a list. Array values
        may be JSON objects (lorebook entries, regex scripts) or plain
        strings (tags, alternate greetings) — we try JSON parse first and
        fall back to the raw string.
        """
        if path.endswith("[]"):
            base = path[:-2]
            if base not in self.fields or not isinstance(self.fields[base], list):
                self.fields[base] = []
            self.fields[base].append(_coerce_value(value))
        else:
            self.fields[path] = _coerce_value(value)
        self.last_active_at = time.time()

    def to_preview(self) -> dict[str, Any]:
        """Snapshot for the live card-preview pane in the modal."""
        return {
            "session_id": self.session_id,
            "card_type": self.card_type,
            "source": self.source,
            "fields": _serialize_fields(self.fields),
            "finalized": self.finalized,
            "message_count": len(self.messages),
        }


def _coerce_value(raw: Any) -> Any:
    """Try JSON-parse strings; pass through anything that isn't a string.

    The streaming parser only ever emits strings, but callers using
    ``commit_field`` directly may pass ints, bools, dicts, lists, etc. — we
    accept those as-is rather than crashing.
    """
    if not isinstance(raw, str):
        return raw
    import json
    s = raw.strip()
    if not s:
        return s
    if s[0] in ("{", "["):
        try:
            return json.loads(s)
        except (ValueError, TypeError):
            return raw
    return raw


def _serialize_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Defensive copy for the JSON wire — no shared refs back to the session."""
    import copy
    return copy.deepcopy(fields)


# ── Global registry ───────────────────────────────────────────────────────
_sessions: OrderedDict[str, CardsmithSession] = OrderedDict()


def _evict_stale() -> None:
    """Drop sessions older than TTL, then make room for one new session under the cap.

    Called from ``get_or_create_session`` BEFORE the new session is inserted, so
    we evict down to ``_MAX_SESSIONS - 1`` to leave space without overshooting.
    """
    cutoff = time.time() - _TTL_SECONDS
    stale = [sid for sid, s in _sessions.items() if s.last_active_at < cutoff]
    for sid in stale:
        _sessions.pop(sid, None)
    while len(_sessions) >= _MAX_SESSIONS:
        _sessions.popitem(last=False)


def get_or_create_session(
    *,
    user_id: str,
    card_type: str,
    source: str,
    seed_prompt: str = "",
) -> CardsmithSession:
    """Create a fresh Cardsmith session and register it."""
    _evict_stale()
    sid = "cs_" + uuid.uuid4().hex[:16]
    now = time.time()
    sess = CardsmithSession(
        session_id=sid,
        user_id=user_id,
        card_type=card_type,
        source=source,
        created_at=now,
        last_active_at=now,
    )
    if seed_prompt:
        sess.meta["seed_prompt"] = seed_prompt
    _sessions[sid] = sess
    return sess


def get_session(session_id: str, *, user_id: str) -> CardsmithSession | None:
    """Look up a session, enforcing user ownership.

    Returns None if the session doesn't exist, has expired, or belongs to a
    different user (the latter case prevents cross-tenant access by guessing
    a session_id).
    """
    sess = _sessions.get(session_id)
    if sess is None:
        return None
    if sess.user_id and user_id and sess.user_id != user_id:
        return None
    if time.time() - sess.last_active_at > _TTL_SECONDS:
        _sessions.pop(session_id, None)
        return None
    # Move to MRU end.
    _sessions.move_to_end(session_id)
    return sess


def drop_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


# ── Persistence helpers (called by cardsmith_routes; SQL lives there) ─────

def register_session(sess: "CardsmithSession") -> None:
    """Insert a (possibly rehydrated) session back into the in-memory dict.

    Used by the disk-fallback path: when ``get_session`` misses, routes
    load the row from SQLite, parse it with ``session_from_row``, and call
    this to make it queryable as a hot session. Skips ``_evict_stale`` so
    rehydration of one session doesn't cascade-evict others.
    """
    _sessions[sess.session_id] = sess


def serialize_for_disk(sess: "CardsmithSession") -> dict[str, Any]:
    """Snapshot the session as a row dict matching the cardsmith_sessions
    schema. Caller binds the result into the SQL params.

    JSON blobs are produced here so the route layer doesn't need to know
    which fields are scalar vs nested. The in-memory representation uses
    native Python lists/dicts; SQLite stores them as TEXT.
    """
    import json
    return {
        "session_id": sess.session_id,
        "user_id": sess.user_id,
        "card_type": sess.card_type,
        "source": sess.source,
        "created_at": sess.created_at,
        "last_active_at": sess.last_active_at,
        "messages": json.dumps(sess.messages),
        "fields": json.dumps(sess.fields),
        "meta": json.dumps(sess.meta),
        "finalized": 1 if sess.finalized else 0,
    }


def session_from_row(row: dict[str, Any]) -> "CardsmithSession":
    """Parse a cardsmith_sessions row back into a CardsmithSession.

    Tolerates rows written by older code where new fields didn't exist —
    missing JSON blobs default to empty list/dict so a partial upgrade
    doesn't crash rehydration.
    """
    import json
    return CardsmithSession(
        session_id=row["session_id"],
        user_id=row["user_id"],
        card_type=row["card_type"],
        source=row["source"],
        created_at=float(row["created_at"]),
        last_active_at=float(row["last_active_at"]),
        messages=json.loads(row.get("messages") or "[]"),
        fields=json.loads(row.get("fields") or "{}"),
        meta=json.loads(row.get("meta") or "{}"),
        finalized=bool(row.get("finalized", 0)),
    )
