"""Presence context — what the user is engaged with RIGHT NOW.

The companion is the translation layer between the user and the
application (Matt, 2026-06-10). Translation requires perception: when
the user says "tell me about this page," the deixis has a referent on
their screen — and without this module she has no organ to see it, so
"this page" became a literal web search that yanked the user out of
the article they were reading (observed live).

Two perception paths, store-first:

  1. **AttentionStore** — an in-memory latest-per-slot record fed by
     the client through ``POST /api/architect/observe``. Every surface
     reports the same way (one attention event per focus change), so
     adding a new perceived surface is a one-line client call plus a
     topic mapping below — no new tables, no new endpoints.
  2. **Table fallback** — ``browse_history`` / ``device_play_history``
     already record attention server-side; they backstop the slots
     whose client events haven't fired (or after a restart).

The store is deliberately ephemeral (no SQLite): attention is a
property of the current moment, and after a restart the user's screen
state is unknown — resurrecting a pre-restart "current page" would be
a false percept. The history tables remain the durable record.

Slots:
  * ``page``    — the open browse article ("this page/article")
  * ``playing`` — audio/video in progress ("this song", "what's on")
  * ``reading`` — comic / read-along / book ("this book/comic")
  * ``scene``   — the active narrative character/story ("this scene")
  * ``working`` — the open coder workspace file ("this file/code")
  * ``mode``    — the active app surface (chat/coder/narrative/...)

Consumers:
  * The architect router's ConfidenceStack (deixis resolution).
  * ``compose_becca_prompt`` via the now-context block in ctx — she can
    TALK about what the user is attending to, not just navigate it.

Freshness windows are deliberately short — presence is about NOW; an
hour-old page is history, not attention.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Outer-dict LRU cap for the per-user attention/loaded stores. Per-slot
# footprint is bounded by the slot vocabulary, but the OUTER user key grew
# unbounded — one entry per distinct user, never evicted (audit
# 2026-06-17). On a single-user box this is 1; the cap only bites
# multi-tenant. Evicting an LRU user just means their next attention
# report re-creates the (cheap) entry.
_MAX_TRACKED_USERS = 256

PAGE_FRESH_S = 20 * 60        # a page open within 20min is "current"
PLAYING_FRESH_S = 60 * 60     # media within the hour is "what's on"
READING_FRESH_S = 60 * 60     # a book/comic session spans pauses
SCENE_FRESH_S = 60 * 60       # a story session spans pauses too
WORKING_FRESH_S = 30 * 60     # an editor file goes stale faster
MODE_FRESH_S = 4 * 60 * 60    # the surface they're in barely goes stale

_SLOT_TTL_S: dict[str, float] = {
    "page": PAGE_FRESH_S,
    "playing": PLAYING_FRESH_S,
    "reading": READING_FRESH_S,
    "scene": SCENE_FRESH_S,
    "working": WORKING_FRESH_S,
    "mode": MODE_FRESH_S,
}


# ---------------------------------------------------------------------------
# AttentionStore — latest attention entry per (user, slot)
# ---------------------------------------------------------------------------


class AttentionStore:
    """In-memory latest-per-slot attention record.

    Single-process, asyncio-single-threaded — no locking needed. Entries
    are tiny dicts ({label, url, kind, ref, ts}); per-user footprint is
    bounded by the slot vocabulary, so no eviction policy is required.
    """

    def __init__(self) -> None:
        self._slots: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()

    def note(self, user_id: str, slot: str, **fields: Any) -> None:
        if not user_id or slot not in _SLOT_TTL_S:
            return
        entry = {k: v for k, v in fields.items() if v not in (None, "")}
        entry["ts"] = time.time()
        self._slots.setdefault(user_id, {})[slot] = entry
        self._slots.move_to_end(user_id)
        while len(self._slots) > _MAX_TRACKED_USERS:
            self._slots.popitem(last=False)  # evict least-recently-active user

    def clear(self, user_id: str, slot: str) -> None:
        self._slots.get(user_id, {}).pop(slot, None)

    def get(self, user_id: str, slot: str) -> dict[str, Any] | None:
        """Return the entry with ``age_s`` if within the slot's TTL."""
        entry = self._slots.get(user_id, {}).get(slot)
        if not entry:
            return None
        age = time.time() - float(entry.get("ts") or 0)
        if age > _SLOT_TTL_S.get(slot, 0):
            self.clear(user_id, slot)
            return None
        out = {k: v for k, v in entry.items() if k != "ts"}
        out["age_s"] = age
        return out

    def reset(self) -> None:
        self._slots.clear()


ATTENTION = AttentionStore()


# ---------------------------------------------------------------------------
# LoadedContextStore — full content the USER explicitly handed her
# ---------------------------------------------------------------------------
#
# AttentionStore holds INDEX/DIGEST fidelity, auto-reported on every focus
# change (cheap, throttled). This store is the opposite end: full-fidelity
# content the user deliberately handed over via the widget's "Read this
# page / chat / file" button. Opt-in, larger, and never auto-populated —
# so the prompt only carries a digest of it (cheap) while the full body
# sits here behind the context_peek 'loaded' door (no prompt-budget cost
# until she actually pulls it). Ephemeral + in-memory, same rationale as
# AttentionStore: a loaded page is a property of the current moment, not
# durable state to resurrect after a restart.

LOADED_FRESH_S = 30 * 60  # a handed-over page/chat is "current" for 30min
_LOADED_MAX_CHARS = 16000  # generous body cap; peek returns a 4k window


class LoadedContextStore:
    """Latest full content per (user, kind) the user handed the companion."""

    def __init__(self) -> None:
        self._items: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
        self._seq = 0

    def load(
        self, user_id: str, kind: str, *,
        label: str, content: str, ref: str = "",
    ) -> int:
        """Store full content for a kind. Returns chars stored (capped)."""
        if not user_id or not kind:
            return 0
        body = (content or "")[:_LOADED_MAX_CHARS]
        # Monotonic seq is the "most recent" tiebreak — two loads inside
        # one clock tick (sub-ms button mashing) must still order by
        # action, not arbitrarily by dict iteration.
        self._seq += 1
        self._items.setdefault(user_id, {})[str(kind)[:32]] = {
            "kind": str(kind)[:32],
            "label": (label or kind)[:120],
            "content": body,
            "ref": (ref or "")[:64],
            "ts": time.time(),
            "seq": self._seq,
        }
        self._items.move_to_end(user_id)
        while len(self._items) > _MAX_TRACKED_USERS:
            self._items.popitem(last=False)  # evict least-recently-active user
        return len(body)

    def _live(self, entry: dict[str, Any] | None) -> dict[str, Any] | None:
        if not entry:
            return None
        if time.time() - float(entry.get("ts") or 0) > LOADED_FRESH_S:
            return None
        return entry

    def get(self, user_id: str, kind: str) -> dict[str, Any] | None:
        return self._live(self._items.get(user_id, {}).get(kind))

    def get_latest(self, user_id: str) -> dict[str, Any] | None:
        """Most-recently-loaded live item across kinds, or None."""
        items = [
            e for e in self._items.get(user_id, {}).values()
            if self._live(e) is not None
        ]
        if not items:
            return None
        return max(items, key=lambda e: int(e.get("seq") or 0))

    def clear(self, user_id: str, kind: str) -> None:
        self._items.get(user_id, {}).pop(kind, None)

    def reset(self) -> None:
        self._items.clear()


LOADED = LoadedContextStore()


# ---------------------------------------------------------------------------
# Topic mapper — client observe events → attention slots
# ---------------------------------------------------------------------------
#
# Topics arrive through the /api/architect/observe allow-list (see
# architect_routes._ALLOWED_TOPIC_PREFIXES). Adding a perceived surface =
# one reportAttention() call client-side + one branch here.


def observe_attention(user_id: str, topic: str, payload: dict[str, Any]) -> None:
    """Fold a client observation into the attention store. Never raises."""
    try:
        if topic == "surface.browse.page_opened":
            ATTENTION.note(
                user_id, "page",
                label=str(payload.get("title") or "")[:120],
                url=str(payload.get("url") or "")[:300],
                # The excerpt is what makes "tell me about this article"
                # answerable in conversation — title alone names the
                # referent but gives her nothing to actually say.
                excerpt=str(payload.get("excerpt") or "")[:1500],
            )
        elif topic == "surface.browse.page_closed":
            ATTENTION.clear(user_id, "page")
        elif topic in ("surface.media.playback_started",
                       "surface.audio.station_playing"):
            ATTENTION.note(
                user_id, "playing",
                label=str(payload.get("label") or "")[:120],
                kind=str(payload.get("kind") or "")[:32],
                ref=str(payload.get("ref") or "")[:64],
            )
        elif topic in ("surface.media.reading_started",
                       "surface.comic.opened"):
            ATTENTION.note(
                user_id, "reading",
                label=str(payload.get("label") or "")[:120],
                kind=str(payload.get("kind") or ("comic" if "comic" in topic else "book"))[:32],
                ref=str(payload.get("ref") or "")[:64],
            )
        elif topic == "surface.narrative.scene_active":
            ATTENTION.note(
                user_id, "scene",
                label=str(payload.get("label") or "")[:120],
                ref=str(payload.get("ref") or "")[:64],
            )
        elif topic == "surface.narrative.scene_closed":
            ATTENTION.clear(user_id, "scene")
        elif topic == "surface.coder.file_opened":
            ATTENTION.note(
                user_id, "working",
                label=str(payload.get("label") or "")[:120],
                path=str(payload.get("path") or "")[:300],
                ref=str(payload.get("ref") or "")[:64],
            )
        elif topic == "surface.coder.closed":
            ATTENTION.clear(user_id, "working")
        elif topic == "surface.attention.mode_changed":
            ATTENTION.note(
                user_id, "mode",
                label=str(payload.get("mode") or "")[:32],
            )
        elif topic == "surface.audio.kind_changed":
            # AudioBus silence event — nothing is playing anymore. The
            # device_play_history fallback still shows recent media with
            # an honest "N minutes ago" recency, so clearing is safe.
            kinds = payload.get("kinds")
            if isinstance(kinds, list) and not kinds:
                ATTENTION.clear(user_id, "playing")
    except Exception:  # noqa: BLE001 — perception must never break observe
        log.debug("attention_observe_failed", topic=topic, exc_info=True)


# ---------------------------------------------------------------------------
# Snapshot — store-first, table fallback
# ---------------------------------------------------------------------------


def _age_s(created_at: str) -> float | None:
    try:
        raw = (created_at or "").replace("Z", "+00:00")
        ts = datetime.fromisoformat(raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return (datetime.now(UTC) - ts).total_seconds()
    except Exception:  # noqa: BLE001
        return None


async def now_context(
    conn: Any, user_id: str, app_state: Any = None,
) -> dict[str, Any]:
    """Snapshot of the user's current attention. All fields optional.

    Returns ``{page, playing, reading, mode}`` — each a dict or None.
    Store entries win (richer: real titles, live state); history tables
    backstop ``page`` and ``playing`` when no client event has arrived.
    Never raises — perception failures degrade to an empty snapshot.

    When ``app_state`` is provided, an active receiver cast session
    OVERRIDES the ``playing`` slot — the registry's session list is
    server truth (no client observe event needed), and when both the
    tab and a TV are playing, the receiver is the one the user means
    (same preference the control ladder applies).
    """
    out: dict[str, Any] = {
        "page": None, "playing": None, "reading": None,
        "scene": None, "working": None, "mode": None, "loaded": None,
        "device": None,
    }
    if not user_id:
        return out

    for slot in ("page", "playing", "reading", "scene", "working", "mode"):
        out[slot] = ATTENTION.get(user_id, slot)

    # Phone presence — the notification WS the Android foreground service
    # holds open is itself a liveness signal. Read live off the hub (a
    # dropped socket self-heals; no staleness): an attached "android"
    # connection means the user is on/near their phone right now, which
    # tells her phone-side capabilities (bluetooth_list, …) are usable.
    hub = getattr(app_state, "notification_hub", None) if app_state else None
    if hub is not None:
        try:
            if "android" in hub.device_types(user_id):
                out["device"] = {"kind": "android", "label": "Android phone"}
        except Exception:  # noqa: BLE001 — presence is best-effort
            log.debug("presence_device_read_failed", exc_info=True)

    # Full content the user explicitly handed her (widget "Read this …"
    # button). Index/digest rides in the prompt; the full body stays
    # behind context_peek('loaded') — see perception_lines.
    out["loaded"] = LOADED.get_latest(user_id)

    reg = getattr(app_state, "device_registry", None) if app_state else None
    if reg is not None:
        try:
            sessions = await reg.list_sessions(user_id=user_id)
            media = [
                s for s in sessions
                if str(getattr(s, "capability_id", "")).startswith("media.")
                and getattr(s, "title", "")
            ]
            if media:
                live = max(media, key=lambda s: float(s.last_event_at or 0))
                label = live.title
                extra = getattr(live, "extra", None) or {}
                device_label = extra.get("device_label") or live.device_id
                if device_label == live.device_id:
                    try:
                        device = await reg.get(live.device_id, user_id=user_id)
                        if device is not None:
                            device_label = device.label or device.id
                    except Exception:  # noqa: BLE001
                        log.debug("presence_receiver_device_failed", exc_info=True)
                kind = "video" if "video" in str(live.capability_id) else "audio"
                out["playing"] = {
                    "label": str(label)[:120],
                    "kind": kind,
                    "device": str(device_label)[:80],
                    "author": str(
                        extra.get("author") or extra.get("artist") or ""
                    )[:80],
                    # A session in the runtime list IS live — stopped
                    # casts are removed. age 0 = "right now", honest.
                    "age_s": 0.0,
                }
        except Exception:  # noqa: BLE001
            log.debug("presence_receiver_read_failed", exc_info=True)

    if out["page"] is None and conn is not None:
        try:
            from augmentum.architect.inference import query_browse_history
            rows = await query_browse_history(conn, user_id, limit=1)
            if rows:
                row = rows[0]
                age = _age_s(row.get("last_visited") or "")
                if age is not None and age <= PAGE_FRESH_S:
                    out["page"] = {
                        "label": (row.get("domain") or "")[:120],
                        "url": (row.get("url") or "")[:300],
                        "age_s": age,
                    }
        except Exception:  # noqa: BLE001
            log.debug("presence_page_read_failed", exc_info=True)

    if out["playing"] is None and conn is not None:
        try:
            from augmentum.architect.inference import query_play_history
            rows = await query_play_history(
                conn, user_id, limit=1, favourites_first=False,
            )
            if rows:
                row = rows[0]
                age = _age_s(row.get("created_at") or "")
                if age is not None and age <= PLAYING_FRESH_S:
                    out["playing"] = {
                        "label": (row.get("content_label") or "")[:120],
                        "kind": (row.get("capability_id") or "")[:32],
                        "age_s": age,
                    }
        except Exception:  # noqa: BLE001
            log.debug("presence_playing_read_failed", exc_info=True)

    return out


def prompt_lines(snapshot: dict[str, Any]) -> list[str]:
    """Render the snapshot as prompt-ready lines (empty when nothing
    fresh — the composer skips empty layers)."""
    lines: list[str] = []

    page = snapshot.get("page")
    if page and (page.get("label") or page.get("title") or page.get("url")):
        label = page.get("label") or page.get("title") or page.get("url")
        lines.append(
            f"They have open on screen: \"{label}\" — when they say "
            f"'this page' or 'this article', this is what they mean."
        )
        # NOTE (2026-06-12): the page excerpt no longer renders here.
        # Depth follows the ring lifecycle (full on first sight, digest
        # while warm, peek when cold) — see perception_lines(), which
        # is what the companion paths consume. This function renders
        # the INDEX tier only.

    reading = snapshot.get("reading")
    if reading and reading.get("label"):
        kind = reading.get("kind") or "book"
        lines.append(
            f"They're reading: \"{reading['label']}\" ({kind}) — "
            f"'this {kind}' / 'what I'm reading' means this."
        )

    playing = snapshot.get("playing")
    if playing and playing.get("label"):
        kind = playing.get("kind") or "media"
        mins = int((playing.get("age_s") or 0) // 60)
        recency = "right now" if mins < 5 else f"{mins} minutes ago"
        device = playing.get("device") or ""
        author = playing.get("author") or ""
        title = playing["label"] + (f" by {author}" if author else "")
        if device:
            lines.append(
                f"Playing ({recency}): {title} ({kind}) on "
                f"{device} — 'turn it up' / 'pause it' means this cast. "
                f"Peek playing for position and state."
            )
        else:
            lines.append(
                f"Playing ({recency}): {title} ({kind})."
            )

    scene = snapshot.get("scene")
    if scene and scene.get("label"):
        lines.append(
            f"They're in a story scene with \"{scene['label']}\" "
            f"(narrative mode) — 'this scene' / 'the story' / 'this "
            f"character' means this. The story itself belongs to "
            f"narrative mode; don't take actions inside it."
        )

    working = snapshot.get("working")
    if working and working.get("label"):
        path = (working.get("path") or "").strip()
        where = f" ({path})" if path and path != working["label"] else ""
        lines.append(
            f"They have a code file open in the workspace: "
            f"\"{working['label']}\"{where} — 'this file' / 'this code' "
            f"means this."
        )

    mode = snapshot.get("mode")
    if mode and mode.get("label"):
        lines.append(
            f"They're in the {mode['label']} area of the app."
        )

    device = snapshot.get("device")
    if device and device.get("label"):
        lines.append(
            f"They're on their {device['label']} right now — phone-side "
            f"capabilities (like checking which Bluetooth devices they're "
            f"connected to) are available if one would actually help."
        )

    loaded = snapshot.get("loaded")
    if loaded and loaded.get("label"):
        kind = loaded.get("kind") or "item"
        head = (loaded.get("content") or "")[:700]
        lines.append(
            f"They handed you the full {kind} to read: \"{loaded['label']}\" "
            f"— 'this {kind}' means this."
        )
        if head:
            lines.append(f"What you have of it: {head}")

    return lines


async def _active_note(
    app_state: Any, user_id: str, session_id: str,
) -> dict[str, str] | None:
    """The open co-author note, or None. Hydrates the cache first so
    the note survives restarts / voice session-id churn. Best-effort.
    """
    if app_state is None or not user_id:
        return None
    try:
        from augmentum.intent.dispatch import get_referent_cache
        refs = get_referent_cache(app_state, user_id, session_id)
        from augmentum.companion_runtime.working_state import (
            hydrate_working_state,
        )
        await hydrate_working_state(app_state, user_id, refs)
        note_id = (getattr(refs, "active_note_id", "") or "").strip()
        if not note_id:
            return None
        store = getattr(app_state, "notes_store", None)
        if store is None:
            return None
        note = await store.get(note_id, user_id=user_id)
        if not note:
            return None
        return {
            "note_id": note_id,
            "title": (note.get("title") or "Untitled").strip(),
            "content": (note.get("content") or "").strip(),
        }
    except Exception:  # noqa: BLE001 — perception is best-effort
        log.debug("active_note_fetch_failed", exc_info=True)
        return None


async def active_note_lines(
    app_state: Any, user_id: str, session_id: str,
) -> list[str]:
    """Co-author context — the open sticky note she's writing with the
    user. Without this she can append but can't SEE the note: "your
    third point contradicts the first" is impossible.

    Legacy always-full renderer; the companion paths now consume
    :func:`perception_lines`, which gives the note the ring lifecycle
    (full tail when touched, digest when idle). Kept for compatibility.

    Returns [] when no note is active (the common case) or on any
    failure — co-author context is best-effort, never blocking.
    """
    note = await _active_note(app_state, user_id, session_id)
    if note is None:
        return []
    lines = [
        f"Open note you two are writing together: \"{note['title']}\". "
        "Additions go in via note.append — keep talking about the "
        "content itself; the write is silent."
    ]
    content = note["content"]
    if content:
        tail = content[-600:]
        marker = "…" if len(content) > 600 else ""
        lines.append(f"The note so far{marker}: {tail}")
    return lines


# ---------------------------------------------------------------------------
# Perception contract — the coder-tree equivalent for companion mode
# ---------------------------------------------------------------------------
#
# Everything visible is named; everything named carries its fidelity
# (FULL — content below / DIGEST — peek for detail / NAMED-ONLY); and
# everything unnamed is declared invisible. The unmarked-territory
# ambiguity is what produced both confabulation ("guess what you can't
# see") and false denial ("I can't see your screen" while the page
# line sat right there) — this block removes the unmarked territory.

_BLIND_LINE = (
    "That's everything of theirs you can perceive. If something isn't "
    "listed — their screen beyond these, games, other tabs — you can't "
    "see it; say so or peek rather than guess."
)


async def perception_lines(
    app_state: Any,
    conn: Any,
    user_id: str,
    session_id: str,
    scoring_text: str = "",
    detail_budget_chars: int = 2400,
) -> list[str]:
    """Assemble the full perception block for the companion prompt.

    One call per user turn (both voice and becca_direct ctx builders) —
    this is also where the ring's TURN CLOCK advances. Pipeline:

      1. bump the turn counter
      2. snapshot presence (AttentionStore + table fallback)
      3. feed presence depth into the ring (page excerpt on a NEW page,
         note tail when its content CHANGED — co-author freshness)
      4. deterministic touch-and-match against the turn text; matched
         entries refresh their decay clock AND re-inflate (full detail
         back in the prompt — the model never decides to re-fetch for
         the common case)
      5. render: fidelity-marked index, warm ring digests, blind line
    """
    from augmentum.config import settings

    keep_turns = int(getattr(settings, "companion_results_ring_turns", 3) or 3)
    enabled = bool(getattr(settings, "companion_results_ring_enabled", True))

    refs = None
    try:
        if app_state is not None and user_id:
            from augmentum.intent.dispatch import get_referent_cache
            refs = get_referent_cache(app_state, user_id, session_id)
    except Exception:  # noqa: BLE001
        log.debug("perception_refs_failed", exc_info=True)

    snapshot = await now_context(conn, user_id, app_state=app_state)
    note = await _active_note(app_state, user_id, session_id)

    if refs is None or not enabled:
        # Degraded path: legacy index + full note, still with the
        # contract's closing line — honesty about blindness doesn't
        # depend on the ring.
        lines = prompt_lines(snapshot)
        if note is not None:
            lines += await active_note_lines(app_state, user_id, session_id)
        lines.append(_BLIND_LINE)
        return lines

    from augmentum.companion_runtime import ring as _ring

    turn = _ring.bump_turn(refs)

    # -- presence feeders -------------------------------------------------
    by_slot = {e.get("slot"): e for e in _ring.alive(refs, keep_turns=keep_turns)}
    page = snapshot.get("page") or {}
    page_label = page.get("label") or page.get("title") or page.get("url") or ""
    if page_label:
        prev = by_slot.get("presence:page")
        if prev is None or prev.get("label") != str(page_label)[:120]:
            # NEW page (or first sight of it) — born now, so it renders
            # full this turn and decays from here.
            _ring.record(
                refs, kind="presence", slot="presence:page",
                label=str(page_label),
                digest="open on their screen",
                detail=(page.get("excerpt") or "").strip()[:700],
                refetch={"slot": "page"},
            )
    if note is not None:
        tail = note["content"][-600:]
        prev = by_slot.get("presence:note")
        if prev is None or prev.get("detail") != tail:
            # Content changed since she last saw it — co-author
            # freshness: every touched turn re-earns the full tail.
            _ring.record(
                refs, kind="presence", slot="presence:note",
                label=note["title"],
                digest=f"{len(note['content'])} chars",
                detail=tail,
                refetch={"slot": "note"},
            )
    loaded = snapshot.get("loaded") or {}
    loaded_label = loaded.get("label") or ""
    if loaded_label:
        # The user-handed full content. Born-full on a NEW load (signature
        # = kind+label+length), decays to a peek pointer after, same
        # lifecycle as the page excerpt. The full body lives in
        # LoadedContextStore behind context_peek('loaded') — only a head
        # window rides the ring.
        loaded_sig = (
            f"{loaded.get('kind')}:{loaded_label}:"
            f"{len(loaded.get('content') or '')}"
        )
        prev = by_slot.get("loaded:current")
        if prev is None or prev.get("digest") != loaded_sig:
            _ring.record(
                refs, kind="presence", slot="loaded:current",
                label=f"{loaded.get('kind') or 'item'}: {loaded_label}",
                digest=loaded_sig,
                detail=(loaded.get("content") or "")[:900],
                refetch={"slot": "loaded"},
            )

    # -- relevance: touch + collect re-inflation candidates ----------------
    matched = _ring.touch_and_match(refs, scoring_text, keep_turns=keep_turns)
    matched_ids = {id(e) for e in matched}
    entries = _ring.alive(refs, keep_turns=keep_turns)
    by_slot = {e.get("slot"): e for e in entries}

    def _fresh(e: dict | None) -> bool:
        return e is not None and (
            int(e.get("born_turn") or -1) == turn or id(e) in matched_ids
        )

    # -- detail budget governor ---------------------------------------------
    # An extreme turn can QUALIFY ~2.5k chars of full-fidelity detail at
    # once (page first-sight + note change + two re-inflations), and the
    # compose ceiling only LOGS overruns — so the block self-caps here.
    # Priority: what the user is asking about NOW (re-inflations) beats
    # the changed-note tail beats the first-sight page excerpt. Anything
    # squeezed out keeps its pointer line and stays one peek away. Voice
    # passes a tighter budget than chat (1800 vs 3200 token ceilings).
    budget = max(0, int(detail_budget_chars))
    trimmed: list[str] = []

    inflate_ok: set[int] = set()
    for e in matched:
        if e.get("kind") == "presence" or not e.get("detail"):
            continue
        if len(inflate_ok) >= 2:
            break
        cost = len(e["detail"][:600])
        if cost <= budget:
            inflate_ok.add(id(e))
            budget -= cost
        else:
            trimmed.append(f"inflate:{e.get('slot') or e.get('label')}")

    note_entry = by_slot.get("presence:note")
    note_full = bool(
        note is not None and _fresh(note_entry)
        and (note_entry or {}).get("detail")
    )
    if note_full:
        cost = len(note_entry["detail"])
        if cost <= budget:
            budget -= cost
        else:
            note_full = False
            trimmed.append("note_tail")

    page_entry = by_slot.get("presence:page")
    page_full = bool(
        page_label and _fresh(page_entry)
        and (page_entry or {}).get("detail")
    )
    if page_full:
        cost = len(page_entry["detail"])
        if cost <= budget:
            budget -= cost
        else:
            page_full = False
            trimmed.append("page_excerpt")

    # The user-handed content is the highest-intent of all — they
    # explicitly asked her to read THIS — so it gets first claim on the
    # budget after live re-inflations.
    loaded_entry = by_slot.get("loaded:current")
    loaded_full = bool(
        loaded_label and _fresh(loaded_entry)
        and (loaded_entry or {}).get("detail")
    )
    if loaded_full:
        cost = len(loaded_entry["detail"])
        if cost <= budget:
            budget -= cost
        else:
            loaded_full = False
            trimmed.append("loaded_head")

    if trimmed:
        log.info(
            "perception_detail_trimmed",
            dropped=trimmed, budget_chars=detail_budget_chars,
        )

    lines: list[str] = []

    # -- page (index + fidelity) -------------------------------------------
    if page_label:
        lines.append(
            f"They have open on screen: \"{page_label}\" — when they say "
            f"'this page' or 'this article', this is what they mean."
        )
        if page_full:
            lines.append(f"How the page starts: {page_entry['detail']}")
        else:
            lines.append(
                "Page text: not in front of you right now — peek page "
                "before quoting or summarizing it."
            )

    # -- loaded full content (the "Read this" widget handoff) --------------
    # The user deliberately handed her this to read — strongest signal of
    # what they want her attending to. She has the ACTUAL text (not just a
    # title), so deixis and quoting are both grounded.
    if loaded_label:
        kind = loaded.get("kind") or "item"
        lines.append(
            f"They handed you the full {kind} to read: \"{loaded_label}\" "
            f"— you have its actual text, and 'this {kind}' means this."
        )
        if loaded_full:
            lines.append(f"What you have of it: {loaded_entry['detail']}")
        else:
            lines.append(
                "Its full text isn't inline right now — peek loaded to "
                "pull it before quoting or summarizing."
            )

    # -- the other index slots render as named-only (their natural tier) ---
    rest = dict(snapshot)
    rest["page"] = None
    lines += prompt_lines(rest)

    # -- co-author note (the collaborative-authoring case) -----------------
    if note is not None:
        lines.append(
            f"Open note you two are writing together: \"{note['title']}\". "
            "Additions go in via note.append — keep talking about the "
            "content itself; the write is silent."
        )
        if note_full:
            marker = "…" if len(note["content"]) > 600 else ""
            lines.append(f"The note so far{marker}: {note_entry['detail']}")
        elif note["content"]:
            lines.append(
                f"Note content: {len(note['content'])} chars — peek note "
                "before editing or quoting specifics."
            )

    # -- recent results (the ring's tool tier) ------------------------------
    tool_entries = [e for e in entries if e.get("kind") != "presence"]
    if tool_entries:
        lines.append(
            "Things you recently looked at (digests — peek recent for "
            "full detail):"
        )
        for e in reversed(tool_entries):
            age = max(0, turn - int(e.get("born_turn") or 0))
            ago = "this turn" if age == 0 else (
                "1 turn ago" if age == 1 else f"{age} turns ago"
            )
            # Warn-before-decay (Phase 2, mirrors Anthropic context
            # editing's save-before-clear): an entry on its LAST alive
            # turn says so, giving her one chance to persist or peek
            # before it silently leaves her context.
            touch = int(e.get("touch_turn") or e.get("born_turn") or 0)
            fade = (
                " (about to fade — memory.save it or peek recent if it "
                "still matters)"
                if turn - touch >= keep_turns else ""
            )
            lines.append(f"  [{ago}] {e.get('label')} — {e.get('digest')}{fade}")
            if id(e) in inflate_ok:
                # Re-inflation: the turn clearly references this entry,
                # so the full text rides back in — no model judgment
                # required, no re-run of a non-repeatable tool.
                lines.append(f"    What it said: {e['detail'][:600]}")

    if lines:
        lines.append(_BLIND_LINE)
    else:
        lines.append(
            "Nothing of theirs is on screen for you right now. If they "
            "reference something visual, say you can't see it — don't "
            "guess."
        )
    return lines
