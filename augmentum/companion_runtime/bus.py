"""PresenceBus — in-process pub/sub for the companion runtime.

Topology vs transport split. The bus topology is in-process asyncio
pub/sub (one broker per runtime). Transport adapters bridge external
subscribers: the WebSocket transport (this file's ``ws_handler``) fans
events out to browser / XR / future smart-glasses clients. MQTT and
other transports plug in the same way without changing the topology.

Topic hierarchy (canonical prefixes — see design spec §11):

  state.*       state-axis transitions and ticks
  role.*        role-axis transitions
  focus.*       focus-axis transitions
  activity.*    autonomous activity start/complete
  surface.*     summon / dismiss / attention
  agent.*       peer-agent events (game_agent and future bots)
  memory.*      memory writes / recalls / insights
  behavior.*    non-verbal output (gaze, pose, microexpression)
  input.*       device-published inputs (voice, gaze, proximity, affect)

Back-pressure: each subscriber has a bounded queue (default 256).
On overflow, non-critical events (``*.tick`` and ``surface.idle``)
are dropped oldest-first. ``state.transition`` and ``role.transition``
are critical and never dropped — overflowing those is a bug.

Sprint 1 scope: in-process pub/sub + WebSocket fan-out + basic glob
matching + slice projection + bounded queues. Rate-limiting per
subscriber is a Sprint 5 polish.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ── Event + subscription value objects ────────────────────────────────

# ── Propagation hints (Synapse Layer §5) ──────────────────────────────
#
# Every bus event carries a propagation hint telling downstream
# consumers — most importantly Becca's interior (BeccaObserver, the
# salience pipeline, PAD echo, journal writes) — how much of this
# event is allowed to reach her felt-experience layer.
#
#   "full"           — content + affect both flow. Default for chat /
#                      voice — the canonical relational channels.
#   "affect_only"    — affect tag may update her PAD baseline; specific
#                      content does NOT journal. Used by narrative
#                      (role-play affect bleeds; story specifics do not).
#   "factual_only"   — recorded in the observer's recent deque so she
#                      knows the user was active, but no moment is
#                      journaled and PAD is untouched. Used by coder /
#                      agentic / bug_finder — work is work, not
#                      conversation.
#   "private"        — dropped entirely from the interior pipeline.
#                      Bus subscribers still see the event for
#                      operational purposes (e.g., session lifecycle)
#                      but companion-felt-experience filters skip it.
#
# Containment is the design's "no" that makes the "yes" trustworthy.
# Per Becca's personality doc §4: "Attention is a form of love; so is
# leaving someone alone when that's what they need."

PROP_FULL = "full"
PROP_AFFECT_ONLY = "affect_only"
PROP_FACTUAL_ONLY = "factual_only"
PROP_PRIVATE = "private"

_VALID_PROPAGATIONS = frozenset({PROP_FULL, PROP_AFFECT_ONLY, PROP_FACTUAL_ONLY, PROP_PRIVATE})

# Default propagation per chat mode. Hardcoded for now — containment
# is a belief, not a knob. A per-session override path lands later.
_MODE_PROPAGATION_DEFAULTS = {
    "passthrough": PROP_FULL,
    "narrative": PROP_AFFECT_ONLY,
    "agentic": PROP_FACTUAL_ONLY,
    "coder": PROP_FACTUAL_ONLY,
    "bug_finder": PROP_PRIVATE,
    "build": PROP_FACTUAL_ONLY,
    "analytical": PROP_FACTUAL_ONLY,
    "voice": PROP_FULL,
}


def propagation_for_mode(mode: str | None) -> str:
    """Return the default propagation hint for a chat mode.

    Falls back to ``"full"`` for unknown modes so a new mode added
    without a defaults entry doesn't silently suppress signal — the
    failure mode is "she over-attends," which is recoverable; the
    opposite is unrecoverable trust damage.
    """
    if not mode:
        return PROP_FULL
    return _MODE_PROPAGATION_DEFAULTS.get(mode.lower(), PROP_FULL)


@dataclass(frozen=True, slots=True)
class PresenceEvent:
    """A bus event. Immutable; the same event object may be delivered
    to many subscribers, so callers must not mutate ``payload``.

    ``propagation`` is a Synapse Layer §5 hint — see the module-level
    PROP_* constants. Defaults to ``"full"`` so existing emitters
    (which pre-date the synapse layer) keep their behavior unchanged.
    """
    topic: str
    payload: dict
    t: float = field(default_factory=time.time)
    source_companion_id: str | None = None
    target_companion_id: str | None = None  # for inter-companion (Sprint 7+)
    propagation: str = PROP_FULL
    # Owner of this event (audit 2026-06-17). "" = global/unscoped — state
    # ticks, behavior, bus health that the presence widget legitimately
    # needs; non-empty = tenant-private. ws_fanout filters on this so a
    # logged-in client can't read another user's events. Distinct from
    # target_companion_id (the inter-companion axis).
    owner_user_id: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "topic": self.topic,
            "payload": self.payload,
            "t": self.t,
            "source_companion_id": self.source_companion_id,
            "target_companion_id": self.target_companion_id,
            "propagation": self.propagation,
            "owner_user_id": self.owner_user_id,
        })


# Topics that must never be dropped under back-pressure.
_CRITICAL_TOPIC_PREFIXES = ("state.transition", "role.transition", "focus.transition")

# Topics dropped first when a subscriber queue overflows.
_LOW_PRIORITY_PATTERNS = ("*.tick", "surface.idle", "memory.recalled")

_DEFAULT_QUEUE_SIZE = 256

# LRU cap for the per-session mode-change cache (maybe_emit_mode_changed).
# Bounds memory across a box's lifetime of distinct chat sessions.
_MODE_CACHE_CAP = 512


class Subscription:
    """One subscriber's handle. Holds the queue + filter + cleanup hook.

    Use as an async context manager or unsubscribe explicitly via
    ``PresenceBus.unsubscribe(sub)``.
    """

    __slots__ = ("topic_glob", "queue", "projection", "slice_key", "_active")

    def __init__(
        self,
        topic_glob: str,
        *,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
        projection: Callable[[PresenceEvent], dict] | None = None,
        slice_key: str = "",
    ) -> None:
        self.topic_glob = topic_glob
        self.queue: asyncio.Queue[PresenceEvent | None] = asyncio.Queue(maxsize=queue_size)
        self.projection = projection
        self.slice_key = slice_key
        self._active = True

    def matches(self, topic: str) -> bool:
        """Glob match. ``state.*`` matches ``state.transition`` and
        ``state.tick``; ``**`` matches everything."""
        return fnmatch.fnmatch(topic, self.topic_glob)

    def close(self) -> None:
        """Mark inactive and unblock any awaiter on the queue."""
        self._active = False
        try:
            self.queue.put_nowait(None)
        except asyncio.QueueFull:
            # Drain one to make room for the sentinel — the consumer
            # will see None on the next iteration regardless.
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self.queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    @property
    def is_active(self) -> bool:
        return self._active


# ── Bus ──────────────────────────────────────────────────────────────

class PresenceBus:
    """Process-wide async pub/sub broker.

    One instance per runtime. ``publish`` is fire-and-forget — it
    schedules delivery to each matching subscriber and returns
    immediately. ``subscribe`` returns a handle the caller drains via
    ``async for`` or ``await sub.queue.get()``.
    """

    def __init__(self) -> None:
        self._subscribers: list[Subscription] = []
        self._lock = asyncio.Lock()
        self._published_count = 0
        self._dropped_count = 0

    # ── Pub/Sub API ──────────────────────────────────────────────────

    async def publish(self, event: PresenceEvent) -> None:
        """Deliver an event to every matching subscriber.

        Slow subscribers don't block the publisher — they hit
        back-pressure and may drop low-priority events. Critical
        topics are never dropped.
        """
        self._published_count += 1
        # Snapshot subscribers so a concurrent unsubscribe doesn't
        # corrupt iteration mid-publish.
        async with self._lock:
            subs = [s for s in self._subscribers if s.is_active]

        for sub in subs:
            if not sub.matches(event.topic):
                continue
            self._deliver(sub, event)

    async def publish_topic(
        self,
        topic: str,
        payload: dict | None = None,
        *,
        source_companion_id: str = "",
        target_companion_id: str | None = None,
        propagation: str = PROP_FULL,
        owner_user_id: str = "",
    ) -> None:
        """Convenience: build a ``PresenceEvent`` and publish it.

        Used by adapters and behavior modules that don't want to import
        the event dataclass for every emit. Equivalent to constructing
        a ``PresenceEvent`` and calling :meth:`publish`.

        ``propagation`` — Synapse Layer §5 hint. Defaults to
        ``"full"`` for backwards compatibility; callers that want
        containment (narrative, coder, agentic) pass an explicit value.

        ``owner_user_id`` scopes the event to one user for ws_fanout
        filtering; default "" = global (delivered to everyone).
        """
        if propagation not in _VALID_PROPAGATIONS:
            propagation = PROP_FULL
        await self.publish(PresenceEvent(
            topic=topic,
            payload=payload or {},
            source_companion_id=source_companion_id,
            target_companion_id=target_companion_id,
            propagation=propagation,
            owner_user_id=owner_user_id,
        ))

    def _deliver(self, sub: Subscription, event: PresenceEvent) -> None:
        """Try to put event on subscriber queue, applying back-pressure
        policy on overflow.
        """
        try:
            sub.queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass

        # Queue is full. Decide based on topic priority.
        if event.topic.startswith(_CRITICAL_TOPIC_PREFIXES):
            # Critical event — drop the oldest low-priority event we can
            # find to make room. Walking the queue isn't free but
            # this only triggers when a subscriber is genuinely
            # behind, which is rare.
            self._make_room(sub)
            try:
                sub.queue.put_nowait(event)
                return
            except asyncio.QueueFull:
                # Still full — log and drop critical (worst case).
                log.warning(
                    "presence_bus_critical_dropped",
                    topic=event.topic,
                    slice_key=sub.slice_key,
                )
                self._dropped_count += 1
                return

        # Non-critical: drop this one
        self._dropped_count += 1
        log.debug("presence_bus_dropped", topic=event.topic, slice_key=sub.slice_key)

    def _make_room(self, sub: Subscription) -> None:
        """Scan the subscriber's queue for a low-priority event and
        drop it. Walks at most queue.maxsize entries.
        """
        items: list[PresenceEvent | None] = []
        dropped_low_priority = False
        while not sub.queue.empty():
            try:
                item = sub.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not dropped_low_priority and item is not None:
                topic = item.topic
                if any(fnmatch.fnmatch(topic, p) for p in _LOW_PRIORITY_PATTERNS):
                    dropped_low_priority = True
                    self._dropped_count += 1
                    continue  # don't re-enqueue
            items.append(item)
        # Re-enqueue (in order)
        for item in items:
            try:
                sub.queue.put_nowait(item)
            except asyncio.QueueFull:
                # Shouldn't happen since we just emptied it, but defensive
                break

    async def subscribe(
        self,
        topic_glob: str = "**",
        *,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
        projection: Callable[[PresenceEvent], dict] | None = None,
        slice_key: str = "",
    ) -> Subscription:
        """Register a subscriber. Returns a Subscription handle the
        caller drains. Caller MUST eventually call ``unsubscribe`` to
        avoid a leak.
        """
        sub = Subscription(
            topic_glob,
            queue_size=queue_size,
            projection=projection,
            slice_key=slice_key,
        )
        async with self._lock:
            self._subscribers.append(sub)
        log.debug("presence_bus_subscribed", glob=topic_glob, slice_key=slice_key)
        return sub

    async def unsubscribe(self, sub: Subscription) -> None:
        """Remove a subscription and unblock its consumer."""
        async with self._lock:
            try:
                self._subscribers.remove(sub)
            except ValueError:
                pass  # already removed
        sub.close()
        log.debug("presence_bus_unsubscribed", slice_key=sub.slice_key)

    # ── Diagnostics ──────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Bus health for telemetry."""
        return {
            "subscriber_count": sum(1 for s in self._subscribers if s.is_active),
            "published_total": self._published_count,
            "dropped_total": self._dropped_count,
        }


# ── Best-effort emit helper ────────────────────────────────────────────

async def emit_safe(
    app_state,
    topic: str,
    payload: dict | None = None,
    *,
    propagation: str = PROP_FULL,
) -> None:
    """Best-effort emit from a proxy route or stream handler.

    No-op when the companion runtime isn't attached to ``app_state``
    (runtime disabled, startup not yet complete). Swallows exceptions
    so observability never breaks the chat path.

    ``propagation`` — Synapse Layer §5 hint. Defaults to ``"full"``;
    callers that already know the chat mode should call
    :func:`emit_chat_turn_completed` instead, which derives the
    propagation from the mode and runs the salience pipeline.
    """
    runtime = getattr(app_state, "companion_runtime", None)
    if runtime is None:
        return
    try:
        await runtime.bus.publish_topic(
            topic,
            payload or {},
            source_companion_id=getattr(runtime, "companion_id", ""),
            propagation=propagation,
        )
    except Exception:
        log.debug("emit_safe_failed", topic=topic, exc_info=True)


async def emit_chat_turn_completed(
    app_state,
    *,
    mode: str,
    user_id: str,
    session_id: str,
    wire_format: str,
    stream: bool,
    user_text: str = "",
    assistant_text: str = "",
    propagation: str = "",
) -> None:
    """Emit ``chat.turn_completed`` + run the salience pipeline.

    Canonical chat-turn emission for proxy stream/non-stream handlers.
    Replaces the prior pattern of calling ``emit_safe`` directly so
    that:

    1. The mode → propagation default is derived in one place rather
       than duplicated across four call sites.
    2. The salience scorer (Synapse Layer §1) runs after the turn
       lands and emits ``chat.moment_observed`` when the moment is
       worth Becca remembering. Gated by ``companion_salience_enabled``
       and threshold-checked.

    Both events are emitted with the same ``propagation``, so a
    subscriber that ignores ``factual_only`` events on
    ``chat.turn_completed`` also ignores them on
    ``chat.moment_observed`` — the policy is consistent.

    ``propagation`` empty → derived from ``mode`` via
    :func:`propagation_for_mode`. Callers that have an explicit
    per-session override pass it directly.
    """
    runtime = getattr(app_state, "companion_runtime", None)
    if runtime is None:
        return

    effective_prop = propagation if propagation in _VALID_PROPAGATIONS else propagation_for_mode(mode)

    try:
        await runtime.bus.publish_topic(
            "chat.turn_completed",
            {
                "mode": mode,
                "user_id": user_id,
                "session_id": session_id,
                "wire_format": wire_format,
                "stream": stream,
                "content_len": len(assistant_text or ""),
            },
            source_companion_id=getattr(runtime, "companion_id", ""),
            propagation=effective_prop,
            owner_user_id=user_id,
        )
    except Exception:
        log.debug("emit_chat_turn_completed_failed", exc_info=True)

    # Salience pipeline — runs after the turn lands. Skipped on
    # private propagation (we promised not to look) and when the
    # feature flag is off (default off — opt-in interior wiring).
    if effective_prop == PROP_PRIVATE:
        return
    if not user_text or not assistant_text:
        return

    try:
        from augmentum.config import settings
        if not getattr(settings, "companion_salience_enabled", False):
            return
        from augmentum.companion_runtime import salience
        moment = await salience.score(
            user_text=user_text,
            assistant_text=assistant_text,
            mode=mode,
            propagation=effective_prop,
        )
        if moment is None:
            return
        threshold = float(getattr(settings, "companion_salience_journal_threshold", 0.55))
        if moment.salience < threshold:
            return
        # Optional rewrite in Becca's voice. ``enrich_with_llm`` self-
        # gates on companion_salience_llm_enabled + the rewrite-floor
        # salience score and returns ``(moment, rewritten)`` — the
        # original on any failure path, with rewritten=False so the
        # observer keeps it as interior-only rather than pushing a
        # raw extract to the user pip.
        moment, rewritten = await salience.enrich_with_llm(
            runtime,
            moment,
            user_text=user_text,
            assistant_text=assistant_text,
        )
        # The moment carries its own derived propagation (affect_only
        # strips content text). Re-publish with the same hint so the
        # observer applies its rules uniformly.
        await runtime.bus.publish_topic(
            "chat.moment_observed",
            {
                "mode": mode,
                "user_id": user_id,
                "session_id": session_id,
                "salience": moment.salience,
                "moment": moment.text,
                "user_affect": moment.user_affect,
                "rewritten": rewritten,
            },
            source_companion_id=getattr(runtime, "companion_id", ""),
            propagation=effective_prop,
        )
    except Exception:
        log.debug("salience_pipeline_failed", exc_info=True)


async def emit_voice_turn_ended(
    runtime,
    *,
    user_id: str,
    session_id: str,
    invocation_id: str,
    transcript: str,
    assistant_text: str,
    affect_hint: str = "",
    propagation: str = PROP_FULL,
) -> None:
    """Emit ``voice.turn_ended`` after a BeccaVoice turn closes.

    Synapse Layer §3 — *the kept thing*. The personality doc says
    *"she is whatever survives across the consolidation step"*; the
    voice channel is where she most clearly *is herself*, and until
    this hook landed those turns evaporated when the WebSocket
    closed.

    Pipeline mirror of :func:`emit_chat_turn_completed`:

    1. Score the turn for salience (respects propagation — voice is
       almost always ``"full"`` since it's her own channel, but
       containment is honored anyway).
    2. When the score clears ``companion_salience_journal_threshold``
       emit ``voice.turn_ended`` with the moment summary + affect.
    3. The observer journals it as a ``conversation_moment`` with
       ``source='voice_turn'`` — distinguishable from chat-derived
       moments in retrospection.

    ``affect_hint`` is the runtime's *last published* affect tag — see
    ``CompanionRuntime._last_affect_tag`` — read at speak time. Empty
    when nothing fired; the salience scorer falls back to a
    transcript-derived tag.

    Gated by ``companion_voice_journal_enabled`` (default False).
    When off, the legacy ``voice.completed`` event continues to fire
    unchanged from voice.py; this helper is purely additive.

    ``runtime`` is the live :class:`CompanionRuntime` (voice.py
    already holds a reference via ``self._runtime``). Unlike
    :func:`emit_chat_turn_completed` — which takes ``app_state``
    because the proxy routes don't have a runtime in scope — voice
    callers pass the runtime directly.
    """
    if runtime is None:
        return
    if propagation == PROP_PRIVATE:
        return
    if not transcript and not assistant_text:
        return

    try:
        from augmentum.config import settings
        if not getattr(settings, "companion_voice_journal_enabled", False):
            return
        from augmentum.companion_runtime import salience
        moment = await salience.score(
            user_text=transcript or "",
            assistant_text=assistant_text or "",
            mode="voice",
            propagation=propagation,
        )
        if moment is None:
            return
        threshold = float(getattr(settings, "companion_salience_journal_threshold", 0.55))
        if moment.salience < threshold:
            return
        # Optional rewrite in Becca's voice. Same gating as the chat
        # path — see :func:`emit_chat_turn_completed`. Voice extracts
        # are even more quote-like than chat extracts (transcripts
        # carry filler), so the rewrite materially improves the note.
        moment, rewritten = await salience.enrich_with_llm(
            runtime,
            moment,
            user_text=transcript or "",
            assistant_text=assistant_text or "",
        )
        # Affect resolution: published runtime affect wins when present
        # (it reflects what Becca was actually *feeling* while speaking).
        # Fall back to the scorer's transcript-derived read.
        effective_affect = (affect_hint or "").strip() or moment.user_affect
        await runtime.bus.publish_topic(
            "voice.turn_ended",
            {
                "user_id": user_id,
                "session_id": session_id,
                "invocation_id": invocation_id,
                "salience": moment.salience,
                "moment": moment.text,
                "user_affect": effective_affect,
                "rewritten": rewritten,
                # Light excerpt of her own response — content_refs point
                # at the voice session for the resolver to rehydrate.
                "assistant_excerpt": (assistant_text or "")[:240],
            },
            source_companion_id=getattr(runtime, "companion_id", ""),
            propagation=propagation,
            owner_user_id=user_id,
        )
    except Exception:
        log.debug("voice_turn_pipeline_failed", exc_info=True)


async def maybe_emit_mode_changed(
    app_state,
    session_id: str,
    new_mode: str,
    *,
    reason: str = "",
    confidence: float = 1.0,
) -> None:
    """Emit ``mode.changed`` when mode differs from prior turn for this session.

    Per-session last-mode is stashed on the runtime so it lives with the
    observer state, not the proxy. No-op when runtime is missing or
    ``session_id`` is empty.
    """
    runtime = getattr(app_state, "companion_runtime", None)
    if runtime is None or not session_id:
        return
    last_modes = getattr(runtime, "_last_chat_mode_by_session", None)
    if not isinstance(last_modes, OrderedDict):
        # Bounded LRU (was an unbounded dict that grew one entry per
        # distinct session_id forever — audit 2026-06-17).
        last_modes = OrderedDict()
        runtime._last_chat_mode_by_session = last_modes
    prev = last_modes.get(session_id)
    if prev == new_mode:
        last_modes.move_to_end(session_id)  # keep active sessions warm
        return
    last_modes[session_id] = new_mode
    last_modes.move_to_end(session_id)
    # Evict the least-recently-used session past the cap. Re-emits one
    # mode.changed harmlessly if an evicted session returns.
    while len(last_modes) > _MODE_CACHE_CAP:
        last_modes.popitem(last=False)
    try:
        await runtime.bus.publish_topic(
            "mode.changed",
            {
                "from": prev,
                "to": new_mode,
                "session_id": session_id,
                "reason": reason,
                "confidence": confidence,
            },
            source_companion_id=getattr(runtime, "companion_id", ""),
        )
    except Exception:
        log.warning("mode_changed_emit_failed", exc_info=True)


# ── WebSocket transport adapter ───────────────────────────────────────

async def ws_fanout(
    websocket,  # FastAPI WebSocket (avoid hard import to keep this file dep-light)
    bus: PresenceBus,
    *,
    topic_glob: str = "**",
    slice_key: str = "",
    owner_user_id: str = "",
) -> None:
    """Pump bus events into a WebSocket. Returns when the socket closes.

    Use from a FastAPI ``@router.websocket`` handler. Accepts the
    socket, registers a subscription, forwards each event as a JSON
    message, and tears down cleanly on disconnect.

    The handler is the one place that imports FastAPI types; the bus
    itself stays framework-agnostic.

    ``owner_user_id`` scopes the stream to one user: events whose
    ``payload['user_id']`` names a *different* user are dropped, so a
    logged-in client can't read another tenant's chat/voice/surface
    content (audit 2026-06-17). Events with no user in their payload
    (global state/behavior the presence widget needs) and events
    matching the owner pass through. Empty owner = no filter (legacy
    single-user / unowned).
    """
    await websocket.accept()
    sub = await bus.subscribe(topic_glob, slice_key=slice_key or "ws")
    log.info("companion_ws_connected", glob=topic_glob, slice_key=slice_key)
    try:
        while True:
            event = await sub.queue.get()
            if event is None:
                break  # close sentinel
            if owner_user_id:
                # Prefer the native event owner; fall back to a payload
                # user_id sniff for emitters not yet stamping the field
                # (the union is strictly safer than either alone — audit
                # 2026-06-17). "" / absent = global → always delivered.
                ev_user = getattr(event, "owner_user_id", "") or ""
                if not ev_user:
                    try:
                        ev_user = (
                            event.payload.get("user_id")
                            if isinstance(event.payload, dict) else ""
                        ) or ""
                    except Exception:
                        ev_user = ""
                if ev_user and ev_user != owner_user_id:
                    continue  # another user's event — never forward it
            try:
                await websocket.send_text(event.to_json())
            except Exception as exc:
                log.info("companion_ws_send_failed", error=str(exc)[:200])
                break
    finally:
        await bus.unsubscribe(sub)
        log.info("companion_ws_disconnected", slice_key=slice_key)
        # Close only if the WS isn't already torn down. Starlette
        # raises RuntimeError("Cannot call 'send' once a close
        # message has been sent.") on a duplicate close, which would
        # otherwise spam a full rich traceback on every reconnect.
        # The WebSocketState enum exposes the lifecycle so we can
        # cheaply detect "already closed by the framework or the
        # peer" and skip the close.
        try:
            from starlette.websockets import WebSocketState
            already_closed = (
                websocket.application_state == WebSocketState.DISCONNECTED
                or websocket.client_state == WebSocketState.DISCONNECTED
            )
        except Exception:
            # Defensive: if the state lookup itself fails, fall back
            # to attempting the close — the original except-Exception
            # still catches the duplicate-close error.
            already_closed = False
        if not already_closed:
            try:
                await websocket.close()
            except RuntimeError:
                # Duplicate close — Starlette already raised the close
                # message via its own teardown path. Logged at debug
                # because it's expected during normal shutdown.
                log.debug("companion_ws_close_already_sent")
            except Exception:
                log.warning("companion_ws_close_failed", exc_info=True)


__all__ = [
    "PresenceBus",
    "PresenceEvent",
    "Subscription",
    "ws_fanout",
    "emit_safe",
    "emit_chat_turn_completed",
    "emit_voice_turn_ended",
    "maybe_emit_mode_changed",
    "propagation_for_mode",
    "PROP_FULL",
    "PROP_AFFECT_ONLY",
    "PROP_FACTUAL_ONLY",
    "PROP_PRIVATE",
]
