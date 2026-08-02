"""Verb dispatcher — formalizes the management-verb subscription pattern
on top of PresenceBus.

This module is the Phase 2 deliverable from
``docs/superpowers/specs/2026-06-05-companion-verbs-architecture-design
.md``. Per the Phase 1 audit at
``docs/superpowers/working/companion_management_verb_catalog.md`` and
the Phase 2 non-duplication audit, the existing
``augmentum/companion_runtime/bus.py`` (PresenceBus) already provides
the pub/sub substrate, propagation hints, back-pressure, and WS
fanout. This module DOES NOT replace it. Instead, it layers a verb-
typed dispatcher on top:

- One master subscription on PresenceBus (``**`` glob)
- Verbs register their topic interests + policy metadata
- Dispatcher routes incoming events to matching verbs with policy
  enforcement (cooldown, cost envelope, chain depth, autonomy gate)
- Every invocation lands a row in ``companion_verb_log`` (migration
  247) regardless of outcome

The dispatcher's eight load-bearing policy decisions:

1. **Cooldown** — SQL-backed via ``companion_verb_log``. Survives
   restarts. Replaces the eight scattered ``_last_*_at`` module-globals
   catalogued in Phase 1.
2. **Cost envelope** — wallclock + db_ops ceilings per invocation;
   exceeding kills the coroutine and records ``budget_exceeded``.
3. **Chain depth limit** — events carry ``chain_depth``; verbs that
   would push chain past
   ``dispatch_policy.DEFAULT_CHAIN_DEPTH_LIMIT`` are skipped.
4. **Autonomy gate** — ONE dispatcher pre-check per fanout that reads
   ``presence_mode.autonomy_allowed()``. Replaces the eight call-site
   reads catalogued in Phase 1 (curator, wondering, today, standing_
   tasks, pre_context, companion_routes×3). EVENT_DRIVEN and user-
   facing verbs gate; TICK_ALIGNED READ verbs continue (invisible
   maintenance).
5. **Auto-pause** — after ``DEFAULT_MAX_CONSECUTIVE_ERRORS`` failures,
   the verb is paused. Mirrors standing_tasks behaviour.
6. **Args-hash dedup** — within the coalesce window, identical
   payloads to the same verb are coalesced to one invocation.
7. **Propagation respect** — verb-emitted bus events inherit the
   propagation hint of the triggering event (defaults to PROP_FULL).
8. **Cite-self provenance** — every dispatch records which substrate
   tables were read/written and which parent verb_log_id caused it.

Phase 2 ships the dispatcher mechanics. Verb subscriptions and bodies
ship in Phase 3a (renames) and 3b (first-time writes).
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from augmentum.companion_runtime import presence_mode
from augmentum.companion_runtime.bus import (
    PROP_FULL,
    PresenceEvent,
)
from augmentum.companion_runtime.dispatch_policy import (
    DEFAULT_CHAIN_DEPTH_LIMIT,
    DEFAULT_COALESCE_WINDOW_S,
    DEFAULT_MAX_CONSECUTIVE_ERRORS,
    DEFAULT_MAX_DB_OPS,
    DEFAULT_MAX_WALLCLOCK_MS,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# ───── Enums ─────────────────────────────────────────────────────────


class VerbClass(str, Enum):
    """Top-level taxonomy from the spec."""
    MANAGEMENT = "management"
    CORE = "core"


class SafetyClass(str, Enum):
    """What kinds of writes the verb performs.

    READ            — pure read; no mutation
    WRITE_SELF      — writes companion-internal substrate only
    WRITE_USER      — writes user-visible state (journal, notifications)
    NO_USER_FACING  — internal maintenance only; never user-surfacing
    """
    READ = "READ"
    WRITE_SELF = "WRITE_SELF"
    WRITE_USER = "WRITE_USER"
    NO_USER_FACING = "NO_USER_FACING"


class DispatchClass(str, Enum):
    """When the verb is allowed to run vs presence_mode.

    IDLE_OK         — Always runs; doesn't care about autonomy gate
    TICK_ALIGNED    — Time-tick verbs; continue in DND when READ-safety
    EVENT_DRIVEN    — Event-fired verbs; gated by autonomy_allowed()
    """
    IDLE_OK = "IDLE_OK"
    TICK_ALIGNED = "TICK_ALIGNED"
    EVENT_DRIVEN = "EVENT_DRIVEN"


class VerbOutcome(str, Enum):
    """Recorded on every dispatch attempt. Drives Phase 5 observability."""
    OK = "ok"
    COOLDOWN_SKIPPED = "cooldown_skipped"
    BUDGET_EXCEEDED = "budget_exceeded"
    AUTONOMY_GATED = "autonomy_gated"
    CHAIN_DEPTH_EXCEEDED = "chain_depth_exceeded"
    ERROR = "error"
    AUTO_PAUSED = "auto_paused"
    DEDUPED = "deduped"


# ───── Cost envelope ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CostEnvelope:
    """Per-invocation ceiling. Wallclock is enforced via asyncio
    timeout; db_ops is post-hoc (verb is responsible for self-counting
    or the dispatcher infers from connection metrics — Phase 5 wires
    aiosqlite hooks)."""
    max_wallclock_ms: int = DEFAULT_MAX_WALLCLOCK_MS
    max_db_ops: int = DEFAULT_MAX_DB_OPS

    @classmethod
    def unlimited(cls) -> CostEnvelope:
        # Use after explicit verb-author opt-out (e.g. nightly
        # baseline rebuild that legitimately runs >5s).
        return cls(max_wallclock_ms=2_147_483_647, max_db_ops=2_147_483_647)


# ───── Event envelope ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class VerbEvent:
    """The dispatcher's typed envelope around a PresenceBus event.

    PresenceEvent carries topic + payload + companion ids + propagation.
    VerbEvent additionally carries chain bookkeeping the dispatcher
    needs (parent verb_log row, chain depth, event-id for dedup).
    """
    topic: str
    payload: dict[str, Any]
    event_id: str
    user_id: str = ""
    companion_id: str = ""
    chain_depth: int = 0
    parent_verb_log_id: int | None = None
    source_companion_id: str = ""
    propagation: str = PROP_FULL

    @classmethod
    def from_presence(
        cls,
        ev: PresenceEvent,
        *,
        owner_user_id: str = "",
        companion_id: str = "",
    ) -> VerbEvent:
        """Wrap an incoming PresenceEvent for verb dispatch.

        chain_depth and parent_verb_log_id are read from payload when
        present (verb-emitted events stamp them); 0 / None otherwise.
        """
        payload = dict(ev.payload or {})
        depth = int(payload.pop("_chain_depth", 0))
        parent_id = payload.pop("_parent_verb_log_id", None)
        return cls(
            topic=ev.topic,
            payload=payload,
            event_id=str(uuid.uuid4()),
            user_id=owner_user_id,
            companion_id=companion_id or getattr(ev, "target_companion_id", "") or "",
            chain_depth=depth,
            parent_verb_log_id=parent_id,
            source_companion_id=getattr(ev, "source_companion_id", "") or "",
            propagation=getattr(ev, "propagation", PROP_FULL),
        )


def hash_payload(payload: dict[str, Any]) -> str:
    """Stable hash of a payload for dedup-within-coalesce. Sorted keys
    so logically-equal payloads collide regardless of insertion order.
    Failures degrade to empty hash (no dedup) rather than raising."""
    try:
        canonical = json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        return ""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


# ───── ManagementVerb dataclass ───────────────────────────────────────


VerbHandler = Callable[[VerbEvent, "VerbContext"], Awaitable[None]]


@dataclass(slots=True)
class ManagementVerb:
    """A registered management verb.

    Use the @verb_register helper or the dispatcher's .register()
    method to install. Subscriptions are topic globs in fnmatch syntax
    matching PresenceBus's existing convention.
    """
    name: str
    handler: VerbHandler
    subscribes_to: tuple[str, ...]
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    can_invoke_core: tuple[str, ...] = ()
    cost_envelope: CostEnvelope = field(default_factory=CostEnvelope)
    safety_class: SafetyClass = SafetyClass.READ
    dispatch_class: DispatchClass = DispatchClass.TICK_ALIGNED
    cooldown_ms: int = 0

    def matches(self, topic: str) -> bool:
        return any(fnmatch.fnmatch(topic, glob) for glob in self.subscribes_to)


# ───── VerbContext — what verbs receive on invocation ────────────────


@dataclass(slots=True)
class VerbContext:
    """Handed to every verb on dispatch. Carries the runtime plus
    helpers for emitting events with provenance (chain_depth + parent
    log id auto-stamped so verbs can't accidentally break chain depth
    tracking)."""
    runtime: CompanionRuntime
    verb: ManagementVerb
    event: VerbEvent
    verb_log_id: int
    # Substrate citations populated by the verb body; recorded into
    # cited_substrate on outcome.
    cited_substrate: list[dict[str, Any]] = field(default_factory=list)
    # db_ops the verb claims to have performed (Phase 5 will replace
    # with aiosqlite hook).
    db_ops: int = 0

    def cite(self, table: str, row_id: int | str | None = None) -> None:
        """Record a substrate reference for the verb_log cited_substrate
        column. Verbs SHOULD call this for every meaningful read/write."""
        entry: dict[str, Any] = {"table": table}
        if row_id is not None:
            entry["row_id"] = row_id
        self.cited_substrate.append(entry)

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        propagation: str = PROP_FULL,
    ) -> None:
        """Emit a follow-up event on the bus with chain depth + parent
        log id stamped. Other verbs subscribed to this topic will be
        dispatched at chain_depth + 1; if that exceeds the configured
        limit they're skipped."""
        bus = self.runtime.bus
        enriched = dict(payload)
        enriched["_chain_depth"] = self.event.chain_depth + 1
        enriched["_parent_verb_log_id"] = self.verb_log_id
        await bus.publish_topic(
            topic, enriched,
            source_companion_id=self.runtime.companion_id,
            propagation=propagation,
        )


# ───── VerbDispatcher ─────────────────────────────────────────────────


class VerbDispatcher:
    """Routes PresenceBus events to registered management verbs with
    policy enforcement.

    Lifecycle: ``register()`` verbs, then ``await start()`` to open the
    bus subscription, ``await stop()`` to close.
    """

    def __init__(
        self,
        runtime: CompanionRuntime,
        *,
        chain_depth_limit: int = DEFAULT_CHAIN_DEPTH_LIMIT,
        coalesce_window_s: float = DEFAULT_COALESCE_WINDOW_S,
        max_consecutive_errors: int = DEFAULT_MAX_CONSECUTIVE_ERRORS,
    ) -> None:
        self._runtime = runtime
        self._verbs: dict[str, ManagementVerb] = {}
        # Keyed by (user_id, verb_name) so one user's failing verb pauses
        # it only for THAT user, not everyone (audit 2026-06-17). Bounded
        # by the OK-path cleanup that pops a (user,verb) on success.
        self._error_counts: dict[tuple[str, str], int] = {}
        self._paused: set[tuple[str, str]] = set()
        self._recent_args_hashes: dict[str, tuple[float, str]] = {}
        self._chain_depth_limit = chain_depth_limit
        self._coalesce_window_s = coalesce_window_s
        self._max_consecutive_errors = max_consecutive_errors
        self._subscription = None
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()

    # ---- Registration ----------------------------------------------------

    def register(self, verb: ManagementVerb) -> None:
        if verb.name in self._verbs:
            raise ValueError(f"verb already registered: {verb.name}")
        self._verbs[verb.name] = verb
        log.info(
            "verb_registered",
            name=verb.name,
            subscribes_to=list(verb.subscribes_to),
            safety_class=verb.safety_class.value,
            dispatch_class=verb.dispatch_class.value,
        )

    def get(self, name: str) -> ManagementVerb | None:
        return self._verbs.get(name)

    def names(self) -> list[str]:
        return list(self._verbs.keys())

    def is_paused(self, name: str, *, user_id: str = "") -> bool:
        return (user_id, name) in self._paused

    def resume(self, name: str, *, user_id: str = "") -> None:
        """Operator-facing un-pause. Resets the error counter so the verb
        gets a fresh window. With ``user_id`` empty, clears the pause for
        EVERY user (the "unstick everything" operator action); with a
        specific user_id, clears just that (user, verb) pair."""
        if user_id:
            self._paused.discard((user_id, name))
            self._error_counts.pop((user_id, name), None)
            return
        for pk in [k for k in self._paused if k[1] == name]:
            self._paused.discard(pk)
        for pk in [k for k in self._error_counts if k[1] == name]:
            self._error_counts.pop(pk, None)

    # ---- Lifecycle -------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            return
        bus = self._runtime.bus
        self._subscription = await bus.subscribe(
            "**", queue_size=1024, slice_key="verb_dispatcher",
        )
        self._stopped.clear()
        self._task = asyncio.create_task(self._run(), name="verb_dispatcher")
        log.info("verb_dispatcher_started", verbs=len(self._verbs))

    async def stop(self) -> None:
        self._stopped.set()
        if self._subscription is not None:
            self._subscription.close()  # sync — places None sentinel
            self._subscription = None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        log.info("verb_dispatcher_stopped")

    # ---- Run loop --------------------------------------------------------

    async def _run(self) -> None:
        """Consume from the Subscription's queue. PresenceBus places a
        ``None`` sentinel on close() — exit cleanly when we see it."""
        sub = self._subscription
        assert sub is not None
        try:
            while not self._stopped.is_set():
                ev = await sub.queue.get()
                if ev is None or self._stopped.is_set():
                    break
                await self._on_event(ev)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("verb_dispatcher_run_failed")

    async def _on_event(self, ev: PresenceEvent) -> None:
        """One PresenceBus event arrives → fan to matching verbs.

        Self-emitted events (source_companion_id == own_id) include both
        legitimate substrate sources (the tick ladder, verb ctx.emit())
        and would-be loops. The chain_depth carried in the event payload
        is the guardrail — events at depth >= ``_chain_depth_limit`` are
        skipped per-verb at line 436 with a CHAIN_DEPTH_EXCEEDED outcome
        for observability. Letting self-emitted events pass through here
        is what enables intentional verb→verb wiring (propose_action →
        enqueue_proposed_action, emit_pad_if_delta → narrate_state_to_user,
        etc.) — the design promise in ``VerbContext.emit()``'s docstring.
        """
        verb_event = VerbEvent.from_presence(
            ev,
            owner_user_id=getattr(self._runtime, "owner_user_id", "") or "",
            companion_id=self._runtime.companion_id,
        )

        # Autonomy gate (ONE read per fanout pass, not per verb).
        autonomy_ok = presence_mode.autonomy_allowed()

        for verb in list(self._verbs.values()):
            if not verb.matches(ev.topic):
                continue
            await self._dispatch(verb, verb_event, autonomy_ok=autonomy_ok)

    # ---- Per-verb dispatch with policy ----------------------------------

    async def _dispatch(
        self,
        verb: ManagementVerb,
        event: VerbEvent,
        *,
        autonomy_ok: bool,
    ) -> None:
        if (event.user_id, verb.name) in self._paused:
            await self._record(verb, event, VerbOutcome.AUTO_PAUSED, latency_ms=0)
            return

        # Chain depth.
        if event.chain_depth >= self._chain_depth_limit:
            await self._record(
                verb, event, VerbOutcome.CHAIN_DEPTH_EXCEEDED, latency_ms=0,
            )
            return

        # Autonomy.
        if not self._autonomy_ok_for_verb(verb, autonomy_ok):
            await self._record(verb, event, VerbOutcome.AUTONOMY_GATED, latency_ms=0)
            return

        # Cooldown — SQL-backed; survives restarts.
        if verb.cooldown_ms > 0:
            recent_at = await self._last_fired_at(verb.name, event)
            if recent_at is not None:
                age_ms = int((time.time() - recent_at) * 1000)
                if age_ms < verb.cooldown_ms:
                    await self._record(
                        verb, event, VerbOutcome.COOLDOWN_SKIPPED, latency_ms=0,
                    )
                    return

        # Args-hash dedup — same payload to same verb within coalesce
        # window collapses to one invocation.
        args_hash = hash_payload(event.payload)
        cache_key = f"{verb.name}:{args_hash}"
        now = time.time()
        last = self._recent_args_hashes.get(cache_key)
        if last is not None and (now - last[0]) < self._coalesce_window_s:
            await self._record(verb, event, VerbOutcome.DEDUPED, latency_ms=0)
            return
        self._recent_args_hashes[cache_key] = (now, args_hash)
        # Cheap cache GC — drop entries older than 10x coalesce window.
        if len(self._recent_args_hashes) > 256:
            stale_threshold = now - (self._coalesce_window_s * 10)
            for k, (t, _) in list(self._recent_args_hashes.items()):
                if t < stale_threshold:
                    self._recent_args_hashes.pop(k, None)

        # Pre-record the verb_log row so the verb body can reference its
        # own id when emitting follow-up events (chain provenance).
        verb_log_id = await self._record(
            verb, event, VerbOutcome.OK, latency_ms=0,
            pre_record=True, args_hash=args_hash,
        )

        ctx = VerbContext(
            runtime=self._runtime,
            verb=verb,
            event=event,
            verb_log_id=verb_log_id,
        )

        # Cost envelope — wallclock enforced via asyncio.wait_for.
        # db_ops is recorded post-hoc from ctx.db_ops; Phase 5 will
        # automate via aiosqlite hooks.
        started = time.monotonic()
        outcome: VerbOutcome = VerbOutcome.OK
        error_msg: str = ""
        try:
            timeout_s = max(0.001, verb.cost_envelope.max_wallclock_ms / 1000.0)
            await asyncio.wait_for(verb.handler(event, ctx), timeout=timeout_s)
            if ctx.db_ops > verb.cost_envelope.max_db_ops:
                outcome = VerbOutcome.BUDGET_EXCEEDED
                error_msg = f"db_ops={ctx.db_ops} > {verb.cost_envelope.max_db_ops}"
        except TimeoutError:
            outcome = VerbOutcome.BUDGET_EXCEEDED
            error_msg = f"wallclock > {verb.cost_envelope.max_wallclock_ms}ms"
        except Exception as exc:
            outcome = VerbOutcome.ERROR
            error_msg = str(exc)[:200]
            log.warning("verb_dispatch_error", verb=verb.name, error=error_msg)
        latency_ms = int((time.monotonic() - started) * 1000)

        # Update the pre-recorded row with the actual outcome.
        await self._update_outcome(
            verb_log_id,
            outcome=outcome,
            latency_ms=latency_ms,
            error=error_msg,
            db_ops=ctx.db_ops,
            cited_substrate=ctx.cited_substrate,
        )

        # Error-count + auto-pause bookkeeping, keyed per (user, verb).
        pk = (event.user_id, verb.name)
        if outcome == VerbOutcome.OK:
            self._error_counts.pop(pk, None)
        else:
            self._error_counts[pk] = self._error_counts.get(pk, 0) + 1
            if self._error_counts[pk] >= self._max_consecutive_errors:
                self._paused.add(pk)
                log.warning(
                    "verb_auto_paused",
                    verb=verb.name,
                    user_id=event.user_id,
                    consecutive_errors=self._error_counts[pk],
                )

    # ---- Policy helpers --------------------------------------------------

    def _autonomy_ok_for_verb(
        self, verb: ManagementVerb, autonomy_ok: bool,
    ) -> bool:
        """Single source of truth for autonomy-gate decisions, replacing
        the eight scattered ``presence_mode.autonomy_allowed()`` reads
        in companion_routes / curator / wondering / today / standing_
        tasks / pre_context that the Phase 2 audit flagged.

        Policy:
          IDLE_OK            — always runs (system housekeeping)
          TICK_ALIGNED+READ  — runs during silent (invisible maintenance)
          everything else    — gated
        """
        if verb.dispatch_class == DispatchClass.IDLE_OK:
            return True
        if autonomy_ok:
            return True
        if (
            verb.dispatch_class == DispatchClass.TICK_ALIGNED
            and verb.safety_class == SafetyClass.READ
        ):
            return True
        return False

    async def _last_fired_at(
        self, verb_name: str, event: VerbEvent,
    ) -> float | None:
        """Read MAX(fired_at) from companion_verb_log for cooldown check.
        Returns None when the verb has never fired for this (user,
        companion) scope or the table doesn't exist yet."""
        try:
            conn = self._runtime.backend.conn
            cur = await conn.execute(
                """SELECT MAX(fired_at)
                   FROM companion_verb_log
                   WHERE user_id = ? AND companion_id = ?
                     AND verb_name = ?
                     AND outcome = ?""",
                (event.user_id, event.companion_id, verb_name, VerbOutcome.OK.value),
            )
            row = await cur.fetchone()
            await cur.close()
        except Exception:
            log.debug("verb_log_cooldown_lookup_failed", verb=verb_name)
            return None
        if not row or row[0] is None:
            return None
        return float(row[0])

    # ---- verb_log persistence -------------------------------------------

    async def _record(
        self,
        verb: ManagementVerb,
        event: VerbEvent,
        outcome: VerbOutcome,
        *,
        latency_ms: int,
        pre_record: bool = False,
        args_hash: str = "",
    ) -> int:
        """Insert a companion_verb_log row. Pre-record path returns the
        id so the verb body's emit() can stamp its children with the
        parent log id."""
        try:
            conn = self._runtime.backend.conn
            cur = await conn.execute(
                """INSERT INTO companion_verb_log
                      (user_id, companion_id, verb_name, verb_class,
                       dispatch_class, event_topic, event_id, args_hash,
                       outcome, latency_ms, cited_verb_log_id, fired_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           CAST(strftime('%s','now') AS INTEGER))""",
                (
                    event.user_id, event.companion_id, verb.name,
                    VerbClass.MANAGEMENT.value, verb.dispatch_class.value,
                    event.topic, event.event_id, args_hash or "",
                    outcome.value, latency_ms,
                    event.parent_verb_log_id,
                ),
            )
            row_id = int(cur.lastrowid or 0)
            await cur.close()
            await conn.commit()
            return row_id
        except Exception:
            log.warning(
                "verb_log_record_failed",
                verb=verb.name, outcome=outcome.value, exc_info=True,
            )
            return 0

    async def _update_outcome(
        self,
        verb_log_id: int,
        *,
        outcome: VerbOutcome,
        latency_ms: int,
        error: str,
        db_ops: int,
        cited_substrate: list[dict[str, Any]],
    ) -> None:
        if verb_log_id <= 0:
            return
        try:
            conn = self._runtime.backend.conn
            await conn.execute(
                """UPDATE companion_verb_log
                      SET outcome = ?, latency_ms = ?,
                          error = ?, db_ops = ?,
                          cited_substrate = ?
                    WHERE id = ?""",
                (
                    outcome.value, latency_ms, error or "", db_ops,
                    json.dumps(cited_substrate) if cited_substrate else "",
                    verb_log_id,
                ),
            )
            await conn.commit()
        except Exception:
            log.warning(
                "verb_log_update_failed",
                verb_log_id=verb_log_id, exc_info=True,
            )

    # ---- Introspection (Phase 5 observability hooks) --------------------

    def snapshot(self) -> dict[str, Any]:
        """Lightweight introspection for ``GET /api/companion/verb-
        dispatch/status``. Phase 5 will surface this in the operator
        panel."""
        return {
            "verbs_registered": len(self._verbs),
            # (user_id, verb) tuple keys → JSON-friendly shapes.
            "paused": [
                {"user_id": u, "verb": v} for (u, v) in sorted(self._paused)
            ],
            "error_counts": {
                f"{u}:{v}": n for (u, v), n in self._error_counts.items()
            },
            "running": self._task is not None and not self._task.done(),
        }


# ───── Helpers ───────────────────────────────────────────────────────


def verb(
    *subscribes_to: str,
    name: str | None = None,
    reads: tuple[str, ...] = (),
    writes: tuple[str, ...] = (),
    can_invoke_core: tuple[str, ...] = (),
    cost_envelope: CostEnvelope | None = None,
    safety_class: SafetyClass = SafetyClass.READ,
    dispatch_class: DispatchClass = DispatchClass.TICK_ALIGNED,
    cooldown_ms: int = 0,
) -> Callable[[VerbHandler], ManagementVerb]:
    """Decorator sugar for declaring a management verb.

    Usage::

        @verb("time.tick(60s)", reads=("companion_drive_state",),
              writes=("companion_drive_state",),
              dispatch_class=DispatchClass.TICK_ALIGNED,
              safety_class=SafetyClass.WRITE_SELF,
              cooldown_ms=55_000)
        async def tick_drive(event, ctx):
            ...

    Phase 3a uses this to migrate the eight TickLoop._tick subsystems
    into declared verbs.
    """
    def decorator(fn: VerbHandler) -> ManagementVerb:
        return ManagementVerb(
            name=name or fn.__name__,
            handler=fn,
            subscribes_to=subscribes_to,
            reads=reads,
            writes=writes,
            can_invoke_core=can_invoke_core,
            cost_envelope=cost_envelope or CostEnvelope(),
            safety_class=safety_class,
            dispatch_class=dispatch_class,
            cooldown_ms=cooldown_ms,
        )
    return decorator


__all__ = [
    "VerbClass", "SafetyClass", "DispatchClass", "VerbOutcome",
    "CostEnvelope", "VerbEvent", "ManagementVerb", "VerbContext",
    "VerbDispatcher", "verb", "hash_payload",
]
