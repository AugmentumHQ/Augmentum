"""CompanionRuntime — the top-level kernel.

Composes :class:`CompanionIdentity`, :class:`CompanionState`,
:class:`CompanionMemory`, and :class:`PresenceBus` into a single
addressable kernel. Sprint 1 ships the substrate: lifecycle, snapshot,
bus pass-throughs, and the stub dispatch path. Sprint 2 wires
subagents/primitives in; Sprint 3 makes ``submit_intent`` actually
route; Sprint 4a starts the tick loop.

Lifecycle is host-driven. The FastAPI lifespan in ``proxy/server.py``
instantiates the runtime when ``settings.companion_runtime_enabled``
is True, calls ``start()``, and ensures ``stop()`` runs on shutdown.

The kernel is inert until flag-gated subsystems flip on:
- ``companion_dispatch_enabled``: ``submit_intent`` actually routes
- ``companion_tick_enabled``: the autonomous tick loop ticks
- ``companion_journal_enabled``: the journal accepts writes
- and so on per the design spec's feature flag map

Design spec: ``2026-05-14-companion-runtime-design-v2.md`` §4.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.companion_runtime.bus import PresenceBus, PresenceEvent
from augmentum.companion_runtime.identity import CompanionIdentity
from augmentum.companion_runtime.memory import CompanionMemory
from augmentum.companion_runtime.state import CompanionState
from augmentum.utils.bg_tasks import track
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.memory.core_profile import CoreProfileManager
    from augmentum.memory.store import MemoryStore
    from augmentum.state.backends.sqlite import SQLiteBackend

log = get_logger(__name__)


# ── Intent shape (Sprint 1 stub; Sprint 3 fills out) ─────────────────

@dataclass(frozen=True, slots=True)
class Intent:
    """An input request flowing through the runtime.

    Sprint 1 carries only the bare minimum; Sprint 3's dispatch engine
    adds priority, privacy_context, source, explicit_subagent, etc.
    """
    text: str
    user_id: str
    source: str = "user_chat"          # user_voice | user_chat | tick | peer_agent
    device_id: str = ""
    explicit_mode: str = ""            # UI mode toggle hint (Sprint 3)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Response:
    """A dispatch result. Streaming responses arrive via the bus."""
    content: str
    handled_by: str                    # subagent or primitive name
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Kernel ───────────────────────────────────────────────────────────

class CompanionRuntime:
    """Top-level kernel for a single companion.

    Use when:
    - FastAPI lifespan needs to bring the runtime up.
    - A route wants to submit an intent or subscribe to the bus.
    - A debug surface wants a health snapshot.

    Concurrency: ``start()`` and ``stop()`` are serialized via a
    lifecycle lock. Per-axis state transitions serialize through
    :class:`CompanionState`'s own lock. The bus is concurrent-safe by
    design.
    """

    def __init__(
        self,
        backend: SQLiteBackend,
        *,
        companion_id: str = "becca",
        app_state: Any = None,
    ) -> None:
        self.companion_id = companion_id
        self.backend = backend
        # Loosely bound to the FastAPI app.state so adapters can reach
        # the shared resources (tool_registry, provider_registry,
        # container_manager, etc.) without an import cycle. The
        # underscore prefix marks it as runtime-internal — callers
        # outside the runtime package shouldn't rely on it.
        self._app_state = app_state
        # Piece 1 per-user pivot (mig 179): the canonical `self.identity`
        # / `self.state` bind to user_id='' — the migration-backfilled
        # legacy seed row. Per-user access goes through `get_identity`
        # / `get_state` which lazy-provisions and caches per-(user_id,
        # companion_id) instances.
        self.identity = CompanionIdentity(backend, companion_id, user_id="")
        self.state = CompanionState(backend, companion_id, user_id="")
        self.memory = CompanionMemory(backend, companion_id)
        self.bus = PresenceBus()
        # Per-user caches. Each user gets their own CompanionIdentity /
        # CompanionState instance bound to their (user_id, companion_id)
        # row. Populated on first get_*() call; never invalidated except
        # by an explicit reset gesture.
        self._identities_by_user: dict[str, CompanionIdentity] = {}
        self._states_by_user: dict[str, CompanionState] = {}
        self._provisioned_users: set[str] = set()
        # Personality facet substrate (migration 160, Lane 2 §1).
        # Imported lazily because the personality package may be absent
        # in minimal test fixtures.
        try:
            from augmentum.personality.store import PersonalityStore
            self.personality_store = PersonalityStore(backend)
        except Exception:
            self.personality_store = None
        # Channel session tracker (Sprint D — companion_runtime/channels.py).
        # Lazily initialized; channels._sessions(runtime) creates if missing.
        self._channel_sessions: dict[str, Any] = {}
        self._started = False
        self._stopping = False
        self._lifecycle_lock = asyncio.Lock()
        self._tick_task: asyncio.Task | None = None
        # Slice 0 resource gates (see companion_runtime/gates.py). Unix
        # timestamps in seconds; 0.0 = "no cooldown active". Input
        # adapters (voice STT finalize, chat send, PTT release) set
        # user_cooldown_until on observed activity. Heavy-tier
        # performers (image, audio) set heavy_quiet_until after a
        # successful run to provide cross-axis backpressure.
        self.user_cooldown_until: float = 0.0
        self.heavy_quiet_until: float = 0.0

        # Initiative-scoring interval cap (Piece 7'). The tick loop fires
        # every 5-30s depending on state. Initiative.step() costs ~4
        # SELECTs + optional INSERT — cheap individually but compounds
        # at high tick rates. Cap the actual scoring to at most one pass
        # per ``companion_initiative_min_interval_s`` (default 60s). Reads
        # are zero between caps; the timestamp lives here so a restart
        # forces a fresh score on the first eligible tick.
        self.last_initiative_score_at: float = 0.0

        # Synapse Layer §2 — user-observed affect tracker. Per-user PAD
        # estimate fed from salient chat + voice moments, decayed on
        # read. The voice composer reads this at speak time so she
        # conditions on the user's current weather without asking.
        # Half-life is config-tunable; constructor pulls the current
        # value (settings is mutable so a restart picks up changes).
        from augmentum.companion_runtime.perception.user_affect import UserAffectTracker
        try:
            from augmentum.config import settings as _settings
            _half_life = float(getattr(_settings, "companion_user_affect_half_life_s", 1800.0))
        except Exception:
            _half_life = 1800.0
        self.user_affect = UserAffectTracker(half_life_s=_half_life)

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(
        self,
        *,
        memory_store: MemoryStore | None = None,
        core_profile: CoreProfileManager | None = None,
    ) -> None:
        """Hydrate state from DB and wire in the existing memory
        subsystem. Idempotent — safe to call twice.
        """
        async with self._lifecycle_lock:
            if self._started:
                return
            log.info("companion_runtime_starting", companion_id=self.companion_id)

            # Load identity + state from their seeded rows (migrations 151+152+157)
            await self.identity.load()
            await self.state.load()

            # Resolve owner_user_id (migration 173). Single-companion phase:
            # the companion is owned by one human. Resolution order:
            #   1. ``companion_default_owner_user_id`` setting (explicit override)
            #   2. The bound owner already stored in companion_identities
            #   3. Auto-bind to the single non-empty user in the ``users`` table
            #      (so fresh installs Just Work after the first auth)
            # Resulting value is exposed as ``runtime.owner_user_id`` and
            # consumed by sleep_wake.invoke_dream + the drift audit scheduler.
            await self._resolve_owner_user_id()

            # Auto-refresh persona-kernel digest if empty. On a fresh
            # install the digest stays None until something explicitly
            # digests the on-disk personality doc — and BeccaVoice's
            # first gate is precisely ``if not digest: BeccaBypassed``,
            # so without this every voice turn falls through to the
            # legacy path. Tolerates a missing doc (tests / dev
            # fixtures) by logging and continuing; non-fatal.
            try:
                digest = (self.identity.persona_kernel_digest or "").strip()
                if not digest:
                    await self.identity.refresh_persona_kernel()
                    log.info(
                        "companion_persona_kernel_auto_refreshed",
                        companion_id=self.companion_id,
                    )
            except FileNotFoundError:
                log.warning(
                    "companion_persona_doc_missing",
                    companion_id=self.companion_id,
                    note=(
                        "BeccaVoice will bypass until /docs/superpowers/"
                        "specs/2026-05-14-{companion_id}-personality.md "
                        "exists or refresh_persona_kernel(force=True) "
                        "is invoked manually."
                    ),
                )
            except Exception:
                log.warning("companion_persona_kernel_refresh_failed", exc_info=True)

            # Seed the personality facet vocabulary (migration 160).
            # Idempotent — INSERT OR IGNORE — safe to call every boot.
            if self.personality_store is not None:
                try:
                    await self.personality_store.seed_vocabulary()
                except Exception:
                    log.warning("personality_seed_failed", exc_info=True)

            # Attach the existing memory subsystem if provided. Both are
            # optional here so the kernel can boot for tests without a
            # full app context — but in production the lifespan hook
            # always passes both.
            if memory_store is not None and core_profile is not None:
                self.memory.attach(memory_store, core_profile)
                log.debug("companion_memory_attached")
            else:
                log.warning(
                    "companion_memory_not_attached",
                    note="MemoryStore/CoreProfileManager refs were None — "
                         "memory operations will fail until attach() is called",
                )

            # Wire state-machine transitions to the bus so subscribers
            # (XR scene, smart-glasses, telemetry) see every change.
            self.state.subscribe("state", self._on_state_transition)
            self.state.subscribe("role", self._on_role_transition)
            self.state.subscribe("focus", self._on_focus_transition)

            # Trigger Sprint 2 adapter registration. Importing the
            # packages runs each adapter module's
            # ``Registry.register(cls)`` at import time. Adapters are
            # inert until their flag flips (companion_subagent_registry_active /
            # companion_primitive_registry_active).
            try:
                import augmentum.companion_runtime.primitives as _pr_pkg  # noqa: F401
                import augmentum.companion_runtime.subagents as _sa_pkg  # noqa: F401
            except Exception:
                log.warning("companion_adapter_import_failed", exc_info=True)

            # Sprint 4a — start the autonomous tick loop. It self-gates
            # on ``companion_tick_enabled``, so the task can run with
            # the flag off and simply returns each iteration. We start
            # it unconditionally here so flipping the flag at runtime
            # doesn't require a restart.
            try:
                from augmentum.companion_runtime.behavior.tick import TickLoop
                self._tick_loop = TickLoop(self)
                await self._tick_loop.start()
            except Exception:
                log.warning("companion_tick_loop_start_failed", exc_info=True)
                self._tick_loop = None

            # Observer — subscribes to the bus and maintains
            # ``observed_state`` so initiative scoring and (later) the
            # direct-address path can read "what's happening" without
            # polling every subsystem. She watches; she doesn't drive.
            try:
                from augmentum.companion_runtime.observer import BeccaObserver
                self._observer = BeccaObserver(self)
                await self._observer.start()
            except Exception:
                log.warning("companion_observer_start_failed", exc_info=True)
                self._observer = None

            # Phase 2 of the companion-verbs architecture — management-verb
            # dispatcher + multi-rate time-tick publisher. The dispatcher
            # owns cooldown / cost-envelope / chain-depth / autonomy-gate
            # enforcement and writes companion_verb_log. The tick ladder
            # publishes ``time.tick(<label>)`` events on wall-clock
            # boundaries so verbs subscribed via fnmatch globs fire. Both
            # are inert until Phase 3 verbs register against the
            # dispatcher; starting them here means the substrate is ready
            # the moment a verb decorator runs at import time.
            try:
                from augmentum.companion_runtime.event_bus import VerbDispatcher
                self._verb_dispatcher = VerbDispatcher(self)
                await self._verb_dispatcher.start()
            except Exception:
                log.warning("companion_verb_dispatcher_start_failed", exc_info=True)
                self._verb_dispatcher = None

            # Import the verbs package — this triggers @verb decorators
            # in each submodule, which append to VerbRegistry at module-
            # import time (mirrors SubagentRegistry / PrimitiveRegistry).
            # Feed every registered verb to the live dispatcher so it
            # starts routing on the next matching bus event.
            if self._verb_dispatcher is not None:
                try:
                    from augmentum.companion_runtime.verbs import VerbRegistry
                    for v in VerbRegistry.all():
                        # Re-register safely: a process restart imports
                        # the modules again so the same ManagementVerb
                        # instances flow back through, and the
                        # dispatcher rejects duplicates by name. Skip
                        # already-present names rather than crashing.
                        if self._verb_dispatcher.get(v.name) is None:
                            self._verb_dispatcher.register(v)
                    log.info(
                        "companion_verbs_registered",
                        count=len(self._verb_dispatcher.names()),
                    )
                except Exception:
                    log.warning(
                        "companion_verbs_registration_failed", exc_info=True,
                    )

            try:
                from augmentum.companion_runtime.tick_ladder import TickLadder
                self._tick_ladder = TickLadder(self)
                await self._tick_ladder.start()
            except Exception:
                log.warning("companion_tick_ladder_start_failed", exc_info=True)
                self._tick_ladder = None

            # Wire sleep/wake bridge: when state transitions out of
            # asleep/dreaming, surface the most recent dream into the
            # journal. Listener uses the runtime's state-machine
            # observer rather than the bus so we get the prior-state
            # value cleanly.
            self.state.subscribe("state", self._on_state_for_wake)

            # One-shot scrub of legacy placeholder notes. Before the
            # surface-eligibility tightening in observer.py/memory.py,
            # affect-only chat moments were journaled with the literal
            # text "a moment landed (affect: X); content not retained"
            # AND auto-flagged quiet_share_ready=1 because the affect
            # tag fell in the surfaceable set. Those rows aren't
            # actionable — by design the content was blanked — so the
            # pip just shows placeholder noise to existing users until
            # they ack/mute each one. Idempotent, bounded by the exact
            # prefix; non-fatal on failure.
            try:
                await self.backend.conn.execute(
                    "UPDATE companion_journal "
                    "SET quiet_share_ready = 0 "
                    "WHERE quiet_share_ready = 1 "
                    "  AND surfaced_at IS NULL "
                    "  AND content LIKE 'a moment landed (affect:%'",
                )
                await self.backend.conn.commit()
            except Exception:
                log.debug(
                    "companion_legacy_placeholder_scrub_failed",
                    exc_info=True,
                )

            self._started = True
            await self.bus.publish(PresenceEvent(
                topic="runtime.started",
                payload={"companion_id": self.companion_id},
                source_companion_id=self.companion_id,
            ))
            log.info("companion_runtime_started", companion_id=self.companion_id)

    async def stop(self, *, grace_seconds: float = 2.0) -> None:
        """Drain in-flight work and close the bus. Idempotent."""
        async with self._lifecycle_lock:
            if not self._started or self._stopping:
                return
            self._stopping = True
            log.info("companion_runtime_stopping", companion_id=self.companion_id)

            await self.bus.publish(PresenceEvent(
                topic="runtime.stopping",
                payload={"companion_id": self.companion_id},
                source_companion_id=self.companion_id,
            ))

            # Stop the autonomous tick loop (Sprint 4a)
            tick_loop = getattr(self, "_tick_loop", None)
            if tick_loop is not None:
                try:
                    await tick_loop.stop()
                except Exception:
                    log.warning("companion_tick_loop_stop_failed", exc_info=True)

            # Stop the observer
            observer = getattr(self, "_observer", None)
            if observer is not None:
                try:
                    await observer.stop()
                except Exception:
                    log.warning("companion_observer_stop_failed", exc_info=True)

            # Stop the tick ladder + verb dispatcher (Phase 2). Order:
            # ladder first so no new ticks land in the dispatcher's
            # subscription queue, then the dispatcher drains and exits.
            tick_ladder = getattr(self, "_tick_ladder", None)
            if tick_ladder is not None:
                try:
                    await tick_ladder.stop()
                except Exception:
                    log.warning("companion_tick_ladder_stop_failed", exc_info=True)
            verb_dispatcher = getattr(self, "_verb_dispatcher", None)
            if verb_dispatcher is not None:
                try:
                    await verb_dispatcher.stop()
                except Exception:
                    log.warning("companion_verb_dispatcher_stop_failed", exc_info=True)

            # Cancel the legacy tick task placeholder (kept for back-compat)
            if self._tick_task and not self._tick_task.done():
                self._tick_task.cancel()
                try:
                    await asyncio.wait_for(self._tick_task, timeout=grace_seconds)
                except (TimeoutError, asyncio.CancelledError):
                    pass

            self._started = False
            self._stopping = False
            log.info("companion_runtime_stopped", companion_id=self.companion_id)

    # ── Owner resolution (migration 173) ──────────────────────────────

    @property
    def owner_user_id(self) -> str:
        """The user_id this companion belongs to. Empty when unowned.

        Consumers: sleep_wake.invoke_dream (user-scoped dream cycles),
        the drift audit scheduler (which user's anchor to compare
        against), and any future user-scoped tick activity.
        """
        return self.identity.owner_user_id if self.identity._row else ""

    async def _resolve_owner_user_id(self) -> None:
        """Resolve and persist the companion's owner_user_id.

        Idempotent. Resolution order:
          1. Explicit setting ``companion_default_owner_user_id``
          2. Stored value from companion_identities (no-op happy path)
          3. Auto-bind to the single user in ``users`` when there's exactly one
        Logs the outcome so a "dreams still skipping" report is debuggable.
        """
        from augmentum.config import settings

        explicit = (
            getattr(settings, "companion_default_owner_user_id", "") or ""
        ).strip()
        current = self.identity.owner_user_id

        # (1) Explicit override always wins; (re)bind if different.
        if explicit and explicit != current:
            await self.identity.set_owner_user_id(explicit)
            await self.lazy_provision(explicit)
            log.info(
                "companion_owner_bound_explicit",
                companion_id=self.companion_id,
                owner_user_id=explicit,
            )
            return

        # (2) Already bound — provision per-user row defensively (cheap
        # idempotent no-op if it already exists, repairs partial state
        # from a process restart between mig 179 and first interaction).
        if current:
            await self.lazy_provision(current)
            log.debug(
                "companion_owner_already_bound",
                companion_id=self.companion_id,
                owner_user_id=current,
            )
            return

        # (3) Auto-bind when exactly one user exists. Multi-user installs
        #     must set ``companion_default_owner_user_id`` explicitly to
        #     avoid binding becca to whichever row sorted first.
        try:
            cursor = await self.backend.conn.execute(
                "SELECT id FROM users LIMIT 2",
            )
            rows = await cursor.fetchall()
            await cursor.close()
        except Exception:
            log.warning("companion_owner_users_query_failed", exc_info=True)
            return
        if len(rows) == 1:
            sole_user_id = rows[0][0]
            await self.identity.set_owner_user_id(sole_user_id)
            # Piece 1 — also provision their per-user row so per-user
            # APIs work immediately, not after the first call.
            await self.lazy_provision(sole_user_id)
            log.info(
                "companion_owner_auto_bound",
                companion_id=self.companion_id,
                owner_user_id=sole_user_id,
                note="single-user install",
            )
        elif len(rows) == 0:
            log.info(
                "companion_owner_unresolved_no_users",
                companion_id=self.companion_id,
                note="dreams + user-scoped activities will skip until first user signs up",
            )
        else:
            log.info(
                "companion_owner_unresolved_multi_user",
                companion_id=self.companion_id,
                note="set companion_default_owner_user_id to bind explicitly",
            )

    # ── Per-user access (Piece 1, Aletheia arc) ───────────────────────

    async def get_identity(self, user_id: str) -> CompanionIdentity:
        """Per-user identity lookup. Lazily provisions on first access.

        Returns the legacy seed identity (``self.identity``) when
        ``user_id`` is empty — preserves the pre-pivot API for code that
        hasn't yet adopted per-user scoping. New code paths should pass
        a real user_id from the request scope or intent.
        """
        if not user_id:
            return self.identity
        cached = self._identities_by_user.get(user_id)
        if cached is not None:
            return cached
        # Provision if first time; lazy_provision is idempotent + caches
        # the provisioned set so repeat calls are cheap.
        await self.lazy_provision(user_id)
        identity = CompanionIdentity(self.backend, self.companion_id, user_id=user_id)
        await identity.load()
        self._identities_by_user[user_id] = identity
        return identity

    async def get_state(self, user_id: str) -> CompanionState:
        """Per-user state lookup. Lazily provisions on first access.

        See :meth:`get_identity` for the empty-user_id fallback rationale.
        """
        if not user_id:
            return self.state
        cached = self._states_by_user.get(user_id)
        if cached is not None:
            return cached
        await self.lazy_provision(user_id)
        state = CompanionState(self.backend, self.companion_id, user_id=user_id)
        await state.load()
        self._states_by_user[user_id] = state
        return state

    async def lazy_provision(self, user_id: str) -> bool:
        """Ensure per-user companion rows exist for (user_id, companion_id).

        Idempotent — no-ops on repeat calls (cached in
        ``_provisioned_users``). Returns True when rows were actually
        created this call, False on no-op.

        What it provisions, copying from the seed row (user_id='') when
        available:
          - companion_identities row with seeded kernel digest + embedding
          - companion_state row with default values (mig 152 defaults)
          - companion_scene row with default values (mig 157 defaults)
          - companion_identities_genesis row — the immutable seed
            snapshot used by reset gestures (anchor doc §7)

        When no seed row exists (truly fresh install, no Becca yet),
        creates a bare row with display_name=companion_id.title() and
        an empty kernel digest. The runtime's auto-refresh on start
        will populate the digest from the on-disk personality doc.
        """
        if not user_id or user_id in self._provisioned_users:
            return False

        backend = self.backend
        try:
            # Fast-path check: does the identity row already exist?
            cur = await backend.conn.execute(
                "SELECT 1 FROM companion_identities "
                "WHERE user_id = ? AND companion_id = ?",
                (user_id, self.companion_id),
            )
            already = await cur.fetchone() is not None
            await cur.close()
            if already:
                self._provisioned_users.add(user_id)
                return False

            # Fetch the seed row (prefer user_id='' which is the
            # migration-179 backfilled singleton). Falls back to the
            # earliest-created row for the companion_id if no '' seed.
            cur = await backend.conn.execute(
                "SELECT display_name, persona_kernel_digest, "
                "persona_kernel_embedding, personality_doc_version, "
                "drift_score "
                "FROM companion_identities "
                "WHERE companion_id = ? "
                "ORDER BY (user_id = '') DESC, created_at ASC LIMIT 1",
                (self.companion_id,),
            )
            seed = await cur.fetchone()
            await cur.close()

            if seed is not None:
                display_name = seed[0]
                seed_digest = seed[1] or ""
                seed_emb = seed[2]
                seed_doc_ver = int(seed[3]) if seed[3] is not None else 0
                seed_drift = float(seed[4]) if seed[4] is not None else 0.0
            else:
                display_name = self.companion_id.title()
                seed_digest = ""
                seed_emb = None
                seed_doc_ver = 0
                seed_drift = 0.0

            # Insert the per-user identity row. INSERT OR IGNORE in case
            # of a concurrent provisioner (cheap belt-and-suspenders).
            await backend.conn.execute(
                "INSERT OR IGNORE INTO companion_identities "
                "(user_id, companion_id, display_name, persona_kernel_digest, "
                " persona_kernel_embedding, personality_doc_version, drift_score, "
                " owner_user_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, self.companion_id, display_name, seed_digest,
                 seed_emb, seed_doc_ver, seed_drift, user_id),
            )

            # State row with mig 152 defaults (dormant + passive-dominant
            # + no focus). Idempotent.
            await backend.conn.execute(
                "INSERT OR IGNORE INTO companion_state "
                "(user_id, companion_id) VALUES (?, ?)",
                (user_id, self.companion_id),
            )

            # Scene row with mig 157 defaults (main_room / idle).
            await backend.conn.execute(
                "INSERT OR IGNORE INTO companion_scene "
                "(user_id, companion_id) VALUES (?, ?)",
                (user_id, self.companion_id),
            )

            # Genesis snapshot — the immutable seed for reset gestures.
            # Only writes when we have a real kernel digest; an empty
            # digest is a bare-install marker, not a genesis.
            if seed_digest:
                await backend.conn.execute(
                    "INSERT INTO companion_identities_genesis "
                    "(user_id, companion_id, seed_kernel_digest, "
                    " seed_kernel_embedding, seed_personality_doc_version) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (user_id, self.companion_id, seed_digest, seed_emb, seed_doc_ver),
                )

            await backend.conn.commit()
            self._provisioned_users.add(user_id)

            log.info(
                "companion_lazy_provisioned",
                user_id=user_id,
                companion_id=self.companion_id,
                seeded_from_existing=seed is not None,
                has_kernel=bool(seed_digest),
            )
            return True
        except Exception:
            log.exception(
                "companion_lazy_provision_failed",
                user_id=user_id,
                companion_id=self.companion_id,
            )
            return False

    # ── State-transition fan-out to bus ─────────────────────────────

    def _on_state_transition(self, axis: str, from_v: str, to_v: str, reason: str) -> None:
        """Listener hooked into CompanionState. Republishes as a bus event."""
        # track(): hold a strong ref so the GC can't drop the publish
        # mid-flight, and log if it raises (audit 2026-06-17).
        track(self.bus.publish(PresenceEvent(
            topic="state.transition",
            payload={"from": from_v, "to": to_v, "reason": reason},
            source_companion_id=self.companion_id,
        )), name="bus_state_transition")

    def _on_role_transition(self, axis: str, from_v: str, to_v: str, reason: str) -> None:
        track(self.bus.publish(PresenceEvent(
            topic="role.transition",
            payload={"from": from_v, "to": to_v, "reason": reason},
            source_companion_id=self.companion_id,
        )), name="bus_role_transition")

    def _on_focus_transition(self, axis: str, from_v: str, to_v: str, reason: str) -> None:
        track(self.bus.publish(PresenceEvent(
            topic="focus.transition",
            payload={"from": from_v, "to": to_v, "reason": reason},
            source_companion_id=self.companion_id,
        )), name="bus_focus_transition")

    # ── Affect → bus fan-out ────────────────────────────────────────
    #
    # Affect tags are the runtime's compact "what flavour is this
    # moment" signal — settled / curious / patient / melancholy /
    # alert / ... — written onto each journal entry. The visual
    # surface (becca-presence) needs them so the avatar's expression
    # can follow her interior. We track the last-published tag here
    # so this helper publishes only on CHANGE (every tick's "settled"
    # would otherwise spam the WS).
    _last_affect_tag: str = ""

    async def publish_affect(self, tag: str, *, reason: str = "") -> None:
        """Publish ``affect.changed`` if ``tag`` differs from the
        previously-published value. Skips ``settled`` and empty values
        — those are the equilibrium baseline and emitting them would
        wash out the channel. Best-effort; failures are silent so the
        caller's primary write isn't blocked by bus contention.
        """
        normalized = (tag or "").strip().lower()
        if not normalized or normalized in ("settled", "none", "neutral"):
            return
        if normalized == self._last_affect_tag:
            return
        self._last_affect_tag = normalized
        try:
            await self.bus.publish_topic(
                "affect.changed",
                {"tag": normalized, "reason": reason or "tick"},
                source_companion_id=self.companion_id,
            )
        except Exception:
            log.debug("publish_affect_failed", tag=normalized, exc_info=True)

    def _on_state_for_wake(self, axis: str, from_v: str, to_v: str, reason: str) -> None:
        """When state leaves asleep/dreaming, fire the sleep_wake bridge."""
        if from_v in ("asleep", "dreaming") and to_v not in ("asleep", "dreaming"):
            async def _bridge() -> None:
                try:
                    from augmentum.companion_runtime.behavior import sleep_wake
                    await sleep_wake.on_wake(self, prior_state=from_v)
                except Exception:
                    log.warning("sleep_wake_bridge_failed", exc_info=True)
            track(_bridge(), name="sleep_wake_bridge")

    # ── Public API ───────────────────────────────────────────────────

    async def submit_intent(self, intent: Intent) -> Response:
        """Submit an intent for dispatch.

        Heuristic-primary + LLM-tie-breaker routing. The winning
        subagent's :meth:`invoke` is called and its result returned.
        Bus emits ``dispatch.decided``, ``dispatch.tiebreaker``,
        ``subagent.invoked``, ``subagent.completed``.

        Flag-gated by ``companion_dispatch_enabled``. When off, this
        returns a tombstone Response and the caller falls back to the
        legacy mode-router. This is by design — Sprint 3 keeps the
        legacy path reachable for fast rollback.
        """
        import uuid

        from augmentum.config import settings
        if not getattr(settings, "companion_dispatch_enabled", False):
            return Response(
                content="",
                handled_by="runtime.disabled",
                metadata={"reason": "companion_dispatch_enabled is off"},
            )

        # Lazy import to keep dispatch out of the cold-import path
        # for code that doesn't use the runtime.
        from augmentum.companion_runtime import dispatch as _dispatch
        from augmentum.companion_runtime.subagents.base import SubagentContext

        invocation_id = uuid.uuid4().hex[:12]
        user_mode_hint = intent.explicit_mode or intent.metadata.get("user_mode_hint", "")

        try:
            decision = await _dispatch.decide(
                intent, runtime=self, user_mode_hint=user_mode_hint,
            )
        except Exception as exc:
            log.exception("dispatch_decide_failed", error=str(exc))
            return Response(
                content="", handled_by="runtime.error",
                metadata={"error": str(exc)[:200], "phase": "decide"},
            )

        await self.bus.publish_topic(
            "dispatch.decided",
            {
                "winner": decision.winner.name if decision.winner else None,
                "used_tiebreaker": decision.used_tiebreaker,
                "decision_ms": round(decision.decision_ms, 1),
                "abstained": decision.abstained,
                "ranked": [
                    {"name": c.name, "utility": round(c.utility, 3)}
                    for c in decision.ranked[:3]
                ],
                "invocation_id": invocation_id,
            },
            source_companion_id=self.companion_id,
        )

        if decision.abstained or decision.winner is None:
            return Response(
                content="", handled_by="runtime.abstained",
                metadata={
                    "reason": "no candidates available "
                              "(subagent registry inactive or empty)",
                    "decision_ms": round(decision.decision_ms, 1),
                },
            )

        if decision.used_tiebreaker:
            await self.bus.publish_topic(
                "dispatch.tiebreaker",
                {
                    "winner": decision.winner.name,
                    "rationale": decision.tiebreaker_rationale[:300],
                    "invocation_id": invocation_id,
                },
                source_companion_id=self.companion_id,
            )

        ctx = SubagentContext(
            intent=intent,
            runtime=self,
            bus=self.bus,
            companion_id=self.companion_id,
            invocation_id=invocation_id,
        )

        try:
            result = await decision.winner.invoke(ctx)
        except Exception as exc:
            log.exception(
                "subagent_invoke_crashed",
                subagent=decision.winner.name, error=str(exc),
            )
            return Response(
                content="", handled_by=decision.winner.name,
                metadata={
                    "error": str(exc)[:200],
                    "phase": "invoke",
                    "invocation_id": invocation_id,
                    "decision_ms": round(decision.decision_ms, 1),
                },
            )

        if not result.error:
            _dispatch.mark_invocation_success(decision.winner.name)

        # Sprint 4b — record the outcome in the skill archive. Signal
        # is derived: +1.0 clean completion, -1.0 errored. Sprint 5+
        # will refine with corrective-utterance detection.
        try:
            from augmentum.companion_runtime import skill_archive
            outcome = -1.0 if result.error else 1.0
            await skill_archive.record_outcome(
                self, intent,
                chosen_subagent=decision.winner.name,
                outcome_signal=outcome,
                outcome_reason=(result.error or "clean_completion")[:200],
                decision_ms=decision.decision_ms,
                used_tiebreaker=decision.used_tiebreaker,
            )
        except Exception:
            log.warning("skill_archive_record_failed", exc_info=True)

        return Response(
            content=result.content,
            handled_by=result.handled_by or decision.winner.name,
            metadata={
                **(result.metadata or {}),
                "invocation_id": invocation_id,
                "decision_ms": round(decision.decision_ms, 1),
                "used_tiebreaker": decision.used_tiebreaker,
                "error": result.error,
            },
        )

    async def subscribe(self, topic_glob: str = "**", *, slice_key: str = ""):
        """Pass-through to the bus. Returns a Subscription."""
        return await self.bus.subscribe(topic_glob, slice_key=slice_key)

    async def publish(self, event: PresenceEvent) -> None:
        """Pass-through to the bus."""
        await self.bus.publish(event)

    # ── Telemetry ────────────────────────────────────────────────────

    async def snapshot(self) -> dict:
        """Read-only health snapshot for telemetry / debug surfaces.

        Used by ``/api/companion/snapshot``. Cheap — no LLM calls,
        only in-memory reads + a small DB count.
        """
        from augmentum.config import settings
        flags = {
            "companion_runtime_enabled": getattr(settings, "companion_runtime_enabled", False),
            "companion_dispatch_enabled": getattr(settings, "companion_dispatch_enabled", False),
            "companion_tick_enabled": getattr(settings, "companion_tick_enabled", False),
            "companion_journal_enabled": getattr(settings, "companion_journal_enabled", True),
            "companion_creations_enabled": getattr(settings, "companion_creations_enabled", False),
            "companion_xr_orchestrator": getattr(settings, "companion_xr_orchestrator", False),
        }
        memory_counts: dict[str, int] = {}
        if self.memory._store is not None:
            try:
                memory_counts = await self.memory.counts(
                    user_id=self.owner_user_id or "",
                )
            except Exception as exc:
                memory_counts = {"_error": str(exc)[:200]}
        # Adapter registries — counts visible even when the flag is off.
        try:
            from augmentum.companion_runtime.primitives.registry import (
                PrimitiveRegistry,
            )
            from augmentum.companion_runtime.subagents.registry import (
                SubagentRegistry,
            )
            registries = {
                "subagent_names": list(SubagentRegistry.names()),
                "primitive_names": list(PrimitiveRegistry.names()),
                "subagents_active": len(SubagentRegistry.available()),
                "primitives_active": len(PrimitiveRegistry.available()),
            }
        except Exception as exc:
            registries = {"_error": str(exc)[:200]}
        tick_snap = (
            self._tick_loop.snapshot()
            if getattr(self, "_tick_loop", None) is not None
            else {"running": False}
        )
        verb_dispatcher = getattr(self, "_verb_dispatcher", None)
        verb_snap = (
            verb_dispatcher.snapshot()
            if verb_dispatcher is not None
            else {"running": False, "verbs_registered": 0}
        )
        return {
            "companion_id": self.companion_id,
            "started": self._started,
            "flags": flags,
            "identity": self.identity.snapshot(),
            "state": self.state.snapshot(),
            "bus": self.bus.snapshot(),
            "memory_counts": memory_counts,
            "registries": registries,
            "tick": tick_snap,
            "verb_dispatcher": verb_snap,
        }


# ── Factory for the FastAPI lifespan ─────────────────────────────────

async def create_runtime(
    *,
    backend: SQLiteBackend,
    memory_store: MemoryStore,
    core_profile: CoreProfileManager,
    companion_id: str = "becca",
    app_state: Any = None,
) -> CompanionRuntime:
    """Build + start a CompanionRuntime with all wiring in place.

    Used from ``proxy/server.py``'s lifespan when
    ``companion_runtime_enabled`` is True. Returns the started runtime;
    caller pins it to ``app.state.companion_runtime`` and calls
    ``stop()`` at shutdown.
    """
    runtime = CompanionRuntime(
        backend,
        companion_id=companion_id,
        app_state=app_state,
    )
    await runtime.start(memory_store=memory_store, core_profile=core_profile)
    return runtime


__all__ = [
    "CompanionRuntime",
    "Intent",
    "Response",
    "create_runtime",
]
