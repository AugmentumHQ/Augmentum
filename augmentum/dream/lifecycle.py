"""Dream system lifecycle — boot/teardown helpers.

Extracted from the server lifespan so the dream system can be started or
stopped dynamically when ``ui.dreamEnabled`` flips at runtime, without
forcing a process restart.

Design contract
---------------
* All four singletons live on ``app.state``:
  ``dream_journal``, ``dream_portrait_manager``, ``dream_engine``,
  ``dream_scheduler``.
* If setup fails partway through, the partial state is torn down so the
  attribute is either fully populated or ``None`` — callers (route
  handlers) gate on these to return 503 cleanly.
* ``setup_dream_system`` and ``teardown_dream_system`` are idempotent.
* The functions read configuration from the settings store on every
  boot, so user changes to thresholds/idle/cooldown take effect on the
  next call.

Multi-tenancy
-------------
The dream subsystem (journal, portrait manager, engine, scheduler) is a
process-level singleton with user-scoped *data*: every DB row carries a
``user_id``, every scheduler counter keys by ``user_id``. The boot
predicate ``should_dream_run`` asks "does any tenant want this?" and
authoritatively decides whether to keep the singleton alive. It evaluates
true when the install-wide default is on, OR when at least one user has
opted in via the personalization UI.

Per-user cycle gating lives in :class:`DreamScheduler` — even when the
singleton is booted process-wide, a user whose ``ui.dreamEnabled`` is not
``"true"`` never has a cycle run on their behalf.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

    from augmentum.state.settings_store import SettingsStore

log = get_logger(__name__)


async def should_dream_run(store: SettingsStore | None) -> bool:
    """Does *anyone* currently want the dream subsystem alive?

    True if the install-wide default is on (legacy / single-tenant), or
    if at least one tenant has ``ui.dreamEnabled = "true"`` in
    ``user_settings``. The two checks run in the cheap-first order: the
    global row is PK-indexed; the user scan is a partial index lookup.
    """
    if store is None:
        return False
    if (await store.get("ui.dreamEnabled")) == "true":
        return True
    return await store.has_any_user_value("ui.dreamEnabled", "true")


async def setup_dream_system(app: FastAPI) -> bool:
    """Boot the dream system. Returns True on success, False otherwise.

    Safe to call when already running — returns True without re-creating.
    Requires ``app.state.memory_store`` to exist (dreams need approved
    memories to dream from).
    """
    # Idempotent: if scheduler already up, nothing to do.
    if getattr(app.state, "dream_scheduler", None) is not None:
        return True

    settings_store = getattr(app.state, "settings_store", None)
    memory_store = getattr(app.state, "memory_store", None)
    if memory_store is None:
        log.warning("dream_setup_skipped", reason="memory_store unavailable")
        return False

    # Lazy imports keep cold-start fast and avoid import cycles.
    from augmentum.config import settings
    from augmentum.dream.engine import DreamEngine
    from augmentum.dream.journal import DreamJournal
    from augmentum.dream.portrait import PortraitManager
    from augmentum.dream.scheduler import DreamScheduler

    try:
        # Resolve the SQLite db_path from the existing memory store so all
        # four dream tables sit alongside memories in the same DB. Public
        # ``db_path`` property is the canonical accessor; legacy attribute
        # names kept as fallbacks for older MemoryStore implementations.
        db_path = (
            getattr(memory_store, "db_path", None)
            or getattr(memory_store, "_db_path", None)
        )
        if not db_path:
            log.warning("dream_setup_skipped", reason="db_path unavailable")
            return False

        journal = DreamJournal(db_path)
        await journal.initialize()
        # Wire app_state so the journal can resolve the provider registry
        # for on-write consolidation. Done after initialize() so the
        # journal is fully ready before being introspected by writers.
        journal.attach_app_state(app.state)
        app.state.dream_journal = journal

        portrait_mgr = PortraitManager(
            journal, settings_store,
            provider_registry=getattr(app.state, "provider_registry", None),
        )
        app.state.dream_portrait_manager = portrait_mgr

        engine = DreamEngine(
            journal=journal,
            memory_store=memory_store,
            state_manager=getattr(app.state, "state_manager", None),
            embedding_service=getattr(app.state, "embedding_service", None),
            portrait_manager=portrait_mgr,
            settings=settings,
            provider_registry=getattr(app.state, "provider_registry", None),
            settings_store=settings_store,
        )
        app.state.dream_engine = engine

        # Read live thresholds from settings store (latest user choices).
        threshold = int(await settings_store.get("ui.dreamMessageThreshold") or "6") if settings_store else 6
        idle_min = int(await settings_store.get("ui.dreamIdleMinutes") or "30") if settings_store else 30
        cooldown = int(await settings_store.get("ui.dreamCooldownMinutes") or "60") if settings_store else 60

        scheduler = DreamScheduler(
            engine=engine,
            settings_store=settings_store,
            enabled=True,
            message_threshold=threshold,
            idle_minutes=idle_min,
            cooldown_minutes=cooldown,
        )
        await scheduler.initialize()  # restore per-user counters
        scheduler.start()
        app.state.dream_scheduler = scheduler
        log.info(
            "dream_system_started",
            threshold=threshold, idle=idle_min, cooldown=cooldown,
        )

        # Fire-and-forget backfill of any pre-existing entries that don't
        # have embeddings yet (rows from before semantic recall was wired).
        # Runs in the background so startup isn't blocked by ~2ms-per-entry
        # FastEmbed work; pre-existing chronological recall keeps working
        # in the meantime. Bounded per-call (max_entries=500) so a long-
        # time user doesn't soak the embedding model on a single startup —
        # the next restart picks up where this one left off.
        if journal._vec_enabled:
            async def _kick_backfill() -> None:
                # Brief delay so the embedding model only loads after the
                # webserver is responsive. Without this, the very first
                # request can race the model warm-up and pay ~2s latency.
                import asyncio as _asyncio
                await _asyncio.sleep(15)
                try:
                    await journal.backfill_embeddings("default")
                except Exception:
                    log.warning("dream_journal.backfill_kickoff_failed", exc_info=True)
            import asyncio
            asyncio.create_task(_kick_backfill())

        # Background dream compactor — semantic dedup + cluster summary.
        # Independent of the dream scheduler; uses its own per-user loop
        # mirroring MemoryCompactor's pattern. Gated by the global
        # admin-controlled setting; disabled installs skip wiring entirely.
        app.state.dream_compactor = None
        if settings.dream_compaction_enabled:
            from augmentum.dream.compactor import DreamCompactor
            compactor = DreamCompactor(
                journal=journal,
                registry=getattr(app.state, "provider_registry", None),
                settings_store=settings_store,
                app_state=app.state,
            )
            compactor.start()
            app.state.dream_compactor = compactor
            log.info(
                "dream_compactor_initialized",
                interval_hours=settings.dream_compaction_interval_hours,
            )
        return True

    except Exception:
        log.error("dream_system_init_failed", exc_info=True)
        await teardown_dream_system(app)
        return False


async def teardown_dream_system(app: FastAPI) -> None:
    """Stop the scheduler and close the journal connection. Idempotent."""
    compactor = getattr(app.state, "dream_compactor", None)
    if compactor is not None:
        try:
            await compactor.stop()
        except Exception:
            log.warning("dream_compactor_stop_failed", exc_info=True)

    scheduler = getattr(app.state, "dream_scheduler", None)
    if scheduler is not None:
        try:
            await scheduler.stop()
        except Exception:
            log.warning("dream_scheduler_stop_failed", exc_info=True)

    journal = getattr(app.state, "dream_journal", None)
    if journal is not None:
        try:
            await journal.close()
        except Exception:
            log.warning("dream_journal_close_failed", exc_info=True)

    app.state.dream_scheduler = None
    app.state.dream_engine = None
    app.state.dream_portrait_manager = None
    app.state.dream_journal = None
    app.state.dream_compactor = None
    log.info("dream_system_stopped")
