"""Sleep/wake transitions — bridges runtime state to the dream subsystem.

Migration 058 (predates the runtime) shipped the dream tables
``dream_entries``, ``dream_portraits``, ``dream_cycles``, and
``dream_memory_log``, all keyed by ``persona_id``. Migration 151
backfilled ``companion_id='becca'`` and we use that key to scope.

Two entry points:

- :func:`invoke_dream` — fires a dream cycle through the existing
  ``augmentum.dream.lifecycle.setup_dream_system`` pathway. Used by
  :mod:`activity_selector` when ``dream_invocation`` wins a tick.
- :func:`on_wake` — when Becca's state transitions from ``asleep`` /
  ``dreaming`` back to ``dormant`` or ``present``, surfaces the
  dream's distilled insights into ``companion_journal`` so they
  flow into normal memory.

Behind ``companion_dreams_enabled`` (default True) — the existing
dream subsystem already has its own enable path; this layer is
strictly additive.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from augmentum.companion_runtime.scoping import owner_clause_nullable
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


async def invoke_dream(runtime: CompanionRuntime) -> bool:
    """Fire one dream cycle for the current companion.

    Returns True if a cycle was attempted (regardless of outcome),
    False if the dream subsystem isn't available or is disabled.
    """
    from augmentum.config import settings
    if not getattr(settings, "companion_dreams_enabled", True):
        return False

    app_state = getattr(runtime, "_app_state", None)
    dream_engine = getattr(app_state, "dream_engine", None) if app_state else None
    if dream_engine is None:
        log.info("invoke_dream_skipped_no_engine")
        return False

    # User-id resolution: CompanionRuntime is companion-scoped (companion_id)
    # but ``DreamEngine.run_cycle`` filters every step by ``user_id``. The
    # runtime's owner is resolved at start() via
    # CompanionRuntime._resolve_owner_user_id and persisted in
    # companion_identities.owner_user_id (migration 173). Single-user
    # installs auto-bind; multi-user installs must set the
    # ``companion_default_owner_user_id`` setting. When still unresolved
    # (fresh install, no users yet), skip rather than fire into a no-op
    # — the per-user scheduler (``app.state.dream_scheduler``) is the
    # canonical threshold-triggered path and continues to work.
    owner_user_id = str(getattr(runtime, "owner_user_id", "") or "").strip()
    if not owner_user_id:
        log.info(
            "invoke_dream_skipped_unresolved_owner",
            companion_id=runtime.companion_id,
            note=(
                "Companion owner_user_id is unresolved; "
                "set companion_default_owner_user_id or add a user. "
                "Per-user dream_scheduler continues to operate."
            ),
        )
        return False

    await runtime.bus.publish_topic(
        "dream.invoked",
        {"companion_id": runtime.companion_id, "ts": time.time()},
        source_companion_id=runtime.companion_id,
    )
    try:
        run = getattr(dream_engine, "run_cycle", None) or \
              getattr(dream_engine, "tick", None) or \
              getattr(dream_engine, "run", None)
        if run is None:
            log.warning("invoke_dream_no_entry_point",
                        engine_type=type(dream_engine).__name__)
            return False
        result = run(
            persona_id=runtime.companion_id,
            trigger_reason="activity_selector",
            user_id=owner_user_id,
        ) if callable(run) else None
        if hasattr(result, "__await__"):
            await result
    except TypeError:
        # Engine signature varies — fall back to no-arg invocation
        try:
            result = run()
            if hasattr(result, "__await__"):
                await result
        except Exception as exc:
            log.warning("invoke_dream_failed", error=str(exc))
            return False
    except Exception as exc:
        log.exception("invoke_dream_crashed", error=str(exc))
        return False
    return True


async def on_wake(runtime: CompanionRuntime, *, prior_state: str) -> None:
    """Called when the state machine transitions out of asleep/dreaming.

    Reads the most recent dream entry for ``becca`` and writes a brief
    journal entry so the dream's content lands in normal memory.
    """
    if prior_state not in ("asleep", "dreaming"):
        return
    from augmentum.config import settings
    if not getattr(settings, "companion_dreams_enabled", True):
        return
    if not getattr(settings, "companion_journal_enabled", True):
        return

    # SQLiteBackend.connect() is a one-shot lifecycle call (already run
    # at runtime.start()); the persistent connection lives on
    # ``backend.conn`` and that's what every other companion_runtime
    # module uses. The prior ``async with backend.connect() as conn``
    # form raised TypeError on every wake because ``connect`` returns a
    # coroutine, not an async context manager — latent because nothing
    # actually moved state out of asleep until the tick-loop state
    # driver landed.
    # Schema (migrations 058 + 151): dream_entries(id, persona_id,
    # content, ..., created_at, ...) plus companion_id added by mig 151.
    # The prior query referenced `summary`/`ts` which don't exist on
    # this table — a second latent bug under the connect()-was-coroutine
    # crash that nothing surfaced until on_wake ran for real.
    # Owner-scope the dream so one user's wake can't surface another
    # user's dream into their journal (audit 2026-06-17). dream_entries
    # has a nullable user_id (mig 089) so pre-pivot rows stay visible on
    # the unowned box. The companion/persona OR is parenthesized so the
    # owner clause AND-applies to the whole match.
    owner = str(getattr(runtime, "owner_user_id", "") or "")
    frag, p = owner_clause_nullable(owner)
    try:
        cur = await runtime.backend.conn.execute(
            "SELECT id, content, created_at FROM dream_entries "
            f"WHERE (companion_id = ? OR persona_id = ?) {frag} "
            "ORDER BY created_at DESC LIMIT 1",
            (runtime.companion_id, runtime.companion_id, *p),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        log.warning("on_wake_dream_lookup_failed", exc_info=True)
        return
    if not row:
        return
    dream_id, dream_content, _dream_ts = row[0], row[1], row[2]
    if dream_id is None:
        return

    # Dedupe: every state bounce out of asleep/dreaming used to re-write
    # the SAME wake-from-dream journal entry because the dream lookup
    # always returns the most-recent row. Check whether we already
    # journaled this dream id (via related_memory_ids) before writing
    # another identical row. The check is cheap (single SELECT against
    # an indexed column) and bails on error so a degraded lookup
    # gracefully falls through to the old "always write" behavior.
    try:
        cur = await runtime.backend.conn.execute(
            "SELECT 1 FROM companion_journal "
            "WHERE companion_id = ? "
            "  AND entry_type = 'noticing' "
            "  AND affect_tag = 'liminal' "
            "  AND instr(related_memory_ids, ?) > 0 "
            "LIMIT 1",
            (runtime.companion_id, f'"{dream_id}"'),
        )
        already = await cur.fetchone()
        await cur.close()
        if already:
            log.debug(
                "on_wake_dream_already_journaled",
                dream_id=dream_id,
                companion_id=runtime.companion_id,
            )
            return
    except Exception:
        log.warning("on_wake_dedupe_check_failed", exc_info=True)

    try:
        await runtime.memory.journal(
            content=f"woke from a dream — {str(dream_content)[:400]}",
            entry_type="noticing",
            user_id=runtime.owner_user_id or None,
            affect_tag="liminal",
            related_memory_ids=[str(dream_id)],
        )
    except Exception:
        log.warning("on_wake_journal_failed", exc_info=True)


__all__ = ["invoke_dream", "on_wake"]
