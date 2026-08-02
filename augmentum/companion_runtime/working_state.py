"""Companion working-set persistence — continuity across restarts.

The ReferentCache holds her working state (active note, trail, last
dispatch) in memory only: every container restart — and on voice,
every reconnect's fresh session id — lobotomized "the note" and
"take me there". This module write-throughs that small working set
to the EXISTING per-user settings store (``user_settings`` via
``SettingsStore.set_user``) and lazily rehydrates fresh caches from
it. No new table, no migration, no new store — one JSON blob per
user under ``companion.working_state``.

Deliberately per-USER, not per-session: the companion is one
continuous relationship; her working context follows the person
across surfaces and reconnects. Screen-state (the attention store)
stays ephemeral by design — what's on a screen she can no longer see
is not continuity, it's confabulation.

Both functions soft-fail: continuity is best-effort, never blocking.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from augmentum.utils.bg_tasks import track
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_KEY = "companion.working_state"
_TRAIL_CAP = 20
_RING_CAP = 4


def _store(app_state: Any):
    return getattr(app_state, "settings_store", None) if app_state else None


async def hydrate_working_state(
    app_state: Any, user_id: str, refs: Any,
) -> None:
    """Fill an unhydrated ReferentCache from the persisted working set.

    Idempotent per cache instance (``_ws_hydrated`` marker). Only fills
    fields that are still empty — live in-memory state always wins over
    the snapshot.
    """
    if refs is None or not user_id:
        return
    if getattr(refs, "_ws_hydrated", False):
        return
    refs._ws_hydrated = True  # noqa: SLF001 — set before I/O so a crash can't loop
    store = _store(app_state)
    if store is None:
        return
    try:
        raw = await store.get_user(user_id, _KEY)
        if not raw:
            return
        data = json.loads(raw)
        if not isinstance(data, dict):
            return
        if not getattr(refs, "active_note_id", None) and data.get("active_note_id"):
            refs.active_note_id = str(data["active_note_id"])
            refs.active_note_title = str(data.get("active_note_title") or "")
        trail = getattr(refs, "trail", None)
        if trail is not None and not trail and isinstance(data.get("trail"), list):
            trail.extend(data["trail"][-_TRAIL_CAP:])
        ring = getattr(refs, "results_ring", None)
        if ring is not None and not ring and isinstance(data.get("results_ring"), list):
            ring.extend(data["results_ring"][-_RING_CAP:])
            # The decay clock must survive with its entries or every
            # rehydrated digest would look freshly-born.
            refs.turn_seq = max(
                int(data.get("turn_seq") or 0), getattr(refs, "turn_seq", 0),
            )
        if not getattr(refs, "last_dispatch_action", None):
            refs.last_dispatch_action = str(data.get("last_dispatch_action") or "")
            refs.last_dispatch_summary = str(data.get("last_dispatch_summary") or "")
        log.debug("working_state_hydrated", user_id=user_id)
    except Exception:  # noqa: BLE001 — continuity is best-effort
        log.debug("working_state_hydrate_failed", exc_info=True)


async def save_working_state(
    app_state: Any, user_id: str, refs: Any,
) -> None:
    """Write-through the working set after a mutation."""
    if refs is None or not user_id:
        return
    store = _store(app_state)
    if store is None:
        return
    try:
        payload = json.dumps({
            "active_note_id": getattr(refs, "active_note_id", "") or "",
            "active_note_title": getattr(refs, "active_note_title", "") or "",
            "trail": list(getattr(refs, "trail", None) or [])[-_TRAIL_CAP:],
            "results_ring": list(
                getattr(refs, "results_ring", None) or [],
            )[-_RING_CAP:],
            "turn_seq": int(getattr(refs, "turn_seq", 0) or 0),
            "last_dispatch_action": getattr(refs, "last_dispatch_action", "") or "",
            "last_dispatch_summary": (
                getattr(refs, "last_dispatch_summary", "") or ""
            )[:200],
        })
        await store.set_user(user_id, _KEY, payload)
    except Exception:  # noqa: BLE001
        log.debug("working_state_save_failed", exc_info=True)


def schedule_save(app_state: Any, user_id: str, refs: Any) -> None:
    """Fire-and-forget save for sync call sites (trail appends)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    # track(): GC-safe ref + the save's failures get logged instead of
    # silently vanishing (audit 2026-06-17).
    track(save_working_state(app_state, user_id, refs), name="working_state_save")
