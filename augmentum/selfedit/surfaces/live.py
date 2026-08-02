"""Live wiring for the surface-reshape system — the single place that binds the
pure layer to Augmentum's real subsystems.

Keeping this in one module means the contract/engine stay pure and testable, and
the server's startup needs exactly one call (``register_default_surfaces``) plus,
per request, one recorder bound to a DB connection. NOTHING here is forced into
the contested server.py by this session — it's the documented wire-up point a
later one-line change activates.

Two bindings:
  - the **config / Adaptation** surface → the real per-user ``SettingsStore``
    (``get_user`` / ``set_user`` map straight onto the adapter's read/write);
  - the **archive recorder** → ``selfedit.store`` (the never-pruned lineage), so
    a surface reshape lands in the SAME archive as a code self-edit (Law 0).
"""

from __future__ import annotations

from typing import Any

from augmentum.selfedit import store
from augmentum.selfedit.surfaces.base import ReshapeChange, SurfaceAdapter, register_surface
from augmentum.selfedit.surfaces.config_surface import build_config_surface
from augmentum.selfedit.surfaces.engine import (
    STATUS_APPLIED_PENDING,
    STATUS_FAILED,
    STATUS_PROMOTED,
    STATUS_REVERTED,
    ReshapeRequest,
)
from augmentum.selfedit.surfaces.reshape import ReshapeResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def register_default_surfaces(settings_store: Any, *,
                              revert_ledger: dict | None = None) -> list[SurfaceAdapter]:
    """Build + register the surfaces that have a real actuator+oracle TODAY.
    Call once at startup. ``settings_store`` is the per-user ``SettingsStore``
    (needs ``get_user``/``set_user``). Returns the registered adapters."""
    config = build_config_surface(
        read=settings_store.get_user, write=settings_store.set_user,
        revert_ledger=revert_ledger)
    register_surface(config)
    log.info("reshape_default_surfaces_registered", surfaces=[config.name])
    return [config]


# Engine status → archive terminal status. APPLIED_PENDING is intentionally NOT
# finalized: the change is applied but awaiting the user's pick, so the attempt
# row stays open until the human verdict finalizes it (promoted / rolled_back).
_ARCHIVE_STATUS = {
    STATUS_PROMOTED: "promoted",
    STATUS_REVERTED: "rolled_back",
    STATUS_FAILED: "failed",
}


def _lesson(status: str, result: ReshapeResult) -> str:
    tier = result.verdict.tier if result.verdict else "?"
    if status == STATUS_PROMOTED:
        return f"surface change auto-applied (mechanically verified, tier={tier})."
    if status == STATUS_REVERTED:
        return f"surface change auto-reverted on verify failure (tier={tier})."
    if status == STATUS_APPLIED_PENDING:
        return f"surface change applied, awaiting human pick (tier={tier})."
    return f"surface change could not apply: {result.detail}"


def build_store_recorder(conn: Any):
    """Return ``(on_start, on_finish)`` archive hooks bound to a DB connection,
    reusing ``selfedit.store``. A surface reshape is recorded in the same
    never-pruned ``self_edit_attempts`` lineage as a code edit."""

    async def on_start(attempt_id: str, request: ReshapeRequest, change: ReshapeChange) -> None:
        try:
            await store.create_attempt(
                conn, attempt_id=attempt_id, user_id=request.actor,
                objective=request.ask, surface=change.surface, tier="green")
        except Exception as exc:  # noqa: BLE001 — recording must never sink a reshape
            log.warning("reshape_record_start_failed", attempt_id=attempt_id, error=repr(exc))

    async def on_finish(attempt_id: str, actor: str, result: ReshapeResult, status: str) -> None:
        lesson = _lesson(status, result)
        if status == STATUS_APPLIED_PENDING:
            log.info("reshape_record_pending", attempt_id=attempt_id)  # left open for the human verdict
            return
        archive_status = _ARCHIVE_STATUS.get(status, "failed")
        try:
            await store.finalize(
                conn, attempt_id=attempt_id, user_id=actor, status=archive_status,
                outcome=result.detail, lesson=lesson)
        except Exception as exc:  # noqa: BLE001
            log.warning("reshape_record_finish_failed", attempt_id=attempt_id, error=repr(exc))

    return on_start, on_finish
