"""Dream System API routes."""
from __future__ import annotations

import dataclasses

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import structlog

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/dream", tags=["dream"])


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _get_journal(request: Request):
    return getattr(request.app.state, "dream_journal", None)


def _get_portrait_manager(request: Request):
    return getattr(request.app.state, "dream_portrait_manager", None)


def _get_scheduler(request: Request):
    return getattr(request.app.state, "dream_scheduler", None)


def _disabled() -> JSONResponse:
    return JSONResponse({"error": "Dream system not enabled"}, status_code=503)


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request (empty string if no auth)."""
    user = request.scope.get("user")
    return user.id if user else ""


def _entry_to_dict(entry, include_context: bool = False) -> dict:
    """Convert a DreamEntry dataclass to a JSON-serialisable dict."""
    d = {
        "id": entry.id,
        "persona_id": entry.persona_id,
        "content": entry.content,
        "entry_type": entry.entry_type.value if hasattr(entry.entry_type, "value") else entry.entry_type,
        "source_memories": entry.source_memories,
        "source_sessions": entry.source_sessions,
        "weight": entry.weight,
        "pinned": entry.pinned,
        "dream_cycle_id": entry.dream_cycle_id,
        "created_at": entry.created_at,
        "expires_at": entry.expires_at,
    }
    if include_context:
        d["context_window"] = entry.context_window
    return d


def _portrait_to_dict(portrait) -> dict:
    """Convert a DreamPortrait dataclass to a JSON-serialisable dict."""
    return {
        "id": portrait.id,
        "persona_id": portrait.persona_id,
        "voice_notes": portrait.voice_notes,
        "active_threads": portrait.active_threads,
        "impressions": portrait.impressions,
        "source_entries": portrait.source_entries,
        "is_current": portrait.is_current,
        "checkpoint_name": portrait.checkpoint_name,
        "created_at": portrait.created_at,
    }


# ---------------------------------------------------------------------------
# Pydantic request bodies
# ---------------------------------------------------------------------------


class UpdateEntryRequest(BaseModel):
    content: str | None = None
    weight: float | None = None
    pinned: bool | None = None


class TriggerRequest(BaseModel):
    persona_id: str = "default"


class RegeneratePortraitRequest(BaseModel):
    persona_id: str = "default"


class CheckpointRequest(BaseModel):
    persona_id: str = "default"
    name: str


class RestoreCheckpointRequest(BaseModel):
    persona_id: str = "default"


class ResetPortraitRequest(BaseModel):
    persona_id: str = "default"


# ---------------------------------------------------------------------------
# Journal endpoints
# ---------------------------------------------------------------------------


@router.get("/journal")
async def list_journal_entries(
    request: Request,
    persona_id: str = "default",
    limit: int = 50,
    offset: int = 0,
    type: str | None = None,
) -> JSONResponse:
    """List dream journal entries with optional type filter and pagination."""
    journal = _get_journal(request)
    if journal is None:
        return _disabled()

    try:
        entries, total = await journal.list_entries(
            persona_id=persona_id,
            limit=limit,
            offset=offset,
            entry_type=type,
            user_id=_user_id(request),
        )
    except Exception:
        log.warning("dream_journal_list_failed", exc_info=True)
        return JSONResponse({"error": "Failed to list journal entries"}, status_code=500)

    return JSONResponse({
        "entries": [_entry_to_dict(e) for e in entries],
        "total": total,
        "has_more": offset + len(entries) < total,
    })


@router.get("/journal/{entry_id}")
async def get_journal_entry(entry_id: str, request: Request) -> JSONResponse:
    """Get a single dream journal entry including its context window."""
    journal = _get_journal(request)
    if journal is None:
        return _disabled()

    try:
        entry = await journal.get_entry(entry_id, user_id=_user_id(request))
    except Exception:
        log.warning("dream_journal_get_failed", entry_id=entry_id, exc_info=True)
        return JSONResponse({"error": "Failed to fetch entry"}, status_code=500)

    if entry is None:
        return JSONResponse({"error": "Entry not found"}, status_code=404)

    return JSONResponse(_entry_to_dict(entry, include_context=True))


@router.put("/journal/{entry_id}")
async def update_journal_entry(
    entry_id: str,
    body: UpdateEntryRequest,
    request: Request,
) -> JSONResponse:
    """Update a dream journal entry. Regenerates portrait if content changed."""
    journal = _get_journal(request)
    if journal is None:
        return _disabled()
    user_id = _user_id(request)

    # Verify the entry exists and belongs to this user
    try:
        existing = await journal.get_entry(entry_id, user_id=user_id)
    except Exception:
        log.warning("dream_journal_update_fetch_failed", entry_id=entry_id, exc_info=True)
        return JSONResponse({"error": "Failed to fetch entry"}, status_code=500)

    if existing is None:
        return JSONResponse({"error": "Entry not found"}, status_code=404)

    try:
        await journal.update_entry(
            entry_id=entry_id,
            content=body.content,
            weight=body.weight,
            pinned=body.pinned,
            user_id=user_id,
        )
    except Exception:
        log.warning("dream_journal_update_failed", entry_id=entry_id, exc_info=True)
        return JSONResponse({"error": "Failed to update entry"}, status_code=500)

    # Regenerate portrait if content changed (scoped to user)
    if body.content is not None:
        portrait_mgr = _get_portrait_manager(request)
        if portrait_mgr is not None:
            try:
                await portrait_mgr.synthesize(existing.persona_id, "", user_id=user_id)
            except Exception:
                log.warning("dream_portrait_regen_failed", entry_id=entry_id, exc_info=True)

    return JSONResponse({"id": entry_id, "status": "updated"})


@router.delete("/journal/{entry_id}")
async def delete_journal_entry(entry_id: str, request: Request) -> JSONResponse:
    """Delete a dream journal entry and trigger portrait regeneration."""
    journal = _get_journal(request)
    if journal is None:
        return _disabled()
    user_id = _user_id(request)

    # Get persona_id before deleting so we can trigger portrait regen
    try:
        existing = await journal.get_entry(entry_id, user_id=user_id)
    except Exception:
        existing = None

    if existing is None:
        return JSONResponse({"error": "Entry not found"}, status_code=404)

    try:
        await journal.delete_entry(entry_id, user_id=user_id)
    except Exception:
        log.warning("dream_journal_delete_failed", entry_id=entry_id, exc_info=True)
        return JSONResponse({"error": "Failed to delete entry"}, status_code=500)

    # Regenerate portrait after deletion (scoped)
    portrait_mgr = _get_portrait_manager(request)
    if portrait_mgr is not None:
        try:
            await portrait_mgr.synthesize(existing.persona_id, "", user_id=user_id)
        except Exception:
            log.warning("dream_portrait_regen_after_delete_failed", entry_id=entry_id, exc_info=True)

    return JSONResponse({"id": entry_id, "status": "deleted"})


# ---------------------------------------------------------------------------
# Trigger endpoint
# ---------------------------------------------------------------------------


@router.post("/compact")
async def compact_dreams(request: Request) -> JSONResponse:
    """Manually trigger a dream compaction pass for the calling user.

    Admin-only — compaction runs LLM calls and modifies journal data,
    so we require admin auth even though the operation is scoped to the
    caller's own user_id (no cross-user surface). Useful for verifying
    settings changes / debugging cluster behavior without waiting for
    the next scheduled cycle.

    Returns the stats dict from ``DreamCompactor.compact()``.
    """
    from augmentum.auth.guards import require_admin

    if (forbidden := require_admin(request)) is not None:
        return forbidden

    compactor = getattr(request.app.state, "dream_compactor", None)
    if compactor is None:
        return JSONResponse(
            {"error": "Dream compactor not running"},
            status_code=503,
        )

    uid = _user_id(request)
    if not uid:
        return JSONResponse(
            {"error": "Compaction requires an authenticated user"},
            status_code=401,
        )

    try:
        stats = await compactor.compact(user_id=uid)
    except Exception:
        log.warning("dream_compact_trigger_failed", user_id=uid, exc_info=True)
        return JSONResponse({"error": "Compaction failed"}, status_code=500)

    return JSONResponse({"status": "complete", "stats": stats})


@router.post("/trigger")
async def trigger_dream(body: TriggerRequest, request: Request) -> JSONResponse:
    """Manually trigger a dream cycle (scoped to caller).

    Distinguishes two refusal cases so the UI can react appropriately:

    * 503 — process-level dream subsystem not running (no tenant has
      opted in). The caller is told "dream system not enabled".
    * 409 — subsystem is running for other tenants, but this caller
      hasn't set ``ui.dreamEnabled = true``. The caller is told to
      enable dreams first.
    """
    from augmentum.dream.scheduler import DreamsDisabledError

    scheduler = _get_scheduler(request)
    if scheduler is None:
        return _disabled()

    try:
        cycle_id = await scheduler.trigger_manual(
            persona_id=body.persona_id, user_id=_user_id(request),
        )
    except DreamsDisabledError:
        return JSONResponse(
            {"error": "Enable the dream system in Settings before triggering a cycle"},
            status_code=409,
        )
    except Exception:
        log.warning("dream_trigger_failed", exc_info=True)
        return JSONResponse({"error": "Failed to trigger dream cycle"}, status_code=500)

    return JSONResponse({"cycle_id": cycle_id, "status": "queued"})


# ---------------------------------------------------------------------------
# Portrait endpoints
# ---------------------------------------------------------------------------


@router.get("/portrait")
async def get_portrait(
    request: Request,
    persona_id: str = "default",
) -> JSONResponse:
    """Get the current dream portrait for a persona (scoped to caller)."""
    portrait_mgr = _get_portrait_manager(request)
    if portrait_mgr is None:
        return _disabled()

    try:
        portrait = await portrait_mgr.get_current(persona_id, user_id=_user_id(request))
    except Exception:
        log.warning("dream_portrait_get_failed", persona_id=persona_id, exc_info=True)
        return JSONResponse({"error": "Failed to fetch portrait"}, status_code=500)

    if portrait is None:
        return JSONResponse(None)

    return JSONResponse(_portrait_to_dict(portrait))


@router.post("/portrait/regenerate")
async def regenerate_portrait(
    body: RegeneratePortraitRequest,
    request: Request,
) -> JSONResponse:
    """Regenerate the dream portrait from the caller's journal entries."""
    portrait_mgr = _get_portrait_manager(request)
    if portrait_mgr is None:
        return _disabled()

    try:
        portrait = await portrait_mgr.synthesize(body.persona_id, "", user_id=_user_id(request))
    except Exception:
        log.warning("dream_portrait_regenerate_failed", persona_id=body.persona_id, exc_info=True)
        return JSONResponse({"error": "Failed to regenerate portrait"}, status_code=500)

    if portrait is None:
        return JSONResponse({"error": "No journal entries to synthesize from"}, status_code=404)

    return JSONResponse(_portrait_to_dict(portrait))


@router.post("/portrait/checkpoint")
async def save_portrait_checkpoint(
    body: CheckpointRequest,
    request: Request,
) -> JSONResponse:
    """Save the current portrait as a named checkpoint."""
    portrait_mgr = _get_portrait_manager(request)
    if portrait_mgr is None:
        return _disabled()

    try:
        checkpoint_id = await portrait_mgr.save_checkpoint(
            body.persona_id, body.name, user_id=_user_id(request),
        )
    except Exception:
        log.warning("dream_checkpoint_save_failed", persona_id=body.persona_id, exc_info=True)
        return JSONResponse({"error": "Failed to save checkpoint"}, status_code=500)

    if checkpoint_id is None:
        return JSONResponse({"error": "No current portrait to checkpoint"}, status_code=404)

    return JSONResponse({"checkpoint_id": checkpoint_id, "name": body.name})


@router.get("/portrait/checkpoints")
async def list_portrait_checkpoints(
    request: Request,
    persona_id: str = "default",
) -> JSONResponse:
    """List saved portrait checkpoints for a persona (scoped to caller)."""
    portrait_mgr = _get_portrait_manager(request)
    if portrait_mgr is None:
        return _disabled()

    try:
        checkpoints = await portrait_mgr.list_checkpoints(persona_id, user_id=_user_id(request))
    except Exception:
        log.warning("dream_checkpoint_list_failed", persona_id=persona_id, exc_info=True)
        return JSONResponse({"error": "Failed to list checkpoints"}, status_code=500)

    return JSONResponse({"checkpoints": [_portrait_to_dict(c) for c in checkpoints]})


@router.post("/portrait/restore/{checkpoint_id}")
async def restore_portrait_checkpoint(
    checkpoint_id: str,
    request: Request,
    persona_id: str = "default",
) -> JSONResponse:
    """Restore a previously saved portrait checkpoint as the current portrait."""
    portrait_mgr = _get_portrait_manager(request)
    if portrait_mgr is None:
        return _disabled()

    try:
        portrait = await portrait_mgr.restore_checkpoint(
            persona_id, checkpoint_id, user_id=_user_id(request),
        )
    except Exception:
        log.warning("dream_checkpoint_restore_failed", checkpoint_id=checkpoint_id, exc_info=True)
        return JSONResponse({"error": "Failed to restore checkpoint"}, status_code=500)

    if portrait is None:
        return JSONResponse({"error": "Checkpoint not found"}, status_code=404)

    return JSONResponse(_portrait_to_dict(portrait))


@router.post("/portrait/reset")
async def reset_portrait(
    body: ResetPortraitRequest,
    request: Request,
) -> JSONResponse:
    """Reset all dream data for a persona back to foundation state (scoped)."""
    portrait_mgr = _get_portrait_manager(request)
    if portrait_mgr is None:
        return _disabled()

    try:
        await portrait_mgr.reset_to_foundation(body.persona_id, user_id=_user_id(request))
    except Exception:
        log.warning("dream_portrait_reset_failed", persona_id=body.persona_id, exc_info=True)
        return JSONResponse({"error": "Failed to reset portrait"}, status_code=500)

    return JSONResponse({"success": True, "persona_id": body.persona_id})


# ---------------------------------------------------------------------------
# Cycles endpoints
# ---------------------------------------------------------------------------


@router.get("/cycles")
async def list_dream_cycles(
    request: Request,
    persona_id: str = "default",
    limit: int = 20,
) -> JSONResponse:
    """List recent dream cycles for a persona."""
    journal = _get_journal(request)
    if journal is None:
        return _disabled()

    # Query directly from the SQLite DB via the journal's db_path
    db_path = getattr(journal, "_db_path", None)
    if db_path is None:
        return JSONResponse({"error": "Journal DB path not available"}, status_code=500)

    user_id = _user_id(request)
    try:
        import aiosqlite

        from augmentum.state.backends.sqlite import apply_augmentum_pragmas

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            # Apply the canonical pragma set so this ad-hoc reader
            # honors the same busy_timeout as every other writer on
            # augmentum.db. A read-only path doesn't take the writer
            # lock, but it CAN sit behind a slow writer; pragmas keep
            # behavior consistent across all connections.
            await apply_augmentum_pragmas(db)
            query = (
                "SELECT id, persona_id, trigger_reason, memories_count, entries_count,"
                " model_used, tokens_used, duration_ms, status, error,"
                " started_at, completed_at"
                " FROM dream_cycles WHERE persona_id = ?"
            )
            params: list = [persona_id]
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            query += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
    except Exception:
        log.warning("dream_cycles_list_failed", persona_id=persona_id, exc_info=True)
        return JSONResponse({"error": "Failed to list dream cycles"}, status_code=500)

    cycles = [
        {
            "id": r["id"],
            "persona_id": r["persona_id"],
            "trigger_reason": r["trigger_reason"],
            "memories_count": r["memories_count"],
            "entries_count": r["entries_count"],
            "model_used": r["model_used"],
            "tokens_used": r["tokens_used"],
            "duration_ms": r["duration_ms"],
            "status": r["status"],
            "error": r["error"],
            "started_at": r["started_at"],
            "completed_at": r["completed_at"],
        }
        for r in rows
    ]
    return JSONResponse({"cycles": cycles})


@router.get("/cycles/{cycle_id}")
async def get_dream_cycle(cycle_id: str, request: Request) -> JSONResponse:
    """Get a single dream cycle, including its associated entry IDs."""
    journal = _get_journal(request)
    if journal is None:
        return _disabled()

    db_path = getattr(journal, "_db_path", None)
    if db_path is None:
        return JSONResponse({"error": "Journal DB path not available"}, status_code=500)

    user_id = _user_id(request)
    try:
        import aiosqlite

        from augmentum.state.backends.sqlite import apply_augmentum_pragmas

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            # Apply the canonical pragma set so this ad-hoc reader
            # honors the same busy_timeout as every other writer on
            # augmentum.db. A read-only path doesn't take the writer
            # lock, but it CAN sit behind a slow writer; pragmas keep
            # behavior consistent across all connections.
            await apply_augmentum_pragmas(db)

            # Fetch the cycle (scoped to caller)
            cycle_query = (
                "SELECT id, persona_id, trigger_reason, memories_count, entries_count,"
                " model_used, tokens_used, duration_ms, status, error,"
                " started_at, completed_at FROM dream_cycles WHERE id = ?"
            )
            cycle_params: list = [cycle_id]
            if user_id:
                cycle_query += " AND user_id = ?"
                cycle_params.append(user_id)
            async with db.execute(cycle_query, cycle_params) as cursor:
                row = await cursor.fetchone()

            if row is None:
                return JSONResponse({"error": "Cycle not found"}, status_code=404)

            # Fetch associated entry IDs (scoped — extra safety even though
            # cycle_id is already user-scoped above)
            entry_query = (
                "SELECT id FROM dream_entries WHERE dream_cycle_id = ?"
            )
            entry_params: list = [cycle_id]
            if user_id:
                entry_query += " AND user_id = ?"
                entry_params.append(user_id)
            entry_query += " ORDER BY created_at ASC"
            async with db.execute(entry_query, entry_params) as cursor:
                entry_rows = await cursor.fetchall()

    except Exception:
        log.warning("dream_cycle_get_failed", cycle_id=cycle_id, exc_info=True)
        return JSONResponse({"error": "Failed to fetch dream cycle"}, status_code=500)

    return JSONResponse({
        "id": row["id"],
        "persona_id": row["persona_id"],
        "trigger_reason": row["trigger_reason"],
        "memories_count": row["memories_count"],
        "entries_count": row["entries_count"],
        "model_used": row["model_used"],
        "tokens_used": row["tokens_used"],
        "duration_ms": row["duration_ms"],
        "status": row["status"],
        "error": row["error"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "entry_ids": [r["id"] for r in entry_rows],
    })


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------


@router.get("/status")
async def get_dream_status(request: Request) -> JSONResponse:
    """Return current dream scheduler status for the caller."""
    scheduler = _get_scheduler(request)
    if scheduler is None:
        return _disabled()

    try:
        status = await scheduler.get_status(user_id=_user_id(request))
    except Exception:
        log.warning("dream_status_failed", exc_info=True)
        return JSONResponse({"error": "Failed to fetch scheduler status"}, status_code=500)

    return JSONResponse(status)
