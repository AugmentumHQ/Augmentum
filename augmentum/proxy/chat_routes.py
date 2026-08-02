"""Chat session CRUD — server-side storage for UI chat sessions.

Stores the full session object (including message tree, metadata, narrative
fields) as a JSON blob.  This is intentionally simple — the client's tree
structure is preserved as-is rather than decomposed into relational tables.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect

from augmentum.proxy import system_events
from augmentum.state.write_guard import incoming_stamp, stored_stamps
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/chats", tags=["chats"])


def _count_messages(session: dict) -> int:
    """Count message nodes in a session's tree."""
    tree = session.get("tree")
    if isinstance(tree, dict):
        return len(tree)
    return 0


def _session_updated_at(session: dict) -> int:
    """Client-stamped edit time (ms epoch) from the session blob.

    This is the ONLY usable staleness signal: the ``updated_at`` COLUMN
    records when a client last *synced*, so a stale tab syncing late gets
    a newer column value than the fresh tab that synced first — ordering
    by sync time would bless exactly the clobber we're guarding against.
    The blob's ``updatedAt`` is stamped at *edit* time by the client that
    made the change. Returns 0 when absent/invalid (legacy blobs).
    """
    return incoming_stamp(session)


async def _stored_updated_at(conn, ids: list[str], uid: str) -> dict[str, int]:
    """Map session id -> client edit-stamp (ms) of the STORED blob.

    Thin wrapper kept for the two call sites below; the implementation now
    lives in ``state/write_guard.py`` so every other user-content table can
    use the same guard instead of chats being the only protected surface.
    """
    return await stored_stamps(conn, "ui_sessions", ids, user_id=uid)


def _extract_group_id(session: dict) -> str | None:
    """Pull groupId out of the session body, normalized to None when absent.

    Mirrored into the ui_sessions.group_id column so the meta-listing path
    can skip reading the full data blob.
    """
    gid = session.get("groupId") if isinstance(session, dict) else None
    if isinstance(gid, str) and gid:
        return gid
    return None


def _backend(request: Request):
    from augmentum.state.backends.sqlite import SQLiteBackend

    sm = getattr(request.app.state, "state_manager", None)
    if not sm or not isinstance(sm.backend, SQLiteBackend):
        return None
    return sm.backend


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request.

    The auth middleware attaches a User object to ``scope["user"]``.
    Returns empty string when auth is disabled, which disables scoping
    for backward compatibility.
    """
    user = request.scope.get("user")
    return user.id if user else ""


# ── List (lightweight — no message trees) ─────────────────────────────────

@router.get("/")
async def list_chats(request: Request):
    """Return all sessions — full data by default, metadata-only with ?meta=1."""
    be = _backend(request)
    if not be:
        return JSONResponse({"sessions": {}})

    meta_only = request.query_params.get("meta") == "1"
    uid = _user_id(request)

    if meta_only:
        # Meta path skips the `data` blob entirely — group_id was
        # promoted to its own column in migration 174 precisely so this
        # listing doesn't drag hundreds of KB per row across the worker.
        q = ("SELECT id, title, mode, group_id, created_at, updated_at, message_count "
             "FROM ui_sessions")
        params: list = []
        if uid:
            q += " WHERE user_id = ?"
            params.append(uid)
        q += " ORDER BY updated_at DESC"
        cursor = await be.conn.execute(q, params)
    else:
        q = ("SELECT id, title, mode, data, created_at, updated_at "
             "FROM ui_sessions")
        params = []
        if uid:
            q += " WHERE user_id = ?"
            params.append(uid)
        q += " ORDER BY updated_at DESC"
        cursor = await be.conn.execute(q, params)
    rows = await cursor.fetchall()
    sessions = {}
    for r in rows:
        if meta_only:
            meta: dict = {
                "id": r[0],
                "title": r[1],
                "mode": r[2],
                "createdAt": r[4] or "",
                "updatedAt": r[5] or "",
                "version": 2,
                "messageCount": r[6] or 0,
            }
            gid = r[3]
            if gid:
                meta["groupId"] = gid
            sessions[r[0]] = meta
        else:
            data = json.loads(r[3])
            data["id"] = r[0]
            data["title"] = r[1]
            data["mode"] = r[2]
            sessions[r[0]] = data
    return JSONResponse({"sessions": sessions})


# ── Get single (full session with tree) ───────────────────────────────────

@router.get("/{session_id}")
async def get_chat(session_id: str, request: Request):
    be = _backend(request)
    if not be:
        return JSONResponse({"error": "No database"}, status_code=503)

    uid = _user_id(request)
    query = "SELECT id, title, mode, data FROM ui_sessions WHERE id = ?"
    params: list = [session_id]
    if uid:
        query += " AND user_id = ?"
        params.append(uid)
    cursor = await be.conn.execute(query, params)
    row = await cursor.fetchone()
    if not row:
        return JSONResponse({"error": "Not found"}, status_code=404)

    raw = row[3] or "{}"
    data = json.loads(raw)
    data["id"] = row[0]
    data["title"] = row[1]
    data["mode"] = row[2]

    # Instrumentation for Task 3.1 (incremental tree hydration). Logs blob
    # size + tree node count so we can decide whether lazy branch loading
    # is worth the complexity. No user-visible effect. Drop this once we
    # have enough data to make the call.
    try:
        tree = data.get("tree") or {}
        node_count = len(tree) if isinstance(tree, dict) else 0
        if node_count >= 100:
            log.info(
                "ui_session_large_tree_load",
                session_id=session_id,
                node_count=node_count,
                blob_bytes=len(raw),
            )
    except Exception as exc:
        log.debug("ui_session_large_tree_instrumentation_failed", session_id=session_id, error=str(exc))

    return JSONResponse(data)


# ── Save / update (upsert) ───────────────────────────────────────────────

@router.put("/{session_id}")
async def save_chat(session_id: str, request: Request):
    be = _backend(request)
    if not be:
        return JSONResponse({"error": "No database"}, status_code=503)

    try:
        body = await request.json()
    except ClientDisconnect:
        log.info("chat_save_client_disconnected", session_id=session_id)
        return JSONResponse({"ok": False, "error": "client disconnected"}, status_code=499)
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    title = body.get("title", "New Chat")
    mode = body.get("mode", "passthrough")
    msg_count = _count_messages(body)
    now = datetime.now(UTC).isoformat()
    data_json = json.dumps(body)
    group_id = _extract_group_id(body)
    uid = _user_id(request)

    # Stale-write guard — same contract as the bulk /sync path: never let
    # an older client edit replace a newer stored tree (multi-tab clobber).
    incoming_ts = _session_updated_at(body)
    if incoming_ts:
        stored_ts = await _stored_updated_at(be.conn, [session_id], uid)
        if stored_ts.get(session_id, 0) > incoming_ts:
            log.warning("chat_save_stale_rejected", session_id=session_id)
            return JSONResponse(
                {"ok": False, "stale": True, "id": session_id}, status_code=409,
            )

    # ON CONFLICT clause carries a user_id guard so a known session_id
    # from another tenant can't be overwritten by passing the same id.
    # Rows with NULL user_id (legacy pre-auth data) remain claimable.
    if uid:
        await be.conn.execute(
            "INSERT INTO ui_sessions (id, title, mode, data, message_count, created_at, updated_at, user_id, group_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET title=?, mode=?, data=?, message_count=?, updated_at=?, group_id=? "
            "WHERE ui_sessions.user_id = ? OR ui_sessions.user_id IS NULL",
            (session_id, title, mode, data_json, msg_count, now, now, uid, group_id,
             title, mode, data_json, msg_count, now, group_id,
             uid),
        )
    else:
        await be.conn.execute(
            "INSERT INTO ui_sessions (id, title, mode, data, message_count, created_at, updated_at, group_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET title=?, mode=?, data=?, message_count=?, updated_at=?, group_id=? "
            "WHERE ui_sessions.user_id IS NULL",
            (session_id, title, mode, data_json, msg_count, now, now, group_id,
             title, mode, data_json, msg_count, now, group_id),
        )
    await be.conn.commit()
    return JSONResponse({"ok": True, "id": session_id})


# ── Delete ────────────────────────────────────────────────────────────────

@router.delete("/{session_id}")
async def delete_chat(session_id: str, request: Request):
    be = _backend(request)
    if not be:
        return JSONResponse({"error": "No database"}, status_code=503)

    uid = _user_id(request)
    query = "DELETE FROM ui_sessions WHERE id = ?"
    params: list = [session_id]
    if uid:
        query += " AND user_id = ?"
        params.append(uid)
    cursor = await be.conn.execute(query, params)
    if cursor.rowcount == 0:
        await be.conn.commit()
        return JSONResponse({"error": "Not found"}, status_code=404)

    # Phase 4: dual cleanup path.
    #
    # 1. Narrative tiers route through narrative_cleanup.purge_narrative_session
    #    (single source of truth, vec-aware, transactional, observable).
    # 2. Non-narrative user-scoped tables stay in this inline frozenset —
    #    different lifecycle owners; the audit script enforces coverage of
    #    BOTH paths against migration analysis.
    # 3. DELETE FROM sessions fires FK CASCADE for any future tables that
    #    declare ON DELETE CASCADE (the new tables in migrations 115-118 do).
    if uid:
        try:
            from augmentum.state.narrative_cleanup import purge_narrative_session
            cleanup_report = await purge_narrative_session(
                be.conn, session_id, user_id=uid,
            )
            log.info("chat_deleted", **cleanup_report.to_event_kwargs())
        except Exception:
            log.warning("narrative_cleanup_failed",
                        session_id=session_id, exc_info=True)

    # Trigger any FK CASCADE rules by removing the canonical sessions row.
    # No-op for sessions that never had narrative data (sessions row may not
    # exist; DELETE without WHERE-match is harmless).
    try:
        if uid:
            await be.conn.execute(
                "DELETE FROM sessions WHERE id = ? AND user_id = ?",
                [session_id, uid],
            )
        else:
            await be.conn.execute(
                "DELETE FROM sessions WHERE id = ?", [session_id],
            )
    except Exception:
        log.warning("chat_delete_sessions_row_failed",
                    session_id=session_id, exc_info=True)

    # Non-narrative user-scoped tables — explicit cleanup.
    # SECURITY: table names are from a hardcoded tuple — no user input.
    _SESSION_CLEANUP_TABLES = frozenset({
        "session_messages",
        "facts", "entities", "plot_threads", "contradictions",
        "lorebook_entries", "assumptions", "character_cards",
        "character_relationships",
    })
    for table in _SESSION_CLEANUP_TABLES:
        try:
            cleanup_q = f"DELETE FROM {table} WHERE session_id = ?"  # noqa: S608 — table from hardcoded frozenset
            cleanup_params: list = [session_id]
            if uid:
                cleanup_q += " AND user_id = ?"
                cleanup_params.append(uid)
            await be.conn.execute(cleanup_q, cleanup_params)
        except Exception as exc:
            log.debug("delete_chat_table_cleanup_skipped", table=table, error=str(exc))

    await be.conn.commit()

    # Clean up any background chain tasks for this session
    bg = getattr(request.app.state, "background_chain_manager", None)
    if bg:
        bg.cleanup_session(session_id, user_id=uid)

    # Evict cached narrative engine/handler for this session
    # Cache key is (user_id, session_id) when auth is active, else plain session_id
    engines = getattr(request.app.state, "narrative_engines", None)
    if engines:
        cache_key = (uid, session_id) if uid else session_id
        if cache_key in engines:
            del engines[cache_key]
    handlers = getattr(request.app.state, "narrative_handlers", None)
    if handlers:
        cache_key = (uid, session_id) if uid else session_id
        if cache_key in handlers:
            del handlers[cache_key]

    # Tell this user's other devices the session list changed (user-scoped).
    system_events.publish("sessions.changed", {"id": session_id, "deleted": True}, user_id=uid)
    return JSONResponse({"ok": True})


# ── Bulk import (localStorage migration) ─────────────────────────────────

@router.post("/import")
async def import_chats(request: Request):
    """Accept a sessions object {id: session, ...} and upsert them all."""
    be = _backend(request)
    if not be:
        return JSONResponse({"error": "No database"}, status_code=503)

    try:
        body = await request.json()
    except ClientDisconnect:
        log.info("chat_sync_client_disconnected")
        return JSONResponse({"ok": False, "error": "client disconnected"}, status_code=499)
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    sessions = body.get("sessions", {})
    if not isinstance(sessions, dict):
        return JSONResponse({"error": "Expected {sessions: {id: ...}}"}, status_code=400)

    now = datetime.now(UTC).isoformat()
    uid = _user_id(request)

    # Stale-write guard: a session whose STORED blob carries a newer client
    # edit-stamp than the incoming one was written by another tab/device
    # since this client last loaded it. Upserting would replace the whole
    # tree and silently erase that device's turns (multi-tab clobber class).
    # Reject those ids and report them in the response as "stale" — the
    # client union-merges the server copy and re-syncs, so neither side's
    # turns are lost. Incoming blobs with no stamp (legacy) are accepted.
    stored_ts = await _stored_updated_at(be.conn, list(sessions.keys()), uid)
    stale: list[str] = []

    # Batch all upserts into one executemany. Each request from the auto-
    # save path can carry every active session in the user's tab, so a
    # 20-session sync was 20 sequential round-trips through aiosqlite's
    # worker thread (blocking other coroutines for that whole time).
    upsert_rows: list[tuple] = []
    for sid, session in sessions.items():
        incoming_ts = _session_updated_at(session) if isinstance(session, dict) else 0
        if incoming_ts and stored_ts.get(sid, 0) > incoming_ts:
            stale.append(sid)
            continue
        title = session.get("title", "New Chat")
        mode = session.get("mode", "passthrough")
        msg_count = _count_messages(session)
        data_json = json.dumps(session)
        group_id = _extract_group_id(session)
        if uid:
            upsert_rows.append((
                sid, title, mode, data_json, msg_count, now, now, uid, group_id,
                title, mode, data_json, msg_count, now, group_id,
                uid,
            ))
        else:
            upsert_rows.append((
                sid, title, mode, data_json, msg_count, now, now, group_id,
                title, mode, data_json, msg_count, now, group_id,
            ))

    count = len(upsert_rows)
    if upsert_rows:
        if uid:
            await be.conn.executemany(
                "INSERT INTO ui_sessions (id, title, mode, data, message_count, created_at, updated_at, user_id, group_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET title=?, mode=?, data=?, message_count=?, updated_at=?, group_id=? "
                "WHERE ui_sessions.user_id = ? OR ui_sessions.user_id IS NULL",
                upsert_rows,
            )
        else:
            await be.conn.executemany(
                "INSERT INTO ui_sessions (id, title, mode, data, message_count, created_at, updated_at, group_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET title=?, mode=?, data=?, message_count=?, updated_at=?, group_id=? "
                "WHERE ui_sessions.user_id IS NULL",
                upsert_rows,
            )

    # Process deletions if provided. Single statement with IN (...) instead
    # of one DELETE per id — fewer round-trips, plus we can derive the
    # delete count from rowcount in one call.
    deleted = [sid for sid in body.get("deleted", []) if isinstance(sid, str)]
    deleted_count = 0
    if deleted:
        placeholders = ",".join("?" * len(deleted))
        if uid:
            cursor = await be.conn.execute(
                f"DELETE FROM ui_sessions WHERE user_id = ? AND id IN ({placeholders})",
                [uid, *deleted],
            )
        else:
            cursor = await be.conn.execute(
                f"DELETE FROM ui_sessions WHERE id IN ({placeholders})",
                deleted,
            )
        deleted_count = cursor.rowcount or 0

    await be.conn.commit()
    # Signal the user's other devices to reconcile their sidebar. ``count`` is
    # an upsert count (the autosave path sends all active sessions), so this
    # fires on most syncs, not just membership changes — the CLIENT handler
    # debounces hard and reconciles non-destructively, so the emit rate is
    # deliberately moot. Skip only genuinely empty syncs.
    if count or deleted_count:
        system_events.publish("sessions.changed", {"imported": count, "deleted": deleted_count}, user_id=uid)
    if stale:
        log.warning("chats_sync_stale_rejected", ids=stale[:10], count=len(stale))
    log.info("chats_imported", count=count, deleted=deleted_count)
    return JSONResponse({
        "ok": True, "imported": count, "deleted": deleted_count, "stale": stale,
    })


# ── Bulk save (batch persist) ─────────────────────────────────────────────

@router.post("/sync")
async def sync_chats(request: Request):
    """Save multiple sessions at once (used by periodic auto-save)."""
    return await import_chats(request)


# ── Multi-model fan-out plan ──────────────────────────────────────────────

@router.post("/fanout-plan")
async def fanout_plan(request: Request):
    """Group compare models by backend so the UI knows what can run in parallel.

    The composer's multi-model fan-out sends one request per model. Models
    served by a single-slot local engine (the bundled llama-server, or an
    external llama.cpp server) cannot generate two different models
    concurrently — a second request would force a hot swap mid-stream (or
    bounce off a pin). The UI serializes models that share an ``exclusive``
    backend and runs everything else in parallel. With the secondary engine
    slot, a pinned model resolves to its own backend key and therefore
    fans out truly concurrently with the primary.
    """
    try:
        body = await request.json()
    except (ClientDisconnect, ValueError):
        body = {}
    models: list[str] = []
    for m in body.get("models") or []:
        if isinstance(m, str) and m.strip() and m.strip() not in models:
            models.append(m.strip())

    registry = getattr(request.app.state, "provider_registry", None)
    plan: list[dict] = []
    for name in models:
        entry: dict = {"model": name, "backend": "", "exclusive": False}
        if registry is None:
            plan.append(entry)
            continue
        try:
            backend, _clean = await registry.resolve_backend_for_model(name)
            for key, be_obj in getattr(registry, "_backends", {}).items():
                if be_obj is backend:
                    entry["backend"] = key
                    break
            # Single-slot local engines: one loaded model per process.
            # Class check covers the bundled engine + external llama.cpp
            # (and the secondary slot, whichever of the two it wraps);
            # the key-prefix check is a forward-compat net for new
            # engine-family backends registered under engine* keys.
            entry["exclusive"] = (
                type(backend).__name__ in ("AugmentumEngineBackend", "LlamaCppBackend")
                or entry["backend"].startswith("engine")
                or entry["backend"] == "llamacpp"
            )
        except Exception as exc:
            # Unresolvable model still gets a plan entry — the UI shows the
            # error on that card instead of silently dropping the model.
            log.warning("fanout_plan_resolve_failed", model=name, error=str(exc))
            entry["error"] = str(exc)
        plan.append(entry)
    return JSONResponse({"plan": plan})
