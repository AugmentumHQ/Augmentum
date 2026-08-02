"""SSE endpoint for background task notifications."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["notifications"])


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""


@router.post("/api/notifications/ingest")
async def notification_ingest(request: Request):
    """Receive webhook events from connected services.

    Unauthenticated (webhook clients don't have Augmentum sessions) —
    validated by a per-service token in ``managed_services.config_json``.
    Accepts JSON::

        {
          "service_id": "uptime-kuma",
          "token": "<webhook_token>",
          "title": "blog.example.com is DOWN",
          "body": "HTTP 500 at 14:32 UTC",
          "source": "uptime-kuma",
          "dedupe_key": "monitor-abc-20260719-1432"
        }

    Returns 200 on publish, 401 on bad token, 404 on unknown service.
    """
    import json as _json

    body = {}
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    service_id = str(body.get("service_id") or "").strip()
    token = str(body.get("token") or "").strip()
    if not service_id or not token:
        return JSONResponse({"error": "service_id and token are required"}, status_code=400)

    # Validate token against managed_services.config_json.
    conn = _resolve_db_conn(request)
    if conn is None:
        return JSONResponse({"error": "database unavailable"}, status_code=503)

    try:
        import sqlite3
        cur = await conn.execute(
            "SELECT config_json FROM managed_services WHERE id = ? AND enabled = 1",
            (service_id,),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        return JSONResponse({"error": "database error"}, status_code=500)

    if row is None or not row[0]:
        return JSONResponse({"error": "unknown service"}, status_code=404)

    try:
        cfg = _json.loads(row[0]) if isinstance(row[0], str) else (row[0] or {})
    except (_json.JSONDecodeError, TypeError):
        return JSONResponse({"error": "invalid config"}, status_code=500)

    expected_token = cfg.get("webhook_token", "")
    if not expected_token or token != expected_token:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    # Map the webhook payload to a notification.
    title = str(body.get("title") or "Service alert").strip()[:200]
    notify_body = str(body.get("body") or "")[:1000]
    source = str(body.get("source") or service_id)[:80]
    dedupe_key = str(body.get("dedupe_key") or "")[:120]
    importance_raw = body.get("importance")

    from augmentum.notifications.catalog import IMPORTANCE_HIGH
    importance = IMPORTANCE_HIGH
    if isinstance(importance_raw, int) and 0 <= importance_raw <= 4:
        importance = importance_raw

    # Resolve a target user for the notification. Service alerts are
    # install-wide — we deliver to the first admin user.
    try:
        admin_cur = await conn.execute(
            "SELECT id FROM users WHERE is_admin = 1 LIMIT 1",
        )
        admin_row = await admin_cur.fetchone()
        await admin_cur.close()
        target_user = admin_row[0] if admin_row else ""
    except Exception:
        target_user = ""

    if not target_user:
        log.warning("notification_ingest_no_admin", service_id=service_id)
        return JSONResponse({"status": "no_target_user"}, status_code=404)

    # Publish via the module-level publish (takes conn directly).
    try:
        from augmentum.notifications.store import publish as publish_notif

        notification_id = await publish_notif(
            conn,
            user_id=target_user,
            channel_id="service.alert",
            source=source,
            title=title,
            body=notify_body,
            dedupe_key=dedupe_key,
            importance=importance,
        )
        log.info(
            "notification_ingest_published",
            service_id=service_id,
            notification_id=notification_id,
            title=title[:80],
        )
        return JSONResponse({
            "status": "published",
            "notification_id": notification_id,
        })
    except Exception:
        log.warning(
            "notification_ingest_publish_failed",
            service_id=service_id, exc_info=True,
        )
        return JSONResponse({"error": "publish failed"}, status_code=500)


def _resolve_db_conn(request: Request):
    """Best-effort aiosqlite connection from app.state."""
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm is not None else None
    return getattr(backend, "conn", None)


@router.get("/api/notifications/{session_id}")
async def notification_stream(request: Request, session_id: str):
    """SSE stream for background task notifications.

    The client subscribes once per session. Events are pushed when
    background flows complete or fail.
    """
    manager = getattr(request.app.state, "background_chain_manager", None)
    if not manager:
        return JSONResponse({"error": "Background chains not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    queue = manager.subscribe(session_id, user_id=uid)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield ":\n\n"  # SSE keepalive comment
                    continue
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            manager.unsubscribe(session_id, queue, user_id=uid)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/api/background-tasks/{session_id}")
async def list_background_tasks(request: Request, session_id: str):
    """List background tasks for a session (pull endpoint)."""
    manager = getattr(request.app.state, "background_chain_manager", None)
    if not manager:
        return JSONResponse({"tasks": []})

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    tasks = manager.get_tasks(session_id, user_id=uid)
    return JSONResponse({
        "tasks": [
            {
                "task_id": t.task_id,
                "flow_name": t.flow_name,
                "status": t.status,
                "query": t.query,
                "result_summary": t.result_summary[:500] if t.result_summary else "",
                "error": t.error,
                "injected": t.injected,
            }
            for t in tasks
        ],
    })


@router.get("/api/background-tasks/{session_id}/{task_id}")
async def get_background_task(request: Request, session_id: str, task_id: str):
    """Get details for a specific background task."""
    manager = getattr(request.app.state, "background_chain_manager", None)
    if not manager:
        return JSONResponse({"error": "Not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    task = manager.get_task(task_id, user_id=uid)
    if not task or task.session_id != session_id:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse({
        "task_id": task.task_id,
        "flow_name": task.flow_name,
        "flow_id": task.flow_id,
        "status": task.status,
        "query": task.query,
        "result_summary": task.result_summary,
        "error": task.error,
        "injected": task.injected,
    })
