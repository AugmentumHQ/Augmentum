"""Perception acquisition routes — the device→server uplink for L0 streams.

The on-device listeners (notification access first) read local data and POST
normalized batches here, onto the user's OWN server. This is the sovereignty
uplink: the data never touches a third party; it lands user-scoped and the
perception fusers correlate it into insights.

  * ``POST /api/perception/notifications`` — batch of normalized notifications
    from the Android ``NotificationListenerService``.

User-scoped: user_id from the request scope; the anon sentinel is rejected.
Each stream is gated by its ``companion_perception_acquire_*`` setting — when
OFF we 200 with ``{stored: 0, disabled: true}`` so the phone's fire-and-forget
uploader doesn't error-loop (same contract as architect/load_context).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/perception", tags=["perception"])

# Defensive: a single upload shouldn't carry more than this many items. The
# store also caps; this keeps a giant body from being parsed at all.
_MAX_ITEMS = 200


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


@router.post("/notifications")
async def ingest_notifications(request: Request) -> JSONResponse:
    """Accept a batch of normalized notifications from the device.

    Body:
      {"notifications": [
         {"source_pkg": "com.whatsapp", "source_app": "WhatsApp",
          "category": "msg", "title": "Jordan", "body": "you around?",
          "person": "Jordan", "is_message": true,
          "posted_at": 1750000000.0, "notif_key": "0|com.whatsapp|123|..."},
         ...
      ]}

    Returns ``{stored: N}`` (new rows after dedup). Always 200 unless the caller
    is unauthenticated or the body is unparseable — the uploader is best-effort.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    from augmentum.config import settings
    if not getattr(settings, "companion_perception_acquire_notifications", False):
        # Gated off — accept-and-drop so the phone stops nothing and retries
        # nothing. It can re-enable server-side without an app change.
        return JSONResponse({"stored": 0, "disabled": True}, status_code=200)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    items = body.get("notifications") if isinstance(body, dict) else None
    if not isinstance(items, list):
        return JSONResponse(
            {"error": "Missing 'notifications' array"}, status_code=400,
        )

    from augmentum.companion_runtime.perception.acquisition import (
        NotificationObservation,
        record_notifications,
    )

    observations = []
    for raw in items[:_MAX_ITEMS]:
        obs = NotificationObservation.from_wire(raw)
        if obs is not None:
            observations.append(obs)

    backend = getattr(request.app.state, "sqlite_backend", None)
    if backend is None or getattr(backend, "conn", None) is None:
        log.warning("perception_ingest_no_backend")
        return JSONResponse({"stored": 0, "error": "no backend"}, status_code=200)

    try:
        stored = await record_notifications(
            backend, user_id=uid, observations=observations,
        )
    except Exception as exc:  # noqa: BLE001 — log and degrade, never 500 the phone
        log.warning("perception_ingest_failed", error=str(exc)[:200])
        return JSONResponse({"stored": 0, "error": "store failed"}, status_code=200)

    log.info(
        "perception_notifications_ingested",
        user_id=uid, received=len(items), stored=stored,
    )
    return JSONResponse({"stored": stored, "received": len(items)}, status_code=200)
