"""Cross-device sync API — reading positions + browse-history ingest.

Server side of the Augmentum Android client's ``SyncRepository``. Two
surfaces, both keyed off the authenticated user:

  - **reading positions** — last-write-wins per ``(user, key)``. The phone
    pushes book/article positions and pulls what other devices recorded, so a
    book started on the phone resumes on the desktop and vice-versa. Backed by
    ``ReadingPositionStore`` (``app.state.sync_store``).
  - **browse history** — phone reader / web-reader surfaces feed the same
    ``browse_history`` substrate the desktop browser-proxy already populates,
    so the discovery + companion cross-modal layers see phone reading too.
    Reuses ``DiscoveryStore.upsert_history`` (``app.state.discovery_store``) —
    it already owns the multi-tenant, UNIQUE(url)-aware upsert.

All four endpoints fail closed (401) without an authenticated user — these
tables are user-scoped and must never accept writes into the anon row.
"""
from __future__ import annotations

import time
from urllib.parse import urlparse

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/sync", tags=["sync"])

# Cap a single push so a misbehaving client can't hand us an unbounded batch.
_MAX_BATCH = 500


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Reading positions
# ---------------------------------------------------------------------------


@router.post("/reading-positions")
async def push_reading_positions(request: Request) -> JSONResponse:
    """Upsert a batch of reading positions for the authenticated user.

    Request:  ``{"device_id": "...", "positions": [ReadingPosition, ...]}``
    Response: ``{"accepted": N, "rejected": M, "conflicts": [keys]}``
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "authentication required"}, status_code=401)
    store = getattr(request.app.state, "sync_store", None)
    if store is None:
        return JSONResponse({"error": "sync store unavailable"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    positions = body.get("positions") if isinstance(body, dict) else None
    if not isinstance(positions, list):
        return JSONResponse({"error": "positions must be a list"}, status_code=400)
    if len(positions) > _MAX_BATCH:
        positions = positions[:_MAX_BATCH]

    try:
        accepted, rejected, conflicts = await store.upsert_positions(
            positions, user_id=uid,
        )
    except Exception:
        log.warning("reading_positions_push_failed", user_id=uid, exc_info=True)
        return JSONResponse({"error": "upsert failed"}, status_code=500)

    return JSONResponse(
        {"accepted": accepted, "rejected": rejected, "conflicts": conflicts}
    )


@router.get("/reading-positions")
async def pull_reading_positions(request: Request) -> JSONResponse:
    """Return reading positions updated since ``since_ms`` (server clock).

    Query:    ``since_ms`` (epoch ms cursor), ``device_id`` (caller's id;
              its own rows are excluded so it never re-pulls its own writes).
    Response: ``{"positions": [ReadingPosition, ...], "now_ms": <server ms>}``
    The ``now_ms`` echo is the client's next cursor — phone-clock-skew immune.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "authentication required"}, status_code=401)
    store = getattr(request.app.state, "sync_store", None)
    if store is None:
        return JSONResponse({"error": "sync store unavailable"}, status_code=503)

    since_ms = _safe_int(request.query_params.get("since_ms", "0"))
    device_id = request.query_params.get("device_id", "") or ""

    try:
        positions = await store.list_since(
            user_id=uid, since_ms=since_ms, exclude_device_id=device_id,
        )
    except Exception:
        log.warning("reading_positions_pull_failed", user_id=uid, exc_info=True)
        return JSONResponse({"error": "query failed"}, status_code=500)

    return JSONResponse({"positions": positions, "now_ms": _now_ms()})


# ---------------------------------------------------------------------------
# Browse history
# ---------------------------------------------------------------------------


@router.post("/browse-history")
async def push_browse_history(request: Request) -> JSONResponse:
    """Ingest phone browse/reader events into the ``browse_history`` table.

    Request:  ``{"device_id": "...", "events": [BrowseEvent, ...]}`` where each
              event is ``{url, title, opened_ms, duration_ms, content_type,
              device_id}``.
    Response: ``{"accepted": N, "skipped": M}``
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "authentication required"}, status_code=401)
    store = getattr(request.app.state, "discovery_store", None)
    if store is None:
        return JSONResponse({"error": "discovery store unavailable"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    events = body.get("events") if isinstance(body, dict) else None
    if not isinstance(events, list):
        return JSONResponse({"error": "events must be a list"}, status_code=400)
    if len(events) > _MAX_BATCH:
        events = events[:_MAX_BATCH]

    accepted = 0
    skipped = 0
    for ev in events:
        if not isinstance(ev, dict):
            skipped += 1
            continue
        url = str(ev.get("url") or "").strip()
        if not url:
            skipped += 1
            continue
        title = str(ev.get("title") or "")
        content_type = str(ev.get("content_type") or "article")
        hostname = urlparse(url).hostname or ""
        domain = hostname.lower().removeprefix("www.")
        metadata = {
            "source": "phone",
            "device_id": str(ev.get("device_id") or ""),
            "opened_ms": _safe_int(ev.get("opened_ms")),
            "duration_ms": _safe_int(ev.get("duration_ms")),
        }
        try:
            result = await store.upsert_history(
                url=url,
                title=title,
                domain=domain,
                content_type=content_type,
                thumbnail="",
                metadata=metadata,
                user_id=uid,
            )
        except Exception:
            log.warning(
                "browse_history_ingest_failed", user_id=uid, exc_info=True,
            )
            skipped += 1
            continue
        # upsert_history returns collision=True when the legacy UNIQUE(url)
        # index is held by another tenant — count those as skipped, not stored.
        if isinstance(result, dict) and result.get("collision"):
            skipped += 1
        else:
            accepted += 1

    return JSONResponse({"accepted": accepted, "skipped": skipped})
