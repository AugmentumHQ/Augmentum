"""Calendar surface API — the unified, first-class Augmentum calendar.

One time grid overlays three layers, each tagged so the UI can color and
toggle them:

  * ``augmentum`` — native user-owned events (calendar_user_events). The
    primary object; created/edited in-app, optionally mirrored to a CalDAV
    server so it reaches the user's phone/laptop via open standards.
  * ``calendar``  — appointments pulled FROM connected CalDAV servers,
    cached in calendar_events (migration 314).
  * ``companion`` — occurrences of the companion's standing tasks
    (briefings / watches / scheduled requests), expanded over the window
    via the engine's own scheduling primitives (no duplicated cron logic).

Every route is user-scoped: user_id comes from the authenticated scope and
filters every read/write. Works whether or not the companion runtime is up
— the companion layer is simply omitted when no scheduler exists.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from augmentum.calendar import user_events as ue
from augmentum.proxy.companion_routes import _resolve_user_id, _tasks_ctx
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter()

# Cap the visible window so a bad ?start/?end can't ask us to expand years
# of recurrences into memory.
_MAX_WINDOW_DAYS = 400


# ── Shared helpers ──────────────────────────────────────────────────────


def _conn(request: Request):
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None)
    return getattr(backend, "conn", None)


async def _user_timezone(request: Request, user_id: str) -> str:
    store = getattr(request.app.state, "settings_store", None)
    if store is None or not user_id:
        return ""
    try:
        tz = await store.get_user_or_global(user_id, "timezone")
    except Exception:
        return ""
    return (tz or "").strip()


def _parse_range(start: str | None, end: str | None) -> tuple[datetime, datetime]:
    """Parse ?start / ?end (ISO) into aware UTC bounds, clamped to a sane
    window. Defaults to the current month ±1 week when unspecified."""
    def _p(s: str | None) -> datetime | None:
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            return None

    rs = _p(start)
    re = _p(end)
    now = datetime.now(UTC)
    if rs is None:
        rs = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
    if re is None:
        re = rs + timedelta(days=42)
    if re <= rs:
        re = rs + timedelta(days=1)
    if (re - rs).days > _MAX_WINDOW_DAYS:
        re = rs + timedelta(days=_MAX_WINDOW_DAYS)
    return rs, re


def _svc_details(service_id: str, cfg: dict[str, Any], internal_port: int) -> dict[str, Any]:
    port = internal_port or 5232  # Radicale default
    return {
        "service_id": service_id,
        "base_url": f"http://augmentum-{service_id}:{port}",
        "username": cfg.get("auth_user", ""),
        "password": cfg.get("auth_pass", ""),
        "calendar_path": cfg.get("calendar_path", f"/{service_id}/"),
        "last_synced_at": cfg.get("last_synced_at") or 0,
    }


async def _resolve_caldav_service(conn, user_id: str) -> dict[str, Any] | None:
    """Resolve the user's connected CalDAV service into connection details, or
    None if no calendar server is installed.

    Detection is two-tier so it works from the very first install — BEFORE any
    events have synced (the point the event-cache probe alone would miss):

      1. Event cache — a ``service_id`` the user already has events on (the
         precise match; mirrors the calendar.add verb's path).
      2. Installed service — an enabled ``managed_services`` row that the
         calendar hook has stamped as a calendar (``config_json.calendar_path``,
         written at install even when zero events came back). This is what makes
         the sync status + "add to my devices" toggle light up the moment
         Radicale is installed from Discover, not only after the first event.
    """
    # Tier 1: precise service_id from the user's cached events.
    service_id = ""
    try:
        cur = await conn.execute(
            "SELECT DISTINCT service_id FROM calendar_events "
            "WHERE user_id = ? LIMIT 1",
            (user_id,),
        )
        row = await cur.fetchone()
        await cur.close()
        service_id = row[0] if row else ""
    except Exception:
        service_id = ""
    if service_id:
        try:
            cur2 = await conn.execute(
                "SELECT config_json, internal_port FROM managed_services "
                "WHERE id = ? AND enabled = 1",
                (service_id,),
            )
            row2 = await cur2.fetchone()
            await cur2.close()
            if row2 and row2[0]:
                cfg = json.loads(row2[0]) if isinstance(row2[0], str) else (row2[0] or {})
                return _svc_details(service_id, cfg, row2[1] or 0)
        except Exception:
            pass  # fall through to the installed-service scan

    # Tier 2: any enabled calendar service the hook has stamped.
    try:
        cur3 = await conn.execute(
            "SELECT id, config_json, internal_port FROM managed_services "
            "WHERE enabled = 1",
        )
        rows = await cur3.fetchall()
        await cur3.close()
    except Exception:
        return None
    for sid, cfg_raw, port in rows:
        if not cfg_raw:
            continue
        try:
            cfg = json.loads(cfg_raw) if isinstance(cfg_raw, str) else (cfg_raw or {})
        except Exception:
            continue
        # The calendar hook writes calendar_path (and last_synced_at) at
        # install — its presence is the "this is a calendar server" marker.
        if cfg.get("calendar_path"):
            return _svc_details(sid, cfg, port or 0)
    return None


async def _touch_last_synced(request: Request, conn, service_id: str, calendar_path: str) -> int:
    """Stamp ``last_synced_at`` (epoch) on the service config after a sync so
    the freshness gate (client auto-sync + companion prompt re-sync) agrees on
    when the cache last refreshed. Prefers the service_manager; falls back to a
    direct config_json read-modify-write. Returns the stamped epoch."""
    now = int(_epoch())
    mgr = getattr(request.app.state, "service_manager", None)
    if mgr is not None and hasattr(mgr, "read_config_json") and hasattr(mgr, "update_config_json"):
        try:
            cfg = await mgr.read_config_json(service_id) or {}
            cfg["last_synced_at"] = now
            if calendar_path:
                cfg["calendar_path"] = calendar_path
            await mgr.update_config_json(service_id, cfg)
            return now
        except Exception:
            log.warning("calendar_touch_synced_mgr_failed", exc_info=True)
    # Direct fallback.
    try:
        cur = await conn.execute(
            "SELECT config_json FROM managed_services WHERE id = ?", (service_id,),
        )
        row = await cur.fetchone()
        await cur.close()
        cfg = json.loads(row[0]) if row and row[0] and isinstance(row[0], str) else (row[0] if row else {}) or {}
        cfg["last_synced_at"] = now
        if calendar_path:
            cfg["calendar_path"] = calendar_path
        await conn.execute(
            "UPDATE managed_services SET config_json = ? WHERE id = ?",
            (json.dumps(cfg), service_id),
        )
        await conn.commit()
    except Exception:
        log.warning("calendar_touch_synced_direct_failed", exc_info=True)
    return now


def _epoch() -> float:
    import time
    return time.time()


# ── Unified feed ────────────────────────────────────────────────────────


@router.get("/api/calendar/events")
async def calendar_events(request: Request) -> JSONResponse:
    """Unified event feed for ``[start, end)``. Query params:
      * ``start`` / ``end`` — ISO datetimes (defaults to a 6-week window).
      * ``layers`` — comma list subset of augmentum,calendar,companion
        (default: all three).
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"events": []}, status_code=401)
    conn = _conn(request)
    if conn is None:
        return JSONResponse({"events": []}, status_code=503)

    qp = request.query_params
    rs, re = _parse_range(qp.get("start"), qp.get("end"))
    want = {s.strip() for s in (qp.get("layers") or "").split(",") if s.strip()}
    if not want:
        want = {"augmentum", "calendar", "companion"}

    events: list[dict[str, Any]] = []

    # ── Layer 1: native Augmentum events ──
    if "augmentum" in want:
        try:
            for ev in await ue.list_events(conn, user_id=user_id, range_start=rs, range_end=re):
                events.append({
                    "id": f"aug:{ev['id']}",
                    "native_id": ev["id"],
                    "layer": "augmentum",
                    "title": ev["title"],
                    "start": ev["start"],
                    "end": ev["end"],
                    "all_day": ev["all_day"],
                    "location": ev["location"],
                    "description": ev["description"],
                    "color": ev["color"] or "blue",
                    "editable": True,
                    "recurring": bool(ev.get("_recurring") or ev.get("rrule")),
                    "synced": bool(ev.get("caldav_uid")),
                })
        except Exception as exc:
            log.warning("calendar_native_list_failed", error=str(exc)[:200])

    # ── Layer 2: CalDAV appointments (cached) ──
    if "calendar" in want:
        try:
            from augmentum.calendar.store import list_events as list_caldav
            rows = await list_caldav(
                conn, user_id=user_id, range_start=rs, range_end=re, limit=1000,
            )
            for ev in rows:
                events.append({
                    "id": f"cal:{ev['service_id']}:{ev['uid']}",
                    "layer": "calendar",
                    "title": ev["summary"],
                    "start": ev["start"],
                    "end": ev["end"],
                    "all_day": "T" not in (ev["start"] or ""),
                    "location": ev["location"],
                    "description": ev["description"],
                    "color": "green",
                    "editable": False,
                    "calendar_name": ev.get("calendar_name", ""),
                })
        except Exception as exc:
            log.warning("calendar_caldav_list_failed", error=str(exc)[:200])

    # ── Layer 3: companion standing-task occurrences ──
    if "companion" in want:
        ctx = _tasks_ctx(request)
        if ctx is not None:
            try:
                from augmentum.companion_runtime import standing_tasks
                tz = await _user_timezone(request, user_id)
                tasks = await standing_tasks.list_tasks(
                    ctx.backend.conn, user_id=user_id, companion_id=ctx.companion_id,
                )
                for t in tasks:
                    if not t.enabled:
                        continue
                    occ = standing_tasks.iter_occurrences(
                        params=t.params, interval_seconds=t.interval_seconds,
                        user_timezone=tz, range_start=rs, range_end=re,
                        next_run_at=t.next_run_at,
                    )
                    for moment in occ:
                        iso = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
                        events.append({
                            "id": f"task:{t.id}:{iso}",
                            "task_id": t.id,
                            "layer": "companion",
                            "title": t.title,
                            "kind": t.kind,
                            "start": iso,
                            "end": iso,
                            "all_day": False,
                            "color": "amber",
                            "editable": False,
                            "opens_task": True,
                        })
            except Exception as exc:
                log.warning("calendar_companion_list_failed", error=str(exc)[:200])

    events.sort(key=lambda e: e["start"])
    return JSONResponse({
        "events": events,
        "range": {"start": rs.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": re.strftime("%Y-%m-%dT%H:%M:%SZ")},
    })


# ── Native event CRUD ───────────────────────────────────────────────────


class _EventBody(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    start: str = Field(..., min_length=4, max_length=40)   # ISO
    end: str = Field("", max_length=40)
    all_day: bool = False
    location: str = Field("", max_length=500)
    description: str = Field("", max_length=5000)
    color: str = Field("", max_length=32)
    rrule: str = Field("", max_length=300)
    sync_to_devices: bool = False


class _EventPatchBody(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=300)
    start: str | None = Field(None, max_length=40)
    end: str | None = Field(None, max_length=40)
    all_day: bool | None = None
    location: str | None = Field(None, max_length=500)
    description: str | None = Field(None, max_length=5000)
    color: str | None = Field(None, max_length=32)
    rrule: str | None = Field(None, max_length=300)


def _iso_to_dt(s: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return None


@router.post("/api/calendar/events")
async def calendar_create(body: _EventBody, request: Request) -> JSONResponse:
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"ok": False, "reason": "auth"}, status_code=401)
    conn = _conn(request)
    if conn is None:
        return JSONResponse({"ok": False, "reason": "unavailable"}, status_code=503)

    try:
        event_id = await ue.create_event(
            conn, user_id=user_id, title=body.title, start_dt=body.start,
            end_dt=body.end, all_day=body.all_day, location=body.location,
            description=body.description, color=body.color, rrule=body.rrule,
        )
    except Exception as exc:
        log.warning("calendar_create_failed", error=str(exc)[:200])
        return JSONResponse({"ok": False, "reason": "server"}, status_code=500)

    synced = False
    if body.sync_to_devices:
        synced = await _mirror_to_caldav(conn, user_id, event_id, body)

    return JSONResponse({"ok": True, "id": event_id, "synced": synced})


async def _mirror_to_caldav(conn, user_id: str, event_id: int, body: _EventBody) -> bool:
    """Best-effort outbound create to the user's CalDAV server; records the
    linkage on success. A sync failure never fails the native save."""
    svc = await _resolve_caldav_service(conn, user_id)
    if svc is None:
        return False
    start = _iso_to_dt(body.start)
    if start is None:
        return False
    end = _iso_to_dt(body.end) or (start + timedelta(hours=1))
    try:
        from augmentum.calendar.sync import create_calendar_event
        created = await create_calendar_event(
            base_url=svc["base_url"], username=svc["username"],
            password=svc["password"], calendar_path=svc["calendar_path"],
            summary=body.title, start_dt=start, end_dt=end,
            description=body.description, location=body.location,
        )
    except Exception:
        log.warning("calendar_mirror_failed", exc_info=True)
        return False
    if created is None:
        return False
    href = f"{svc['calendar_path'].rstrip('/')}/{created.uid}.ics"
    try:
        await ue.update_event(conn, event_id, user_id=user_id, fields={
            "caldav_service_id": svc["service_id"],
            "caldav_uid": created.uid,
            "caldav_href": href,
        })
    except Exception:
        log.warning("calendar_mirror_link_failed", exc_info=True)
    return True


@router.patch("/api/calendar/events/{event_id}")
async def calendar_update(event_id: int, body: _EventPatchBody, request: Request) -> JSONResponse:
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"ok": False, "reason": "auth"}, status_code=401)
    conn = _conn(request)
    if conn is None:
        return JSONResponse({"ok": False, "reason": "unavailable"}, status_code=503)

    fields: dict[str, Any] = {}
    mapping = {
        "title": "title", "start": "start_dt", "end": "end_dt",
        "all_day": "all_day", "location": "location",
        "description": "description", "color": "color", "rrule": "rrule",
    }
    for attr, col in mapping.items():
        val = getattr(body, attr)
        if val is not None:
            fields[col] = val
    if not fields:
        return JSONResponse({"ok": False, "reason": "empty"}, status_code=400)
    try:
        changed = await ue.update_event(conn, event_id, user_id=user_id, fields=fields)
    except Exception as exc:
        log.warning("calendar_update_failed", error=str(exc)[:200])
        return JSONResponse({"ok": False, "reason": "server"}, status_code=500)
    if not changed:
        return JSONResponse({"ok": False, "reason": "not_found"}, status_code=404)
    return JSONResponse({"ok": True})


@router.delete("/api/calendar/events/{event_id}")
async def calendar_delete(event_id: int, request: Request) -> JSONResponse:
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"ok": False, "reason": "auth"}, status_code=401)
    conn = _conn(request)
    if conn is None:
        return JSONResponse({"ok": False, "reason": "unavailable"}, status_code=503)
    try:
        removed = await ue.delete_event(conn, event_id, user_id=user_id)
    except Exception as exc:
        log.warning("calendar_delete_failed", error=str(exc)[:200])
        return JSONResponse({"ok": False, "reason": "server"}, status_code=500)
    if removed is None:
        return JSONResponse({"ok": False, "reason": "not_found"}, status_code=404)
    return JSONResponse({"ok": True})


# ── CalDAV services + manual sync ───────────────────────────────────────


@router.get("/api/calendar/services")
async def calendar_services(request: Request) -> JSONResponse:
    """Report the user's connected CalDAV service (if any) so the UI can show
    a sync status and enable the 'add to my devices' toggle."""
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"services": []}, status_code=401)
    conn = _conn(request)
    if conn is None:
        return JSONResponse({"services": []}, status_code=503)
    svc = await _resolve_caldav_service(conn, user_id)
    if svc is None:
        return JSONResponse({"services": [], "sync_available": False})
    return JSONResponse({
        "services": [{"service_id": svc["service_id"], "last_synced_at": svc["last_synced_at"]}],
        "sync_available": True,
        "last_synced_at": svc["last_synced_at"],
    })


@router.get("/api/calendar/connection")
async def calendar_connection(request: Request) -> JSONResponse:
    """Device-setup details for the connected CalDAV server, for the calendar's
    "Add to your devices" card. LAN-only by design: the reachable URL is derived
    from the host the user reached Augmentum on (the same LAN address their
    phone can use) plus Radicale's published host port. Credentials are the
    Augmentum-managed pair (or the user-set pair recorded in config).

    Adding a CalDAV account is inherently a manual OS step on every platform —
    this makes it copy-paste instead of a hunt. Requires an authenticated user;
    the credential is a shared server login, not per-user.
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"installed": False}, status_code=401)
    conn = _conn(request)
    if conn is None:
        return JSONResponse({"installed": False}, status_code=503)
    svc = await _resolve_caldav_service(conn, user_id)
    if svc is None:
        return JSONResponse({"installed": False})

    # Published host port (0 = not LAN-published yet).
    host_port = 0
    try:
        cur = await conn.execute(
            "SELECT host_port FROM managed_services WHERE id = ?", (svc["service_id"],),
        )
        row = await cur.fetchone()
        await cur.close()
        host_port = int(row[0]) if row and row[0] else 0
    except Exception:
        host_port = 0

    # The LAN host is whatever the browser used to reach Augmentum — the same
    # address a phone on the same network resolves. Strip Augmentum's own port.
    host_header = request.headers.get("host", "") or ""
    lan_host = host_header.split(":")[0].strip()
    path = svc["calendar_path"] or f"/{svc['service_id']}/"
    lan_url = (
        f"http://{lan_host}:{host_port}{path}"
        if lan_host and host_port else ""
    )

    # Credentials: user-set (config) win; else the Augmentum-managed derived
    # pair so the card always shows a consistent, working login.
    username = svc.get("username") or ""
    password = svc.get("password") or ""
    if not username or not password:
        try:
            from augmentum.providers.service_auth import managed_service_credentials
            mu, mp = managed_service_credentials(svc["service_id"])
            username = username or mu
            password = password or mp
        except Exception:
            pass

    return JSONResponse({
        "installed": True,
        "service_id": svc["service_id"],
        "lan_url": lan_url,
        "lan_published": bool(host_port),
        "path": path,
        "username": username,
        "password": password,
    })


@router.post("/api/calendar/sync")
async def calendar_sync(request: Request) -> JSONResponse:
    """Pull fresh events from the connected CalDAV server.

    Freshness gate: when ``?if_stale_seconds=N`` is passed and the cache was
    synced less than N seconds ago, this is a cheap no-op (``skipped``) — this
    is how the calendar surface auto-syncs on open without hammering the
    server on every navigation. A bare POST (no gate) always forces a pull.
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        return JSONResponse({"ok": False}, status_code=401)
    conn = _conn(request)
    if conn is None:
        return JSONResponse({"ok": False}, status_code=503)
    svc = await _resolve_caldav_service(conn, user_id)
    if svc is None:
        return JSONResponse({"ok": False, "reason": "no_service"})

    stale_s = request.query_params.get("if_stale_seconds")
    if stale_s:
        try:
            gate = int(stale_s)
            last = int(svc.get("last_synced_at") or 0)
            if last and (_epoch() - last) < gate:
                return JSONResponse({
                    "ok": True, "skipped": True, "last_synced_at": last,
                })
        except (ValueError, TypeError):
            pass

    try:
        from augmentum.calendar.sync import sync_calendar_events
        synced = await sync_calendar_events(
            conn, user_id=user_id, service_id=svc["service_id"],
            base_url=svc["base_url"], username=svc["username"],
            password=svc["password"], calendar_path=svc["calendar_path"],
        )
    except Exception as exc:
        log.warning("calendar_sync_failed", error=str(exc)[:200])
        return JSONResponse({"ok": False, "reason": "sync_error"})
    stamped = await _touch_last_synced(request, conn, svc["service_id"], svc["calendar_path"])
    return JSONResponse({"ok": True, "synced": synced, "last_synced_at": stamped})
