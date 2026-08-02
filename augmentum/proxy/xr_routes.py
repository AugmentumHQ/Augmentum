"""WebXR app-surface routes.

These endpoints make the browser XR runtime durable: preflight capabilities,
session creation/resume, seat calibration, and lightweight telemetry.  The
WebXR APIs still live entirely in the browser; the server provides the app
spine that lets a Quest Browser/PWA experience recover like a native app.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
from http.cookies import SimpleCookie
from typing import Any
from urllib.parse import urljoin, urlparse

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from augmentum.xr.browser_panel import ChromiumNotAvailable, XRBrowserPanelManager
from augmentum.xr.session import DEFAULT_ROOM_ID, DEFAULT_SEAT_ID
from augmentum.utils.logging import get_logger

router = APIRouter(prefix="/api/xr", tags=["xr"])
log = get_logger(__name__)


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _store(request: Request):
    store = getattr(request.app.state, "xr_store", None)
    if store is None:
        raise HTTPException(503, "XR session store unavailable")
    return store


def _browser_panel_manager_for_app(app) -> XRBrowserPanelManager:
    manager = getattr(app.state, "xr_browser_panel_manager", None)
    if manager is None:
        manager = XRBrowserPanelManager()
        app.state.xr_browser_panel_manager = manager
    return manager


def _browser_panel_manager(request: Request) -> XRBrowserPanelManager:
    return _browser_panel_manager_for_app(request.app)


def _is_local_host(hostname: str, *, request_host: str = "") -> bool:
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return False
    request_host = (request_host or "").split(":", 1)[0].strip().lower().rstrip(".")
    if request_host and host == request_host:
        return True
    if host in {"localhost", "127.0.0.1", "::1", "host.docker.internal"}:
        return True
    if host.endswith(".local") or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local)


def _resolve_local_browser_url(request: Request, raw_url: str) -> str:
    raw = str(raw_url or "").strip()
    if not raw:
        raise HTTPException(400, "url required")
    resolved = urljoin(str(request.base_url), raw)
    parsed = urlparse(resolved)
    scheme = parsed.scheme.lower()
    request_host = request.headers.get("host", "") or request.url.netloc
    if scheme not in {"https", "http"}:
        raise HTTPException(400, "only http/https URLs are supported")
    if scheme == "http" and not _is_local_host(parsed.hostname or "", request_host=request_host):
        raise HTTPException(400, "http browser panels are limited to local origins")
    if not _is_local_host(parsed.hostname or "", request_host=request_host):
        raise HTTPException(400, "XR browser panels are limited to local/private HTTPS pages")
    return resolved


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _same_request_origin(request: Request, resolved_url: str) -> bool:
    request_origin = urlparse(str(request.base_url))
    target = urlparse(resolved_url)
    return (
        request_origin.scheme == target.scheme
        and (request_origin.hostname or "").lower() == (target.hostname or "").lower()
        and (request_origin.port or _default_port(request_origin.scheme))
        == (target.port or _default_port(target.scheme))
    )


def _same_origin_browser_context(request: Request, resolved_url: str) -> tuple[dict[str, str], dict[str, str]]:
    if not _same_request_origin(request, resolved_url):
        return {}, {}

    auth_headers: dict[str, str] = {}
    authorization = request.headers.get("authorization")
    if authorization:
        auth_headers["Authorization"] = authorization

    cookies: dict[str, str] = {}
    raw_cookie = request.headers.get("cookie") or ""
    if raw_cookie:
        jar = SimpleCookie()
        try:
            jar.load(raw_cookie)
        except Exception:
            jar = SimpleCookie()
        cookies = {name: morsel.value for name, morsel in jar.items()}
    return auth_headers, cookies


class XRSessionCreate(BaseModel):
    session_id: str | None = None
    surface: str = "voice"
    voice_session_id: str = ""
    room_id: str = DEFAULT_ROOM_ID
    seat_id: str = DEFAULT_SEAT_ID
    device_hint: dict[str, Any] = Field(default_factory=dict)
    pwa: bool = False


class XRSessionPatch(BaseModel):
    status: str | None = None
    surface: str | None = None
    voice_session_id: str | None = None
    room_id: str | None = None
    seat_id: str | None = None
    device_hint: dict[str, Any] | None = None
    room_state: dict[str, Any] | None = None
    input_preferences: dict[str, Any] | None = None
    performance_profile: dict[str, Any] | None = None
    last_snapshot: dict[str, Any] | None = None


class XREventCreate(BaseModel):
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class XRBrowserPanelCreate(BaseModel):
    url: str
    width: int = Field(default=1440, ge=320, le=3840)
    height: int = Field(default=900, ge=240, le=2160)
    device_scale_factor: float = Field(default=1.0, ge=0.5, le=3.0)
    format: str = "jpeg"
    quality: int = Field(default=82, ge=40, le=95)


class XRBrowserPanelInput(BaseModel):
    type: str
    x: float = 0.5
    y: float = 0.5
    normalized: bool = True
    deltaX: float = 0.0
    deltaY: float = 0.0
    text: str = ""
    key: str = ""


class XRSeatPut(BaseModel):
    label: str = "Default seat"
    x: float = -0.30
    y: float = 0.0
    z: float = 2.30
    rotY: float = 3.141592653589793
    envId: str = DEFAULT_ROOM_ID
    avatar: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.get("/capabilities")
async def get_xr_capabilities(request: Request) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    return JSONResponse(await _store(request).capabilities())


@router.post("/sessions")
async def create_xr_session(body: XRSessionCreate, request: Request) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    session = await _store(request).create_or_resume_session(
        user_id=uid,
        session_id=body.session_id,
        surface=body.surface,
        voice_session_id=body.voice_session_id,
        room_id=body.room_id,
        seat_id=body.seat_id,
        device_hint=body.device_hint,
        pwa=body.pwa,
    )
    return JSONResponse({"ok": True, "session": session}, status_code=201)


@router.get("/sessions/{session_id}/resume")
async def get_xr_resume(session_id: str, request: Request) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    snapshot = await _store(request).resume_snapshot(session_id, user_id=uid)
    if snapshot is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(snapshot)


@router.patch("/sessions/{session_id}")
async def patch_xr_session(
    session_id: str, body: XRSessionPatch, request: Request
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    ok = await _store(request).patch_session(
        session_id,
        user_id=uid,
        status=body.status,
        surface=body.surface,
        voice_session_id=body.voice_session_id,
        room_id=body.room_id,
        seat_id=body.seat_id,
        device_hint=body.device_hint,
        room_state=body.room_state,
        input_preferences=body.input_preferences,
        performance_profile=body.performance_profile,
        last_snapshot=body.last_snapshot,
    )
    if not ok:
        return JSONResponse({"error": "Not found"}, status_code=404)
    session = await _store(request).get_session(session_id, user_id=uid)
    return JSONResponse({"ok": True, "session": session})


@router.post("/sessions/{session_id}/events")
async def post_xr_event(
    session_id: str, body: XREventCreate, request: Request
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    event_id = await _store(request).record_event(
        session_id,
        user_id=uid,
        event_type=body.type,
        payload=body.payload,
    )
    if event_id is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({"ok": True, "event_id": event_id})


@router.post("/browser-panels")
async def create_xr_browser_panel(body: XRBrowserPanelCreate, request: Request) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    url = _resolve_local_browser_url(request, body.url)
    auth_headers, cookies = _same_origin_browser_context(request, url)
    manager = _browser_panel_manager(request)
    try:
        panel = await manager.create(
            user_id=uid,
            url=url,
            width=body.width,
            height=body.height,
            device_scale_factor=body.device_scale_factor,
            image_format=body.format,
            quality=body.quality,
            auth_headers=auth_headers,
            cookies=cookies,
        )
    except ChromiumNotAvailable as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except Exception as exc:
        return JSONResponse({"error": f"XR browser panel failed: {str(exc)[:200]}"}, status_code=502)
    return JSONResponse({
        "ok": True,
        "panel": {
            "id": panel.id,
            "url": panel.url,
            "width": panel.width,
            "height": panel.height,
            "revision": panel.revision,
            "frame_url": f"/api/xr/browser-panels/{panel.id}/frame",
        },
    }, status_code=201)


@router.get("/browser-panels/{panel_id}")
async def get_xr_browser_panel(panel_id: str, request: Request) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    panel = await _browser_panel_manager(request).get(panel_id, user_id=uid)
    if panel is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({
        "ok": True,
        "panel": {
            "id": panel.id,
            "url": panel.url,
            "width": panel.width,
            "height": panel.height,
            "revision": panel.revision,
            "frame_url": f"/api/xr/browser-panels/{panel.id}/frame?rev={panel.revision}",
        },
    })


@router.get("/browser-panels/{panel_id}/frame")
async def get_xr_browser_panel_frame(panel_id: str, request: Request) -> Response:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    panel = await _browser_panel_manager(request).capture(panel_id, user_id=uid)
    if panel is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return Response(
        panel.last_frame,
        media_type=panel.last_media_type,
        headers={
            "Cache-Control": "no-store",
            "X-Augmentum-XR-Panel-Revision": str(panel.revision),
        },
    )


@router.websocket("/browser-panels/{panel_id}/stream")
async def stream_xr_browser_panel(websocket: WebSocket, panel_id: str) -> None:
    await websocket.accept()
    user = websocket.scope.get("user")
    uid = user.id if user else ""
    if not uid:
        await websocket.close(code=4403, reason="Unauthorized")
        return
    manager = _browser_panel_manager_for_app(websocket.app)
    panel = await manager.get(panel_id, user_id=uid)
    if panel is None:
        await websocket.close(code=4404, reason="XR browser panel not found")
        return
    try:
        await websocket.send_json({
            "type": "started",
            "panel_id": panel.id,
            "revision": panel.revision,
            "width": panel.width,
            "height": panel.height,
            "media_type": panel.last_media_type,
        })
        async for frame in manager.stream_frames(panel, image_format=panel.image_format, quality=panel.quality):
            await websocket.send_json(frame)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as exc:
        log.warning("xr_browser_panel_stream_failed", panel_id=panel_id, error=str(exc))
        try:
            await websocket.send_json({"type": "error", "message": str(exc)[:200]})
        except Exception as send_exc:
            log.debug(
                "xr_browser_panel_error_emit_failed",
                panel_id=panel_id,
                error=str(send_exc),
            )
    finally:
        try:
            await websocket.close()
        except Exception as close_exc:
            log.debug(
                "xr_browser_panel_ws_close_failed",
                panel_id=panel_id,
                error=str(close_exc),
            )


@router.post("/browser-panels/{panel_id}/input")
async def post_xr_browser_panel_input(
    panel_id: str, body: XRBrowserPanelInput, request: Request
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    try:
        panel = await _browser_panel_manager(request).input(
            panel_id,
            user_id=uid,
            event=body.model_dump(),
        )
    except Exception as exc:
        return JSONResponse({"error": f"XR browser input failed: {str(exc)[:200]}"}, status_code=502)
    if panel is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({
        "ok": True,
        "revision": panel.revision,
        "frame_url": f"/api/xr/browser-panels/{panel.id}/frame?rev={panel.revision}",
    })


@router.delete("/browser-panels/{panel_id}")
async def delete_xr_browser_panel(panel_id: str, request: Request) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    closed = await _browser_panel_manager(request).close(panel_id, user_id=uid)
    return JSONResponse({"ok": closed})


@router.get("/sessions/{session_id}/events")
async def list_xr_events(
    session_id: str,
    request: Request,
    limit: int = Query(default=200, ge=1, le=500),
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    events = await _store(request).list_events(session_id, user_id=uid, limit=limit)
    if events is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({"events": events})


@router.put("/seats/{seat_id}")
async def put_xr_seat(
    seat_id: str, body: XRSeatPut, request: Request
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    seat = await _store(request).upsert_seat(
        user_id=uid,
        seat_id=seat_id,
        label=body.label,
        x=body.x,
        y=body.y,
        z=body.z,
        rot_y=body.rotY,
        env_id=body.envId,
        avatar=body.avatar,
        metadata=body.metadata,
    )
    return JSONResponse({"ok": True, "seat": seat})


@router.get("/seats/{seat_id}")
async def get_xr_seat(seat_id: str, request: Request) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    seat = await _store(request).get_seat(user_id=uid, seat_id=seat_id)
    return JSONResponse({"seat": seat})
