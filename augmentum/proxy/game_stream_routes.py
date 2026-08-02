"""Game streaming endpoints.

The browser hits these routes to discover available games, manage
worlds, start a streaming session, and exchange WebRTC signaling
messages with the container that's actually running the game.

Phase 0 wires everything *except* the WebRTC signaling proxy itself --
that lives in Phase 2 alongside the container infrastructure (the
signaling endpoint is a thin auth-gated proxy in front of Selkies).
This file declares the WS endpoint with a placeholder so the route
table is stable when Phase 2 lands.
"""

from __future__ import annotations

import asyncio
import os
from ipaddress import ip_address, ip_network
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request as UrlRequest, urlopen

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import JSONResponse

from augmentum.config import settings
from augmentum.game_stream import (
    ConcurrentStreamLimitError,
    GameStreamRuntime,
    GameStreamRuntimeError,
)
from augmentum.proxy.game_agent_routes import (
    create_emulator_companion_session,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/game-stream", tags=["game-stream"])

_TAILSCALE_CGNAT = ip_network("100.64.0.0/10")


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _runtime(request: Request) -> GameStreamRuntime | None:
    return getattr(request.app.state, "game_stream_runtime", None)


def _store(request: Request):
    return getattr(request.app.state, "game_stream_store", None)


def _gate(request: Request) -> JSONResponse | None:
    """Master-toggle + dependency check. Returns a 503 response when
    streaming is disabled or the runtime didn't initialise; otherwise
    None and the caller continues. Authentication is checked separately
    so disabled-but-authed callers see 503 (not 401).
    """
    if not getattr(settings, "game_stream_enabled", False):
        return JSONResponse(
            {"error": "Game streaming is disabled"}, status_code=503,
        )
    if _runtime(request) is None or _store(request) is None:
        return JSONResponse(
            {"error": "Game streaming is not available"}, status_code=503,
        )
    return None


def _session_for_client(row: dict, rt: GameStreamRuntime | None) -> dict:
    out = dict(row)
    if rt is not None and getattr(rt, "host_network", False):
        out["allocated_stream_port"] = out.get("stream_port")
        out["stream_port"] = 8080
    return out


def _stream_probe_host() -> str:
    return os.environ.get(
        "AUGMENTUM_GAME_STREAM_PROBE_HOST",
        "host.docker.internal",
    ).strip() or "host.docker.internal"


def _browser_scheme(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-proto", "").split(",", 1)[0]
    if forwarded in {"http", "https"}:
        return forwarded
    return request.url.scheme


def _browser_host(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-host", "").split(",", 1)[0]
    host = forwarded or request.headers.get("host", "")
    return host.strip() or "localhost"


def _host_for_url(host: str) -> str:
    hostname = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    if ":" in hostname and not hostname.startswith("["):
        return f"[{hostname}]"
    return hostname


def _stream_urls(request: Request, port: int) -> dict[str, str]:
    host = _browser_host(request)
    host_only = _host_for_url(host)
    stream_path = f"/stream/{port}/"
    direct_url = f"http://{host_only}:{port}/"
    stream_url = stream_path if _browser_scheme(request) == "https" else direct_url
    return {
        "stream_path": stream_path,
        "stream_url": stream_url,
        "direct_url": direct_url,
        "proxy_target": f"http://{_stream_probe_host()}:{port}/",
    }


def _host_network_kind(host: str) -> str:
    raw = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    raw = raw.strip("[]")
    try:
        ip = ip_address(raw)
    except ValueError:
        return "hostname"
    if ip in _TAILSCALE_CGNAT:
        return "tailscale"
    if ip.is_private:
        return "private"
    if ip.is_loopback:
        return "loopback"
    return "public"


def _probe_stream_root_sync(port: int) -> dict[str, Any]:
    target = f"http://{_stream_probe_host()}:{port}/"
    req = UrlRequest(
        target,
        method="GET",
        headers={"User-Agent": "AugmentumStreamReadiness/1.0"},
    )
    try:
        with urlopen(req, timeout=0.8) as resp:
            status = int(getattr(resp, "status", 0) or 0)
            return {
                "ok": 200 <= status < 500 and status != 502,
                "status": status,
                "target": target,
                "error": "",
            }
    except HTTPError as exc:
        status = int(exc.code or 0)
        return {
            "ok": 200 <= status < 500 and status != 502,
            "status": status,
            "target": target,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "target": target,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _probe_stream_root(port: int) -> dict[str, Any]:
    return await asyncio.to_thread(_probe_stream_root_sync, port)


# ── Profile discovery ────────────────────────────────────────────────


@router.get("/profiles")
async def list_profiles(request: Request) -> JSONResponse:
    """Available game profiles (the Game Portal renders these as tiles)."""
    if (gate := _gate(request)) is not None:
        return gate
    rt = _runtime(request)
    profiles = [
        {
            "id": p.id,
            "display_name": p.display_name,
            "description": p.description,
            "default_resolution": p.default_resolution,
            "default_bitrate_mbps": p.default_bitrate_mbps,
            "supported_encoders": list(p.supported_encoders),
            "recommended_encoder": p.recommended_encoder,
            "multiplayer": p.multiplayer,
            "scriptable": p.scriptable,
            "wants_gamepad": p.wants_gamepad,
            "input_capabilities": p.input_capabilities,
            "settings_schema": p.settings_schema,
        }
        for p in rt.registry.list()
    ]
    return JSONResponse({"profiles": profiles})


# ── Worlds (per-user, persistent) ────────────────────────────────────


@router.get("/worlds")
async def list_worlds(
    request: Request, profile_id: str | None = None,
) -> JSONResponse:
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = _store(request)
    worlds = await store.list_worlds_for_user(user_id=uid, profile_id=profile_id)
    return JSONResponse({"worlds": worlds})


@router.post("/worlds")
async def create_world(request: Request) -> JSONResponse:
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    rt = _runtime(request)
    store = _store(request)

    body: dict[str, Any] = await request.json()
    profile_id = str(body.get("profile_id", "")).strip()
    name = str(body.get("name", "")).strip()
    if not profile_id or not name:
        return JSONResponse(
            {"error": "profile_id and name required"}, status_code=400,
        )
    if not rt.registry.has(profile_id):
        return JSONResponse(
            {"error": f"unknown profile {profile_id!r}"}, status_code=400,
        )
    # Use a local alias so we don't shadow the imported `settings` singleton.
    body_settings = body.get("settings") if isinstance(body.get("settings"), dict) else {}
    whitelist = body.get("whitelist") if isinstance(body.get("whitelist"), list) else []

    world_id = await store.create_world(
        user_id=uid,
        profile_id=profile_id,
        name=name,
        settings=body_settings,
        whitelist=[str(u) for u in whitelist],
    )
    world = await store.get_world(world_id, user_id=uid)
    return JSONResponse({"world": world}, status_code=201)


@router.patch("/worlds/{world_id}")
async def update_world(request: Request, world_id: str) -> JSONResponse:
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = _store(request)
    body: dict[str, Any] = await request.json()
    name = body.get("name")
    body_settings = body.get("settings") if isinstance(body.get("settings"), dict) else None
    whitelist = body.get("whitelist") if isinstance(body.get("whitelist"), list) else None
    ok = await store.update_world(
        world_id,
        user_id=uid,
        name=str(name) if isinstance(name, str) else None,
        settings=body_settings,
        whitelist=[str(u) for u in whitelist] if whitelist is not None else None,
    )
    if not ok:
        return JSONResponse({"error": "World not found"}, status_code=404)
    world = await store.get_world(world_id, user_id=uid)
    return JSONResponse({"world": world})


@router.delete("/worlds/{world_id}")
async def delete_world(request: Request, world_id: str) -> JSONResponse:
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = _store(request)
    ok = await store.delete_world(world_id, user_id=uid)
    if not ok:
        return JSONResponse({"error": "World not found"}, status_code=404)
    return JSONResponse({"ok": True})


# ── Sessions (live stream lifecycle) ─────────────────────────────────


@router.get("/sessions")
async def list_sessions(
    request: Request, live_only: bool = True, limit: int = 20,
) -> JSONResponse:
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = _store(request)
    rt = _runtime(request)
    sessions = await store.list_sessions_for_user(
        user_id=uid, live_only=live_only, limit=limit,
    )
    sessions = [_session_for_client(row, rt) for row in sessions]
    return JSONResponse({"sessions": sessions})


@router.post("/sessions")
async def start_session(request: Request) -> JSONResponse:
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    rt = _runtime(request)
    body: dict[str, Any] = await request.json()
    profile_id = str(body.get("profile_id", "")).strip()
    if not profile_id:
        return JSONResponse({"error": "profile_id required"}, status_code=400)

    world_id = body.get("world_id")
    bitrate = body.get("bitrate_mbps")
    resolution = body.get("resolution")
    encoder = body.get("encoder")
    input_opts = body.get("input") if isinstance(body.get("input"), dict) else {}
    touch_mode = _as_bool(
        input_opts.get("touch_mode", body.get("touch_mode", False)),
        default=False,
    )
    mouse_sensitivity = _bounded_float(
        input_opts.get("mouse_sensitivity", body.get("mouse_sensitivity")),
        min_value=0.01,
        max_value=2.0,
    )
    controller_deadzone = _bounded_float(
        input_opts.get("controller_deadzone", body.get("controller_deadzone")),
        min_value=0.0,
        max_value=0.5,
    )
    gamepad_enabled = _as_bool(
        input_opts.get("gamepad_enabled", body.get("gamepad_enabled", True)),
        default=True,
    )

    # AI co-pilot block. Setting ``companion: true`` triggers the route
    # to create a paired game-agent session BEFORE the container starts
    # so the bridge URL (with its session-scoped token) can be threaded
    # into the container's env. The in-container agent-bridge.py daemon
    # then dials that URL on boot.
    companion_block = body.get("companion") if isinstance(body.get("companion"), dict) else None
    agent_bridge_url = ""
    agent_session_id = ""
    if companion_block:
        # Sensible defaults for the streamed-emulator vocabulary, lifted
        # from the deprecated EmulatorAdapter scaffold. Callers can
        # override per-game by passing their own ``semantic_inputs``.
        default_semantics = [
            "button_a", "button_b", "button_x", "button_y",
            "dpad_up", "dpad_down", "dpad_left", "dpad_right",
            "shoulder_l", "shoulder_r",
            "trigger_l", "trigger_r",
            "start", "select",
        ]
        result = await create_emulator_companion_session(
            request,
            objective=str(companion_block.get("objective") or "play the game"),
            semantic_inputs=list(
                companion_block.get("semantic_inputs") or default_semantics
            ),
            log_schema=str(companion_block.get("log_schema") or "emulator.v1"),
            character_id=companion_block.get("character_id"),
            title_id=companion_block.get("title_id"),
            controller_profile=companion_block.get("controller_profile"),
            game_profile=companion_block.get("game_profile"),
        )
        if isinstance(result, JSONResponse):
            return result
        agent_record, agent_bridge_url = result
        agent_session_id = agent_record.session_id

    # Build the cast-input bridge URL template for this session. The
    # runtime substitutes {session_id} + {token} after minting the
    # token; an empty base URL leaves the template empty and the
    # runtime treats that as "no phone-as-controller for this session".
    from augmentum.config import settings as _settings
    cast_input_base = (
        _settings.cast_input_bridge_base_url
        or _settings.agent_bridge_base_url
        or ""
    ).rstrip("/")
    cast_input_url_template = (
        f"{cast_input_base}/api/cast/input/container-ws/"
        "{session_id}?token={token}"
        if cast_input_base else ""
    )

    try:
        info = await rt.start_session(
            user_id=uid,
            profile_id=profile_id,
            world_id=str(world_id) if isinstance(world_id, str) else None,
            bitrate_mbps=int(bitrate) if isinstance(bitrate, int) else None,
            resolution=str(resolution) if isinstance(resolution, str) else None,
            encoder=str(encoder) if isinstance(encoder, str) else None,
            touch_mode=touch_mode,
            mouse_sensitivity=mouse_sensitivity,
            gamepad_enabled=gamepad_enabled,
            controller_deadzone=controller_deadzone,
            agent_bridge_url=agent_bridge_url,
            cast_input_bridge_url_template=cast_input_url_template,
        )
    except ConcurrentStreamLimitError as exc:
        return JSONResponse({"error": str(exc)}, status_code=429)
    except GameStreamRuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    response: dict[str, Any] = {
        "session_id": info.session_id,
        "profile_id": info.profile_id,
        "status": info.status,
        "stream_port": info.stream_port,
        "game_port": info.game_port,
        "bitrate_mbps": info.bitrate_mbps,
        "resolution": info.resolution,
        "signaling_path": info.signaling_path,
        "input": {
            "touch_mode": touch_mode,
            "mouse_sensitivity": mouse_sensitivity,
            "gamepad_enabled": gamepad_enabled,
            "controller_deadzone": controller_deadzone,
        },
    }
    if agent_session_id:
        # Surfaced so the browser stage can subscribe to the
        # game-agent session's SSE log alongside the stream itself.
        response["agent_session_id"] = agent_session_id
    return JSONResponse(response, status_code=201)


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str) -> JSONResponse:
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = _store(request)
    row = await store.get_session(session_id, user_id=uid)
    if not row:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return JSONResponse(_session_for_client(row, _runtime(request)))


@router.get("/sessions/{session_id}/readiness")
async def get_session_readiness(request: Request, session_id: str) -> JSONResponse:
    """Report whether the iframe stream endpoint is actually usable.

    The browser should poll this endpoint instead of hammering
    `/stream/<port>/` directly. Caddy returns 502 while the Selkies
    upstream is still booting, and those expected retries make the
    console look broken. This endpoint converts the same condition into
    a clean JSON state machine with enough diagnostics to explain the
    failure when startup does not converge.
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    store = _store(request)
    rt = _runtime(request)
    row = await store.get_session(session_id, user_id=uid)
    if not row:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    stored_stream_port = row.get("stream_port")
    stream_port = 8080 if getattr(rt, "host_network", False) else stored_stream_port
    status = str(row.get("status") or "unknown")
    urls = _stream_urls(request, int(stream_port or 0)) if stream_port else {}
    host = _browser_host(request)
    body: dict[str, Any] = {
        "session_id": session_id,
        "status": status,
        "ready": False,
        "stage": "starting_container",
        "message": "Starting the stream container.",
        "stream_port": stream_port,
        "game_port": row.get("game_port"),
        "container_id": row.get("container_id") or "",
        "browser_host": host,
        "browser_host_kind": _host_network_kind(host),
        **urls,
    }

    if status in {"stopped", "crashed"}:
        body.update({
            "stage": status,
            "message": row.get("exit_reason") or f"Session is {status}.",
        })
        return JSONResponse(body)

    container_id = row.get("container_id") or ""
    if not container_id:
        return JSONResponse(body)

    try:
        container_alive = await rt.container_alive(container_id)
    except Exception as exc:
        container_alive = None
        body["container_error"] = f"{type(exc).__name__}: {exc}"
    body["container_alive"] = container_alive

    if container_alive is False:
        await store.update_session(
            session_id,
            user_id=uid,
            status="crashed",
            exit_reason="container_not_running",
        )
        body.update({
            "status": "crashed",
            "stage": "container_stopped",
            "message": "The stream container is no longer running.",
        })
        return JSONResponse(body)

    if not stream_port:
        body.update({
            "stage": "waiting_port",
            "message": "Waiting for a stream port assignment.",
        })
        return JSONResponse(body)

    probe = await _probe_stream_root(int(stream_port))
    body["probe"] = probe
    if probe.get("ok"):
        if status == "starting":
            await rt.mark_ready(session_id, user_id=uid)
            status = "ready"
        body.update({
            "ready": True,
            "status": status,
            "stage": "ready",
            "message": "Stream viewer is ready.",
        })
        return JSONResponse(body)

    body.update({
        "stage": "waiting_stream",
        "message": "Waiting for Selkies to accept connections.",
    })
    return JSONResponse(body)


@router.delete("/sessions/{session_id}")
async def stop_session(request: Request, session_id: str) -> JSONResponse:
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    rt = _runtime(request)
    ok = await rt.stop_session(session_id, user_id=uid, reason="clean")
    if not ok:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return JSONResponse({"ok": True})


@router.post("/sessions/{session_id}/telemetry")
async def post_telemetry(request: Request, session_id: str) -> JSONResponse:
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    rt = _runtime(request)
    store = _store(request)

    # Confirm the session belongs to this user before recording.
    row = await store.get_session(session_id, user_id=uid)
    if not row:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    body: dict[str, Any] = await request.json()
    await rt.heartbeat(session_id, user_id=uid)
    await rt.record_telemetry(
        session_id=session_id,
        user_id=uid,
        rtt_ms=_as_float(body.get("rtt_ms")),
        jitter_ms=_as_float(body.get("jitter_ms")),
        packet_loss=_as_float(body.get("packet_loss")),
        bitrate_kbps=_as_int(body.get("bitrate_kbps")),
        fps=_as_float(body.get("fps")),
    )
    return JSONResponse({"ok": True})


@router.post("/sessions/{session_id}/heartbeat")
async def post_heartbeat(request: Request, session_id: str) -> JSONResponse:
    """Keep a mounted stream stage marked connected.

    This is intentionally tiny and cheap: it updates ``last_seen_at`` and
    promotes ready/idle sessions back to connected. Without it, a sleeping
    mobile browser or quiet iframe can leave the server with stale state
    and the next foreground/reload looks like a server outage.
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    ok = await _runtime(request).heartbeat(session_id, user_id=uid)
    if not ok:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    row = await _store(request).get_session(session_id, user_id=uid)
    return JSONResponse({
        "ok": True,
        "status": (row or {}).get("status", "connected"),
    })


# ── Signaling (WebRTC handshake; Phase 2 wires the proxy body) ──────


@router.websocket("/signal/{session_id}")
async def signaling(websocket: WebSocket, session_id: str) -> None:
    """WebRTC signaling proxy.

    Phase 0 stub: refuses immediately so the client surfaces the right
    error. Phase 2 replaces the body with the Selkies-side proxy and
    MUST authenticate + verify ``session_id`` belongs to the connecting
    user BEFORE forwarding any frames -- otherwise this becomes a free
    proxy to arbitrary internal containers.
    """
    # Master toggle: if streaming is disabled, refuse before accepting.
    if not getattr(settings, "game_stream_enabled", False):
        await websocket.close(code=1008)  # 1008 = policy violation
        return
    await websocket.accept()
    try:
        await websocket.send_json({
            "type": "error",
            "code": "signaling_not_ready",
            "message": "Game streaming signaling not available in this build.",
            "session_id": session_id,
        })
    finally:
        await websocket.close()


# ── helpers ──────────────────────────────────────────────────────────


def _as_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def _as_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    return None


def _as_bool(v: Any, *, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        lowered = v.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return default


def _bounded_float(
    v: Any,
    *,
    min_value: float,
    max_value: float,
) -> float | None:
    parsed = _as_float(v)
    if parsed is None:
        return None
    return min(max(parsed, min_value), max_value)
