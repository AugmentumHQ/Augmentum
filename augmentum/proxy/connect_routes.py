"""Connect routes — HTTP surface + WebSocket signaling.

Three endpoints today:

* ``GET  /api/connect/turn-credentials``  — mints HMAC ephemeral
  credentials so the UI can build an ``RTCIceServer`` config.
* ``GET  /api/connect/presence``          — snapshot of who's online
  (Phase 1: just the requesting user's online same-instance peers).
* ``WS   /api/connect/signaling``         — the signaling channel.

Feature-gated by ``settings.connect_enabled``. With the flag off
(default), HTTP endpoints return 503 and WS connections close with
policy-violation. Mirrors the fabric routes' pattern so the user
sees a consistent "subsystem disabled" signal across the OS.

Phase 1 scope of the WS endpoint:

* Accept + identify + ship welcome envelope (with TURN creds).
* Maintain presence via ConnectHub.
* Echo ``ping`` → ``pong`` for keepalive.
* Stub handlers for invite/offer/answer/ICE/hangup that ACK back
  but don't yet route to peers — peer-routing requires the
  ``connect_contacts`` DID → user_id resolver, which is the next
  task. The envelope contract is stable; the routing layer plugs
  in behind it without breaking clients.

See ``docs/superpowers/specs/2026-06-01-connect-and-os-positioning-design.md``
for the broader design + sequencing.
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from starlette.responses import JSONResponse

from augmentum.calling.turn_credentials import mint_ephemeral
from augmentum.config import settings
from augmentum.connect.call_routing import (
    handle_call_action,
    handle_signaling_envelope,
    new_party_id,
)
from augmentum.connect.contacts import local_did_for
from augmentum.connect.hub import ConnectHub
from augmentum.connect.message_routing import (
    handle_message_action,
    handle_message_envelope,
)
from augmentum.connect.protocol import (
    EVENT_ERROR,
    EVENT_PONG,
    EVENT_TEXT_DELIVERED,
    EVENT_WELCOME,
    MAX_ENVELOPE_BYTES,
    MSG_PING,
    MSG_TEXT_DELETE,
    MSG_TEXT_DELIVERED,
    MSG_TEXT_EDIT,
    MSG_TEXT_REACT,
    MSG_TEXT_READ,
    MSG_TEXT_SEND,
    MSG_TYPING_START,
    MSG_TYPING_STOP,
    ConnectEnvelope,
    deserialise_envelope,
    serialise_envelope,
)
from augmentum.connect.rate_limit import KeyedRateLimiter, WsRateLimiter
from augmentum.notifications import (
    NotificationHub,
    register_action_handler,
)
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


router = APIRouter(prefix="/api/connect", tags=["connect"])

# Per-IP throttle for the PUBLIC, unauthenticated guest-session bootstrap.
_GUEST_SESSION_LIMITER = KeyedRateLimiter(limit=30, window_s=60.0)


# Register the Connect call action handler against the notification
# substrate at module import. ``connect.call.*`` notifications (incoming
# / missed) get accept/decline routed through ConnectHub by
# ``handle_call_action``. Idempotent — re-importing replaces the handler
# rather than stacking duplicates.
register_action_handler("connect.call.*", handle_call_action)
register_action_handler("connect.message*", handle_message_action)


# ── Helpers ───────────────────────────────────────────────────────


def _connect_enabled() -> bool:
    return bool(getattr(settings, "connect_enabled", False))


def _turn_url() -> str:
    """Compose the canonical turn: URL the browser plugs in."""

    host = os.environ.get("AUGMENTUM_TURN_HOST", "localhost")
    port = os.environ.get("AUGMENTUM_TURN_PORT", "3478")
    return f"turn:{host}:{port}?transport=udp"


def _user_did_for(user_id: str) -> str:
    """Derive the user's DID surface form. Thin wrapper around the
    canonical helper in ``augmentum.connect.contacts``."""

    return local_did_for(user_id)


def _get_conn(request_or_ws):
    """Resolve aiosqlite conn from app state. Works for Request or WebSocket."""

    app = getattr(request_or_ws, "app", None)
    sm = getattr(getattr(app, "state", None), "state_manager", None)
    if sm is not None and isinstance(getattr(sm, "backend", None), SQLiteBackend):
        return sm.backend.conn
    return None


def _get_notification_hub(app) -> NotificationHub:
    hub = getattr(app.state, "notification_hub", None)
    if hub is None:
        hub = NotificationHub()
        app.state.notification_hub = hub
        _wire_hub_to_fabric(app, notification_hub=hub)
    return hub


def _local_url_base(request_or_ws) -> str:
    """Derive THIS instance's public-facing URL base for attachment
    fetch URLs sent to fabric peers.

    Uses the inbound request's scheme + host — works behind a reverse
    proxy as long as the proxy forwards Host. When the inbound side
    is a WebSocket, prefer the same host header. Falls back to empty
    string when neither is available (tests / unit-runs without an
    HTTP context).
    """
    headers = getattr(request_or_ws, "headers", None)
    if headers is None:
        return ""
    host = headers.get("host") or headers.get("x-forwarded-host")
    if not host:
        return ""
    proto = (
        headers.get("x-forwarded-proto")
        or getattr(getattr(request_or_ws, "url", None), "scheme", None)
        or "http"
    )
    # WebSocket schemes (ws / wss) don't apply to fetch URLs.
    if proto in ("ws", "wss"):
        proto = "https" if proto == "wss" else "http"
    return f"{proto}://{host}"


def _wire_hub_to_fabric(app, *, connect_hub=None, notification_hub=None) -> None:
    """Wire the local hubs into the fabric coordinator if fabric is up.

    Idempotent — calling repeatedly with the same hub is fine. The
    coordinator uses the hubs to re-inject inbound MSG_CONNECT_ENVELOPE
    frames into the local signaling fan-out (Wedge B fabric routing).
    When fabric is disabled or hasn't started yet, this is a no-op.
    """
    coord = getattr(app.state, "fabric_coordinator", None)
    if coord is None:
        return
    if connect_hub is not None:
        coord.connect_hub = connect_hub
    if notification_hub is not None:
        coord.notification_hub = notification_hub


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    if user is None:
        raise HTTPException(status_code=401, detail="auth required")
    return user.id


def _username_for(request: Request) -> str:
    """Best-effort username of the authed caller (for the @handle surface)."""
    user = request.scope.get("user")
    return getattr(user, "username", "") if user is not None else ""


# ── HTTP ─────────────────────────────────────────────────────────


@router.get("/turn-credentials")
async def get_turn_credentials(request: Request) -> dict[str, Any]:
    """Return an ephemeral TURN credential bundle for the requester.

    Response shape:
        {
            "ice_servers": [
                {"urls": [...], "username": "...", "credential": "..."}
            ],
            "expires_at": <unix-ts>,
        }

    The UI uses ``ice_servers`` directly in ``new RTCPeerConnection({...})``.
    ``expires_at`` lets the UI schedule a re-mint before the cred TTL
    runs out (default 24h, so most calls never need re-mint).
    """

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    uid = _user_id(request)
    creds = mint_ephemeral(uid)
    return {
        "ice_servers": [creds.as_ice_server(_turn_url())],
        "expires_at": creds.expires_at,
    }


@router.get("/presence")
async def get_presence(request: Request) -> dict[str, Any]:
    """Same-instance online-users snapshot.

    Phase 1: returns every Connect-attached user_id on this instance
    other than the requester. The contact-scoped variant (only return
    users in the requester's connect_contacts) ships when the contacts
    store is being populated.
    """

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    uid = _user_id(request)
    hub = _get_hub(request)
    others = [u for u in hub.online_user_ids() if u != uid]
    return {
        "online_user_ids": others,
        "server_time": int(time.time()),
    }


@router.get("/directory")
async def get_directory(request: Request) -> dict[str, Any]:
    """Auto-discovered, mutual-consent peer directory.

    Returns the list of users this caller can see based on the
    "mutual enablement as consent" model from the design spec:

      * Both the caller AND the target have ``connect_enabled = True``
      * Both have ``connect_discoverable_same_instance = True``
        (the matching discoverability scope is on for both)
      * The target is an active user account (``is_active = 1``)
      * Self is excluded

    No request/accept dance — the substrate IS the consent. Either
    party can flip their discoverability off at any time and the
    other side stops seeing them.

    Response shape::

        {
          "people": [
            {
              "peer_did":     "alice@this-instance",
              "user_id":      "alice",
              "display_name": "Alice",
              "online":       true,
              "accepts_calls":    true,
              "accepts_messages": true,
              "discovery_source": "same_instance"
            },
            ...
          ],
          "self_discoverable_same_instance": true,
          "self_discoverable_fabric_peers":  false,
          "server_time": <epoch>
        }

    Fabric peers are returned with ``discovery_source = "fabric"`` once
    cross-instance dispatch lands; today they are excluded since the
    routing layer still returns ``fabric_routing_pending``.
    """

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    uid = _user_id(request)
    conn = _get_conn(request)
    hub = _get_hub(request)
    if conn is None:
        return {
            "people": [],
            "self_discoverable_same_instance": False,
            "self_discoverable_fabric_peers": False,
            "server_time": int(time.time()),
        }

    # Same-instance is an INTERNAL DIRECTORY: every active non-guest user on this
    # machine is a possible contact, so the directory is shown to any user
    # regardless of their own discoverability, and lists everyone who hasn't
    # explicitly hidden themselves. ``connectDiscoverableSameInstance`` is now an
    # opt-OUT (default visible); we still read the caller's value so the UI can
    # offer a "hide me from the directory" toggle. Fabric stays opt-in (a
    # different trust boundary). Keys use the ``ui.`` prefix (see _UI_SETTINGS).
    self_same = await _read_user_setting_bool(
        conn, uid, "ui.connectDiscoverableSameInstance",
        default=True,
    )
    self_fabric = await _read_user_setting_bool(
        conn, uid, "ui.connectDiscoverableFabricPeers",
        default=False,
    )

    rows = await _query_discoverable_same_instance_peers(conn, uid)
    people = await _enrich_people(conn, hub, rows)

    # Sort: online first, then alpha by display_name.
    people.sort(key=lambda p: (
        0 if p["online"] else 1,
        (p["display_name"] or "").lower(),
    ))

    return {
        "people": people,
        "self_discoverable_same_instance": self_same,
        "self_discoverable_fabric_peers": self_fabric,
        "server_time": int(time.time()),
    }


async def _search_discoverable_peers(
    conn: Any, caller_user_id: str, query: str, *, limit: int = 25,
) -> list[tuple[str, str, str]]:
    """Same-instance peers whose username/display_name match the query.

    Internal-directory model: every active non-guest user is reachable unless
    they explicitly hid themselves (opt-OUT), with an added case-insensitive
    substring match on username or display_name. Shares the machine-account
    exclusion with ``_query_discoverable_same_instance_peers`` -- search and
    browse must agree on who is a person, or a fabric row filtered out of the
    list reappears the moment the user types. The ``query`` is matched as a
    LIKE pattern with SQL wildcards escaped so a user typing ``%`` can't widen
    the search.
    """
    safe = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    like = f"%{safe}%"
    cur = await conn.execute(
        """
        SELECT u.id, COALESCE(NULLIF(u.display_name, ''), u.username) AS dn,
               u.username
          FROM users u
          LEFT JOIN user_settings se
            ON se.user_id = u.id AND se.key = 'ui.connectEnabled'
          LEFT JOIN user_settings sd
            ON sd.user_id = u.id AND sd.key = 'ui.connectDiscoverableSameInstance'
         WHERE u.is_active = 1
           AND u.id != ?
           AND LOWER(COALESCE(se.value, 'true')) IN ('1', 'true', 'yes', 'on')
           AND LOWER(COALESCE(sd.value, 'true')) NOT IN ('0', 'false', 'no', 'off')
           AND (u.role IS NULL OR u.role != 'guest')
           AND (u.role IS NULL OR u.role != 'peer')
           AND u.id NOT LIKE 'fabric:%'
           AND (u.username LIKE ? ESCAPE '\\' OR u.display_name LIKE ? ESCAPE '\\')
         ORDER BY u.username
         LIMIT ?
        """,
        (caller_user_id, like, like, max(1, min(limit, 50))),
    )
    rows = await cur.fetchall()
    await cur.close()
    return [(r[0], r[1], r[2]) for r in rows]


@router.get("/search")
async def search_directory(request: Request) -> dict[str, Any]:
    """Find discoverable same-instance peers by handle / display name.

    Replaces "paste the full DID" — the New-conversation and call-picker search
    fields point here. Requires the caller to be discoverable themselves (same
    mutual-consent gate as the directory). Returns ``[]`` for a blank query or
    when the caller isn't discoverable (with a ``self_discoverable`` hint).
    """
    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    uid = _user_id(request)
    conn = _get_conn(request)
    hub = _get_hub(request)
    query = (request.query_params.get("q") or "").strip()
    if conn is None or not query:
        return {"people": [], "self_discoverable": False, "server_time": int(time.time())}

    # Any user can search the internal directory (everyone on the machine is a
    # possible contact); results list everyone who hasn't hidden themselves.
    rows = await _search_discoverable_peers(conn, uid, query)
    people = await _enrich_people(conn, hub, rows)
    people.sort(key=lambda p: (0 if p["online"] else 1, (p["display_name"] or "").lower()))
    return {
        "people": people,
        "server_time": int(time.time()),
    }


@router.get("/profile")
async def get_my_profile(request: Request) -> dict[str, Any]:
    """Return the caller's Connect profile (bio / status)."""
    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")
    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="storage unavailable")
    from augmentum.connect.profile_store import get_profile
    profile = await get_profile(conn, user_id=uid)
    profile["handle"] = _handle_for(_username_for(request))
    return {"profile": profile}


@router.put("/profile")
async def update_my_profile(request: Request) -> dict[str, Any]:
    """Patch the caller's Connect profile. Only the provided fields change."""
    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")
    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="storage unavailable")
    body = await request.json()
    from augmentum.connect.profile_store import upsert_profile
    profile = await upsert_profile(
        conn, user_id=uid,
        bio=body.get("bio"),
        status_message=body.get("status_message"),
        status_emoji=body.get("status_emoji"),
        avatar_ref=body.get("avatar_ref"),
    )
    return {"profile": profile}


# ── Guest pass (Phase 3a) ───────────────────────────────────────────────────
#
# The PUBLIC ``/guest/session`` exchanges a durable grant token (the saved PWA's
# credential) for a scoped guest session — same shape as the cast guest path.
# The HOST-authed ``/guests`` management routes (list/revoke/scopes) are plural
# and stay behind normal auth (a host manages only their own guests).


def _get_session_manager(request: Request):
    return getattr(request.app.state, "session_manager", None)


@router.post("/guest/session")
async def guest_session(request: Request) -> Any:
    """PUBLIC: durable grant token → scoped guest session (Set-Cookie).

    The saved homescreen surface calls this on launch to (re)establish a
    session. 410 when the grant is revoked or the guest account is disabled —
    the surface renders "access ended".
    """
    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")
    allowed, retry = _GUEST_SESSION_LIMITER.check(_get_ip_from(request))
    if not allowed:
        return JSONResponse(
            {"error": "Too many requests. Please slow down."},
            status_code=429, headers={"Retry-After": str(retry)},
        )
    conn = _get_conn(request)
    sm = _get_session_manager(request)
    if conn is None or sm is None:
        return JSONResponse({"error": "service unavailable"}, status_code=503)
    body = await request.json()
    grant_token = str(body.get("grant_token") or "")

    from augmentum.connect.contacts import display_name_for_did
    from augmentum.connect.guest_grant_store import (
        get_by_token,
        grant_is_live,
        touch_last_used,
    )

    grant = await get_by_token(conn, grant_token)
    if grant is None or not grant_is_live(grant):
        return JSONResponse({"error": "This guest access has ended."}, status_code=410)
    guest = await sm.get_user_by_id(grant["guest_user_id"])
    if guest is None or not guest.is_active:
        return JSONResponse({"error": "This guest access has ended."}, status_code=410)

    await touch_last_used(conn, grant_id=grant["grant_id"])
    token = await sm.create_session(
        guest.id, ip_address=_get_ip_from(request),
        user_agent=request.headers.get("user-agent", ""), source="connect-guest",
    )
    host_name = await display_name_for_did(conn, grant["host_did"])
    resp = JSONResponse({
        "guest": {"user_id": guest.id, "display_name": guest.display_name},
        "host": {"did": grant["host_did"], "display_name": host_name},
        "scopes": (grant["scopes"] or "text").split(","),
    })
    from augmentum.proxy.auth_routes import _set_session_cookie
    return _set_session_cookie(resp, token, request=request)


@router.post("/guest/ping")
async def guest_ping(request: Request) -> Any:
    """Wake a peer to "request a call/text" (Phase 3d push-wake).

    Auth-gated (the caller has a session). Sends a Web Push / notification to the
    target via the existing ``publish_and_dispatch`` so a closed app still rings.
    A guest caller is scope-checked: they may ping only their granted host, and
    ``kind='call'`` requires the call scope. The actual call/text then proceeds
    over normal Connect once the woken peer opens the app.
    """
    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")
    user = request.scope.get("user")
    if user is None:
        raise HTTPException(status_code=401, detail="auth required")
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="storage unavailable")
    body = await request.json()
    kind = body.get("kind") if body.get("kind") in ("call", "text") else "text"
    peer_did = str(body.get("peer_did") or "")

    from augmentum.connect.contacts import display_name_for_did, resolve_peer_did
    resolved = resolve_peer_did(peer_did)
    if resolved is None or resolved.kind != "local":
        raise HTTPException(status_code=400, detail="a same-instance peer is required")
    target_user_id = resolved.address

    # A guest may ping ONLY their granted host, and call only if call-scoped.
    if getattr(user, "role", "") == "guest":
        from augmentum.connect.guest_grant_store import grant_allows, is_guest_of
        ok = (
            await grant_allows(conn, guest_user_id=user.id, host_user_id=target_user_id, scope="call")
            if kind == "call"
            else await is_guest_of(conn, guest_user_id=user.id, host_user_id=target_user_id)
        )
        if not ok:
            raise HTTPException(status_code=403, detail="guest_scope_violation")

    from augmentum.notifications import IMPORTANCE_CRITICAL, IMPORTANCE_HIGH
    from augmentum.notifications.hub import publish_and_dispatch
    sender_name = await display_name_for_did(conn, _user_did_for(user.id)) or user.display_name
    verb = "call" if kind == "call" else "message"
    nid = await publish_and_dispatch(
        conn,
        hub=_get_notification_hub(request.app),
        user_id=target_user_id,
        channel_id="connect.message",
        source="connect",
        title=f"{sender_name} wants to {verb} you",
        body="Open Connect to respond.",
        importance=IMPORTANCE_CRITICAL if kind == "call" else IMPORTANCE_HIGH,
        dedupe_key=f"guest-ping:{user.id}:{kind}",
        payload={"from_did": _user_did_for(user.id), "kind": kind, "ping": True},
        icon="phone" if kind == "call" else "message",
    )
    return {"woke": True, "notification_id": nid}


@router.get("/guests")
async def list_my_guests(request: Request) -> dict[str, Any]:
    """HOST: list the guests I've invited (status, scopes, last_used)."""
    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")
    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="storage unavailable")
    from augmentum.connect.guest_grant_store import list_for_host
    return {"guests": await list_for_host(conn, host_user_id=uid)}


@router.patch("/guests/{grant_id}")
async def update_guest_scopes(grant_id: str, request: Request) -> dict[str, Any]:
    """HOST: narrow/widen a guest's scopes (e.g. enable call)."""
    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")
    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="storage unavailable")
    body = await request.json()
    from augmentum.connect.guest_grant_store import set_scopes
    ok = await set_scopes(conn, grant_id=grant_id, host_user_id=uid, scopes=body.get("scopes"))
    if not ok:
        raise HTTPException(status_code=404, detail="grant not found or revoked")
    return {"updated": True}


@router.post("/guests/{grant_id}/revoke")
async def revoke_guest(grant_id: str, request: Request) -> dict[str, Any]:
    """HOST: revoke a guest — the single kill-switch.

    Cascades: grant revoked → all the guest's sessions revoked → guest account
    deactivated → any IP-pinned door released. The saved surface goes dark.
    """
    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")
    uid = _user_id(request)
    conn = _get_conn(request)
    sm = _get_session_manager(request)
    if conn is None or sm is None:
        raise HTTPException(status_code=503, detail="service unavailable")
    from augmentum.connect.guest_grant_store import revoke as revoke_grant
    grant = await revoke_grant(conn, grant_id=grant_id, host_user_id=uid)
    if grant is None:
        raise HTTPException(status_code=404, detail="grant not found or already revoked")
    guest_user_id = grant["guest_user_id"]
    # Cascade — each step best-effort so one failure can't leave the grant live.
    try:
        await sm.revoke_all_sessions(guest_user_id)
    except Exception:
        log.warning("guest_revoke_sessions_failed", guest_user_id=guest_user_id, exc_info=True)
    try:
        await sm.update_user(guest_user_id, is_active=False)
    except Exception:
        log.warning("guest_revoke_deactivate_failed", guest_user_id=guest_user_id, exc_info=True)
    return {"revoked": True}


def _get_ip_from(request: Request) -> str:
    """Originating IP (Cf-Connecting-Ip through a tunnel, then XFF, then peer)."""
    cf = request.headers.get("cf-connecting-ip", "").strip()
    if cf:
        return cf
    xff = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
    return xff or (request.client.host if request.client else "")


async def _read_user_setting_bool(
    conn: Any, user_id: str, key: str, *, default: bool,
) -> bool:
    """Read a single bool setting for a user from user_settings.

    Returns ``default`` when the row is absent or unparseable. We
    don't fall back to ``app_settings`` here — the discoverability
    flags are per-user-only and a missing row genuinely means
    "user hasn't opted in", not "use the install default".
    """

    cur = await conn.execute(
        "SELECT value FROM user_settings WHERE user_id = ? AND key = ?",
        (user_id, key),
    )
    row = await cur.fetchone()
    await cur.close()
    if not row:
        return default
    raw = (row[0] or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off", ""):
        return False
    return default


async def _query_discoverable_same_instance_peers(
    conn: Any, caller_user_id: str,
) -> list[tuple[str, str]]:
    """Every same-instance user who is a reachable contact for the caller.

    Internal-directory model — a user is listed unless they:
      * are inactive (``users.is_active != 1``),
      * are the caller themselves,
      * turned Connect off (``connectEnabled = false``),
      * explicitly HID themselves (``connectDiscoverableSameInstance = false``), or
      * are a ``role='guest'`` account (guests are never in the directory), or
      * are a fabric MACHINE account rather than a person (see below).

    On machine accounts: ``SessionManager.get_or_create_fabric_peer_user``
    provisions a ``fabric:<node-id>`` user for every peer instance that
    dispatches to us, so peer-owned data has something to belong to at the
    trust boundary. Those rows are service accounts for a *machine*, not
    people you can call, and they were landing in the human directory --
    on this dogfood box that meant 7 of 11 listed "people" were fabric
    rows, five of them displaying the identical name "Fabric peer
    loopback". Filtered on BOTH role and id prefix on purpose: the role
    convention post-dates the first peer rows, so the oldest one on this
    box carries ``role='user'`` and a role-only filter would still leak it.
    A genuine cross-instance PERSON is not affected -- they arrive as
    ``user@instance`` through the fabric discovery path, not as a
    ``fabric:`` service account.

    So the default (no setting at all) is VISIBLE — everyone on the machine is a
    possible contact. Returns ``(user_id, display_name, username)`` tuples.

    Note on the JOIN: user_settings has a (user_id, key) composite PK so the two
    LEFT JOINs don't multiply rows; the COALESCE-on-NULL defaults (enabled→true,
    discoverable→true) encode "visible unless explicitly opted out".
    """

    # ``connectEnabled`` defaults to true in config.py — absent rows
    # (user never explicitly saved their settings) should still count
    # as "Connect on" because that's the new install default. The
    # COALESCE fallback to 'true' encodes that.
    #
    # ``connectDiscoverableSameInstance`` is opt-OUT: the COALESCE fallback to
    # 'true' makes an absent row count as VISIBLE (everyone on the machine is a
    # possible contact). A user is hidden only when they explicitly saved
    # 'false'/'0'/'no'/'off'. Invitees can set this at onboarding to stay private
    # to just their inviter (see auth_routes.public_claim_invite).
    #
    # Keys live under ``ui.<camelCase>`` because the UI POST handler
    # persists per-user settings with that prefix (see
    # ``store.set_user(uid, f"ui.{key}", coerced)`` in
    # config_routes.py::update_ui_settings).
    cur = await conn.execute(
        """
        SELECT u.id, COALESCE(NULLIF(u.display_name, ''), u.username) AS dn,
               u.username
          FROM users u
          LEFT JOIN user_settings se
            ON se.user_id = u.id AND se.key = 'ui.connectEnabled'
          LEFT JOIN user_settings sd
            ON sd.user_id = u.id AND sd.key = 'ui.connectDiscoverableSameInstance'
         WHERE u.is_active = 1
           AND u.id != ?
           AND LOWER(COALESCE(se.value, 'true')) IN ('1', 'true', 'yes', 'on')
           AND LOWER(COALESCE(sd.value, 'true')) NOT IN ('0', 'false', 'no', 'off')
           AND (u.role IS NULL OR u.role != 'guest')
           AND (u.role IS NULL OR u.role != 'peer')
           AND u.id NOT LIKE 'fabric:%'
        """,
        (caller_user_id,),
    )
    rows = await cur.fetchall()
    await cur.close()
    return [(r[0], r[1], r[2]) for r in rows]


def _handle_for(username: str) -> str:
    """Human ``@handle`` form for the directory/search surface."""
    return f"@{username}" if username else ""


async def _enrich_people(
    conn: Any, hub: ConnectHub, rows: list[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    """Turn (user_id, display_name, username) rows into directory entries.

    Adds the human ``handle``, profile ``status_message``/``status_emoji``,
    live ``online`` flag, and persisted ``last_seen_at`` for offline peers.
    """
    from augmentum.connect.presence_store import get_presence_for
    from augmentum.connect.profile_store import get_profiles_for

    ids = [r[0] for r in rows]
    profiles = await get_profiles_for(conn, ids)
    presence = await get_presence_for(conn, ids)
    people: list[dict[str, Any]] = []
    for target_uid, target_display, target_username in rows:
        prof = profiles.get(target_uid, {})
        online = hub.is_online(target_uid)
        people.append({
            "peer_did": local_did_for(target_uid),
            "user_id": target_uid,
            "username": target_username,
            "handle": _handle_for(target_username),
            "display_name": (target_display or target_uid),
            "status_message": prof.get("status_message", ""),
            "status_emoji": prof.get("status_emoji", ""),
            "online": online,
            "last_seen_at": "" if online else presence.get(target_uid, {}).get("last_seen_at", ""),
            "accepts_calls": True,      # Phase 1 — no per-modality opt-out yet
            "accepts_messages": True,
            "discovery_source": "same_instance",
        })
    return people


def _make_presence_sink(app):
    """Build the hub's persistent-presence hook bound to this app's DB.

    Reads the conn at call time (not capture time) so it stays correct if the
    backend reconnects. Best-effort: ``mark_presence`` failures are swallowed
    by the hub's ``_emit_presence_sink`` wrapper so the WS path never breaks.
    """
    async def _sink(user_id: str, online: bool) -> None:
        sm = getattr(getattr(app, "state", None), "state_manager", None)
        backend = getattr(sm, "backend", None)
        conn = getattr(backend, "conn", None) if isinstance(backend, SQLiteBackend) else None
        if conn is None:
            return
        from augmentum.connect.presence_store import mark_presence
        await mark_presence(conn, user_id=user_id, online=online)

    return _sink


def _get_hub(request: Request) -> ConnectHub:
    hub = getattr(request.app.state, "connect_hub", None)
    if hub is None:
        # Lazy init — the hub is process-local, has no startup
        # dependencies, and costs ~nothing when idle. Creating on
        # first hit keeps the wiring simple without forcing every
        # test fixture to bootstrap an empty hub.
        hub = ConnectHub()
        hub.set_presence_sink(_make_presence_sink(request.app))
        request.app.state.connect_hub = hub
        _wire_hub_to_fabric(request.app, connect_hub=hub)
    return hub


def _get_rate_limiter(app) -> WsRateLimiter:
    """Lazy-init the per-app rate limiter for the Connect WS surface.

    State is per-process, not per-connection, so reconnects don't
    reset the bucket. Created on first hit so test fixtures don't
    need to bootstrap one.
    """

    limiter = getattr(app.state, "connect_rate_limiter", None)
    if limiter is None:
        limiter = WsRateLimiter()
        app.state.connect_rate_limiter = limiter
    return limiter


# ── HTTP: messaging ──────────────────────────────────────────────


@router.get("/threads")
async def list_threads(
    request: Request,
    include_archived: bool = False,
    limit: int = 100,
) -> dict[str, Any]:
    """List all text threads the user owns (newest-first by tail)."""

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.message_store import list_threads_for_user

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")
    threads = await list_threads_for_user(
        conn, user_id=uid, include_archived=include_archived,
        limit=max(1, min(limit, 500)),
    )
    return {"threads": [t.to_dict() for t in threads]}


@router.get("/threads/{thread_id}/messages")
async def list_thread_messages(
    request: Request,
    thread_id: str,
    limit: int = 100,
    before: str | None = None,
    since: str | None = None,
) -> dict[str, Any]:
    """Newest-first message history for one thread.

    Two cursors:

    * ``before`` — pagination (sent_at of the oldest row already loaded).
    * ``since``  — catch-up after reconnect. Returns messages strictly
      newer than the cursor and, as a side effect, sends an
      ``EVENT_TEXT_DELIVERED`` back to senders for any of those rows
      that were inbound + undelivered (so their UI swaps "sent" for
      "delivered" once the recipient comes back online).
    """

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.message_store import (
        get_thread,
        list_messages_for_thread,
        stamp_delivered,
    )

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")
    thread = await get_thread(conn, thread_id=thread_id, user_id=uid)
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")
    msgs = await list_messages_for_thread(
        conn, thread_id=thread_id, user_id=uid,
        limit=max(1, min(limit, 500)),
        before_sent_at=before,
        after_sent_at=since,
    )

    # Catch-up side effect: any inbound message in this batch that
    # the recipient (i.e. this caller) hasn't acked yet gets its
    # delivered_at stamped, and we fire EVENT_TEXT_DELIVERED to the
    # original sender. This is what closes the loop when the
    # recipient was offline: the catch-up fetch itself is treated
    # as the delivery ack.
    if since and msgs:
        own_did = local_did_for(uid)
        # Group undelivered inbound rows by sender so each gets one
        # batched receipt rather than N pings.
        by_sender: dict[str, list[str]] = {}
        for m in msgs:
            if m.sender_did == own_did:
                continue  # outbound — already stamped on the sender side
            if m.delivered_at:
                continue
            await stamp_delivered(
                conn, message_id=m.message_id, user_id=m.user_id,
            )
            by_sender.setdefault(m.sender_did, []).append(m.message_id)

        if by_sender:
            hub = _get_hub(request)
            for sender_did, ids in by_sender.items():
                resolved = (
                    sender_did.split("@", 1)[0]
                    if "@" in sender_did
                    else ""
                )
                if not resolved:
                    continue
                try:
                    await hub.route_to_user(
                        target_user_id=resolved,
                        envelope=ConnectEnvelope(
                            kind="event",
                            verb=EVENT_TEXT_DELIVERED,
                            peer=own_did,
                            data={
                                "thread_id": thread_id,
                                "message_ids": ids,
                            },
                        ),
                    )
                except Exception as exc:
                    # Receipt fan-out is best-effort. The DB row
                    # already records the delivery; the sender will
                    # learn about it on their own next catch-up.
                    log.warning(
                        "connect_catchup_receipt_failed",
                        sender_did=sender_did,
                        count=len(ids), error=str(exc)[:160],
                    )

    # Reactions per message — one extra query batched by message_id list.
    # Returns {message_id: [{emoji, reactor_did, count}, ...]} grouped
    # so the UI can render a pill stack underneath each bubble.
    reactions_by_msg: dict[str, list[dict[str, Any]]] = {}
    if msgs:
        msg_ids = [m.message_id for m in msgs]
        placeholders = ",".join("?" * len(msg_ids))
        cur = await conn.execute(
            f"""SELECT message_id, emoji, reactor_did
                  FROM connect_message_reactions
                 WHERE user_id = ? AND message_id IN ({placeholders})
                 ORDER BY reacted_at""",
            (uid, *msg_ids),
        )
        # Build per-message {emoji: [reactor_did, ...]} so the UI gets
        # both the count and the reactors (lets it show "you + 2" or
        # disable a tap-toggle if you've already reacted).
        agg: dict[str, dict[str, list[str]]] = {}
        for row in await cur.fetchall():
            mid, emoji, reactor = row
            agg.setdefault(mid, {}).setdefault(emoji, []).append(reactor)
        for mid, by_emoji in agg.items():
            reactions_by_msg[mid] = [
                {"emoji": e, "reactor_dids": rs, "count": len(rs)}
                for e, rs in by_emoji.items()
            ]

    # Fabric attachment fetch fields per message — populated only for
    # cross-instance attachment-bearing messages. One batch SELECT
    # keeps the catch-up endpoint at a constant query count regardless
    # of how many messages were missed.
    fabric_attach_by_msg: dict[str, tuple[str | None, str | None]] = {}
    if msgs:
        msg_ids_with_attach = [
            m.message_id for m in msgs if m.attachment_ref
        ]
        if msg_ids_with_attach:
            placeholders = ",".join("?" * len(msg_ids_with_attach))
            try:
                cur = await conn.execute(
                    f"""SELECT message_id, attachment_fetch_url, attachment_fetch_token
                          FROM connect_messages
                         WHERE user_id = ? AND message_id IN ({placeholders})""",
                    (uid, *msg_ids_with_attach),
                )
                for mid, url, tok in await cur.fetchall():
                    fabric_attach_by_msg[mid] = (url, tok)
                await cur.close()
            except Exception:
                # Older DB without the columns — fabric-attach fetch
                # gracefully degrades to the local-route fetch.
                fabric_attach_by_msg = {}

    def _msg_with_reactions(m: Any) -> dict[str, Any]:
        d = m.to_dict()
        d["reactions"] = reactions_by_msg.get(m.message_id, [])
        url, tok = fabric_attach_by_msg.get(m.message_id, (None, None))
        if url and tok:
            d["attachment_fetch_url"] = url
            d["attachment_fetch_token"] = tok
        return d

    return {
        "thread": thread.to_dict(),
        "messages": [_msg_with_reactions(m) for m in msgs],
    }


@router.get("/threads/{thread_id}/messages/{message_id}/attachment")
async def get_message_attachment(
    request: Request, thread_id: str, message_id: str,
):
    """Serve the attachment bytes for a Connect message.

    Access model: any user who has a ``connect_messages`` row for
    ``message_id`` can fetch — that row IS the access grant. The
    sender owns the underlying ``uploads`` row, but the recipient
    can resolve through this endpoint without needing a copy in
    their own namespace. This keeps the blob refcount honest (one
    upload row, one blob ref) and means soft-deleting the message
    breaks attachment access on both sides automatically.

    Resolution path:
      1. Verify caller has ``(message_id, user_id=caller)`` row.
         404 on miss — same response regardless of "wrong message"
         vs "not a participant" so we don't leak existence.
      2. Read ``attachment_ref`` (it's an upload_id like ``ul_xxx``).
      3. Look up the ``uploads`` row by id (NOT user-scoped — the
         sender owns it).
      4. Resolve the blob via ``BlobStore.get`` and serve the file.

    Supports ``?download=1`` to force ``Content-Disposition:
    attachment`` instead of inline rendering (matches the files
    routes' download/preview split).
    """

    from fastapi.responses import FileResponse

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.message_store import get_message
    from augmentum.vfs.blobs import BlobStore

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")

    msg = await get_message(conn, message_id=message_id, user_id=uid)
    if msg is None or msg.thread_id != thread_id:
        # Conflating wrong-thread / wrong-message / not-a-participant
        # is intentional — distinguishing them would let a curious
        # client probe for message-id existence across users.
        raise HTTPException(status_code=404, detail="attachment not found")
    if not msg.attachment_ref:
        raise HTTPException(status_code=404, detail="no attachment on message")

    cur = await conn.execute(
        "SELECT blob_sha, filename, mime_type, mime_sniffed, size_bytes "
        "FROM uploads WHERE id = ?",
        (msg.attachment_ref,),
    )
    upload_row = await cur.fetchone()
    await cur.close()
    if upload_row is None:
        # Sender's upload row vanished (manual cleanup / sender's
        # account deletion). 404 to the recipient — there's nothing
        # to serve.
        raise HTTPException(status_code=404, detail="attachment expired")
    blob_sha, filename, claimed_mime, sniffed_mime, size_bytes = upload_row

    blobs = BlobStore(conn)
    blob = await blobs.get(blob_sha)
    if blob is None or not blob.get("real_path"):
        raise HTTPException(status_code=404, detail="attachment bytes missing")
    from pathlib import Path
    real_path = Path(blob["real_path"])
    if not real_path.exists():
        raise HTTPException(status_code=404, detail="attachment bytes missing")

    # Prefer the sniffed MIME (server-detected from magic bytes) over
    # the client-claimed type so we don't honour a lie.
    effective_mime = sniffed_mime or claimed_mime or "application/octet-stream"
    download = str(request.query_params.get("download") or "").lower() in (
        "1", "true", "yes",
    )
    disposition = "attachment" if download else "inline"

    # RFC 6266 Content-Disposition: emit both a fallback ASCII filename
    # (with CR/LF/quote/control chars stripped — sender-controlled, must
    # not break out of the header) and a percent-encoded UTF-8 variant
    # so clients can recover the original name.
    from urllib.parse import quote
    safe_name = (filename or "attachment")
    # Drop CR, LF, and other C0 control chars to prevent header injection.
    safe_name = "".join(ch for ch in safe_name if ord(ch) >= 0x20 and ch != "\x7f")
    ascii_fallback = safe_name.encode("ascii", "replace").decode("ascii")
    ascii_fallback = ascii_fallback.replace("\\", "_").replace('"', "_")
    utf8_quoted = quote(safe_name, safe="")
    content_disposition = (
        f"{disposition}; filename=\"{ascii_fallback}\"; "
        f"filename*=UTF-8''{utf8_quoted}"
    )

    return FileResponse(
        path=str(real_path),
        media_type=effective_mime,
        filename=safe_name or "attachment",
        headers={
            "Content-Disposition": content_disposition,
            # Long-cache: blob bytes are immutable by content-address.
            # The message_id in the path means a different message
            # never collides with another's cache entry.
            "Cache-Control": "private, max-age=86400, immutable",
        },
    )


@router.head("/threads/{thread_id}/messages/{message_id}/attachment")
async def head_message_attachment(
    request: Request, thread_id: str, message_id: str,
) -> Response:
    """Cheap metadata probe: filename + MIME + size without the body.

    Used by the catch-up render path — when a recipient pulls
    previously-missed messages via ``?since=`` they don't have the
    live event data the WS push would have carried, so this lets the
    bubble renderer pick the right widget (image, audio, file pill)
    without downloading the bytes.
    """

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.message_store import get_message

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")

    msg = await get_message(conn, message_id=message_id, user_id=uid)
    if msg is None or msg.thread_id != thread_id or not msg.attachment_ref:
        raise HTTPException(status_code=404, detail="attachment not found")

    cur = await conn.execute(
        "SELECT filename, mime_type, mime_sniffed, size_bytes "
        "FROM uploads WHERE id = ?",
        (msg.attachment_ref,),
    )
    row = await cur.fetchone()
    await cur.close()
    if row is None:
        raise HTTPException(status_code=404, detail="attachment expired")
    filename, claimed_mime, sniffed_mime, size_bytes = row
    mime = sniffed_mime or claimed_mime or "application/octet-stream"

    return Response(
        status_code=200,
        headers={
            "Content-Type": mime,
            "Content-Length": str(size_bytes or 0),
            "X-Attachment-Filename": filename or "attachment",
        },
    )


@router.get("/fabric/attachments/{ref}")
async def get_fabric_attachment(request: Request, ref: str):
    """Serve attachment bytes to a fabric peer's recipient browser.

    Access model: token-bearer. The recipient's instance got the
    token at fabric-message inbound time (signed by our identity);
    they pass it back here verbatim via ``?token=...``. We verify the
    token: signature, ref binding, expiry. No authentication on the
    HTTP request itself — the token IS the access grant.

    The local same-instance path (``/threads/.../attachment``) still
    handles same-box fetches and uses session auth, not tokens.
    """
    from fastapi.responses import FileResponse

    from augmentum.connect.fabric_transport import verify_attachment_token
    from augmentum.vfs.blobs import BlobStore

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    token = str(request.query_params.get("token") or "")
    if not token:
        raise HTTPException(status_code=400, detail="token required")

    identity = getattr(request.app.state, "fabric_identity", None)
    ok, err = verify_attachment_token(
        identity=identity, token=token, expected_ref=ref,
    )
    if not ok:
        # Don't leak which specific check failed — all token failures
        # collapse to 403 so a token-fishing attacker learns nothing
        # about format / expiry / ref-binding from the response.
        raise HTTPException(status_code=403, detail="token invalid")

    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")

    cur = await conn.execute(
        "SELECT blob_sha, filename, mime_type, mime_sniffed, size_bytes "
        "FROM uploads WHERE id = ?",
        (ref,),
    )
    upload_row = await cur.fetchone()
    await cur.close()
    if upload_row is None:
        raise HTTPException(status_code=404, detail="attachment not found")
    blob_sha, filename, claimed_mime, sniffed_mime, _size = upload_row

    blobs = BlobStore(conn)
    blob = await blobs.get(blob_sha)
    if blob is None or not blob.get("real_path"):
        raise HTTPException(status_code=404, detail="attachment bytes missing")
    from pathlib import Path
    real_path = Path(blob["real_path"])
    if not real_path.exists():
        raise HTTPException(status_code=404, detail="attachment bytes missing")

    effective_mime = sniffed_mime or claimed_mime or "application/octet-stream"
    download = str(request.query_params.get("download") or "").lower() in (
        "1", "true", "yes",
    )
    disposition = "attachment" if download else "inline"
    safe_name = (filename or "attachment").replace(chr(34), "")
    return FileResponse(
        path=str(real_path),
        media_type=effective_mime,
        filename=filename or "attachment",
        headers={
            "Content-Disposition": f'{disposition}; filename="{safe_name}"',
            # Tokens are short-lived; cache the bytes for the token's
            # remaining lifetime at most. Conservative private cache.
            "Cache-Control": "private, max-age=300",
        },
    )


@router.post("/threads/{thread_id}/send")
async def send_thread_message(
    request: Request, thread_id: str,
) -> dict[str, Any]:
    """HTTP fallback for sending a message into a thread.

    The WS path (``MSG_TEXT_SEND``) is the primary surface for live
    typing UIs; this HTTP route covers retry-from-background +
    server-rendered widgets that don't keep a WS open.

    Body: ``{"peer_did": str, "body": str, "format": "plain"|...,
    "message_id": str (optional), "reply_to": str (optional)}``.
    """

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.message_routing import handle_message_envelope
    from augmentum.connect.protocol import MSG_TEXT_SEND, ConnectEnvelope

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="json object required")
    peer_did = str(body.get("peer_did") or "")
    if not peer_did:
        raise HTTPException(status_code=400, detail="peer_did required")
    sender_did = local_did_for(uid)

    env_data = dict(body)
    env_data.pop("peer_did", None)
    if "thread_id" not in env_data and thread_id:
        env_data["thread_id"] = thread_id

    env = ConnectEnvelope(
        kind="msg",
        verb=MSG_TEXT_SEND,
        peer=peer_did,
        data=env_data,
    )
    notification_hub = _get_notification_hub(request.app)
    result = await handle_message_envelope(
        conn=conn,
        connect_hub=_get_hub(request),
        notification_hub=notification_hub,
        env=env,
        sender_user_id=uid,
        sender_did=sender_did,
        sender_role=getattr(request.scope.get("user"), "role", ""),
        fabric_coordinator=getattr(
            request.app.state, "fabric_coordinator", None,
        ),
        fabric_identity=getattr(
            request.app.state, "fabric_identity", None,
        ),
        our_attachment_base_url=_local_url_base(request),
    )
    if result.error_code:
        # 503 for "we can't reach the peer right now" conditions; the
        # request itself was well-formed. 400 for everything else.
        fabric_pending = {
            "fabric_routing_pending",  # legacy pre-Wedge-B; kept for shim
            "fabric_unavailable",      # fabric not enabled on this instance
            "fabric_peer_unknown",     # target hostname not paired
        }
        raise HTTPException(
            status_code=503 if result.error_code in fabric_pending else 400,
            detail={"code": result.error_code, "message": result.error_message},
        )
    return {
        "thread_id": result.thread_id,
        "message_id": result.message_id,
        "routed": result.routed,
        "notification_id": result.notification_id,
    }


@router.get("/contacts")
async def list_contacts_route(
    request: Request,
    include_blocked: bool = False,
    tag: str | None = None,
) -> dict[str, Any]:
    """All Connect contacts the user has accumulated."""

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.contact_store import list_contacts
    from augmentum.connect.contacts import display_name_for_did

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")
    rows = await list_contacts(
        conn, user_id=uid,
        include_blocked=include_blocked, tag=tag,
    )
    contacts = []
    for r in rows:
        d = r.to_dict()
        # Contacts created implicitly on first inbound message/call carry an
        # empty peer_display_name; resolve the canonical username from the DID
        # so the UI never falls back to rendering the raw usr_<hash> form.
        if not d.get("peer_display_name"):
            d["peer_display_name"] = await display_name_for_did(conn, d.get("peer_did") or "")
        contacts.append(d)
    return {"contacts": contacts}


@router.post("/contacts")
async def add_contact_route(request: Request) -> dict[str, Any]:
    """Body: ``{"peer_did": str, "peer_display_name": str (optional),
    "tags": list[str] (optional)}``."""

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.contact_store import add_contact

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="json object required")
    peer_did = str(body.get("peer_did") or "")
    if "@" not in peer_did:
        raise HTTPException(
            status_code=400,
            detail="peer_did must be in user@instance form",
        )
    contact = await add_contact(
        conn,
        user_id=uid,
        peer_did=peer_did,
        peer_display_name=str(body.get("peer_display_name") or ""),
        peer_avatar_url=str(body.get("peer_avatar_url") or ""),
        tags=list(body.get("tags") or []),
    )
    return contact.to_dict()


@router.delete("/contacts/{contact_id}")
async def remove_contact_route(
    request: Request, contact_id: str,
) -> dict[str, Any]:
    """Hard-delete the contact row. 404 when the contact doesn't exist
    for this user."""

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.contact_store import remove_contact

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")
    deleted = await remove_contact(
        conn, user_id=uid, contact_id=contact_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="contact not found")
    return {"ok": True}


@router.patch("/contacts/{contact_id}")
async def update_contact_route(
    request: Request, contact_id: str,
) -> dict[str, Any]:
    """Update blocked / tags. Body: ``{"blocked": bool, "tags": list[str]}``.

    Both fields are optional; only the present ones are updated.
    """

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.contact_store import (
        get_contact,
        set_blocked,
        set_tags,
    )

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="json object required")

    if "blocked" in body:
        if not await set_blocked(
            conn, user_id=uid, contact_id=contact_id,
            blocked=bool(body["blocked"]),
        ):
            raise HTTPException(status_code=404, detail="contact not found")
    if "tags" in body:
        tags = body["tags"]
        if not isinstance(tags, list):
            raise HTTPException(status_code=400, detail="tags must be a list")
        if not await set_tags(
            conn, user_id=uid, contact_id=contact_id,
            tags=[str(t) for t in tags],
        ):
            raise HTTPException(status_code=404, detail="contact not found")

    updated = await get_contact(
        conn, user_id=uid, contact_id=contact_id,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="contact not found")
    return updated.to_dict()


@router.post("/contacts/block")
async def block_contact_by_did(
    request: Request,
) -> dict[str, Any]:
    """Block or unblock a peer by DID.

    Body: ``{"peer_did": "...", "blocked": bool}``.

    Auto-creates a contact row when none exists, so the UI can block
    someone whose only relationship is an open thread (never explicitly
    added to contacts). Returns the updated contact row.
    """

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.contact_store import (
        ensure_contact,
        get_contact,
        set_blocked,
    )

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="json object required")
    peer_did = str(body.get("peer_did") or "").strip()
    if not peer_did:
        raise HTTPException(status_code=400, detail="peer_did required")
    blocked = bool(body.get("blocked"))

    contact = await ensure_contact(
        conn, user_id=uid, peer_did=peer_did,
        discovery_source="block_action",
    )
    if not await set_blocked(
        conn, user_id=uid, contact_id=contact.contact_id,
        blocked=blocked,
    ):
        raise HTTPException(status_code=404, detail="contact not found")
    updated = await get_contact(
        conn, user_id=uid, contact_id=contact.contact_id,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="contact not found")
    return updated.to_dict()


@router.get("/calls")
async def list_calls(
    request: Request,
    limit: int = 100,
    before: str | None = None,
    state: str | None = None,
) -> dict[str, Any]:
    """Most-recent-first call history for the user.

    ``state`` filters to one lifecycle state — useful for the
    missed-calls inbox (``state=missed``). ``before`` is a cursor
    (initiated_at of the oldest row already loaded).
    """

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.call_store import list_calls_for_user
    from augmentum.connect.contact_store import list_contacts
    from augmentum.connect.contacts import display_name_for_did

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")
    rows = await list_calls_for_user(
        conn, user_id=uid,
        limit=max(1, min(limit, 500)),
        before=before,
        state_filter=state,
    )
    # Call rows store only DIDs; resolve the peer's human name so the history
    # list renders a username instead of usr_<hash>@instance. Same-instance
    # peers resolve from the users table (display_name_for_did); cross-instance
    # FABRIC peers have no local user row, so we fall back to the name cached on
    # their contact row (populated by the fabric inbound dispatcher).
    contact_names = {
        c.peer_did: c.peer_display_name
        # include_stale: a call to someone since deleted (or a guest since
        # revoked) still needs their name in history. Live lists drop those
        # rows; the archive must not lose who the call was WITH.
        for c in await list_contacts(
            conn, user_id=uid, include_blocked=True, include_stale=True,
        )
        if c.peer_display_name
    }
    name_cache: dict[str, str] = {}
    calls = []
    for r in rows:
        d = r.to_dict()
        peer = d.get("peer_did") or ""
        if peer not in name_cache:
            resolved = await display_name_for_did(conn, peer)
            # display_name_for_did returns the bare local-part for a fabric DID
            # (no local user); prefer the contact-cached name in that case.
            if peer in contact_names and resolved == peer.rpartition("@")[0]:
                resolved = contact_names[peer]
            name_cache[peer] = resolved
        d["peer_display_name"] = name_cache[peer]
        calls.append(d)
    return {"calls": calls}


@router.get("/calls/{call_id}")
async def get_call_detail(
    request: Request, call_id: str,
) -> dict[str, Any]:
    """Single-call detail + the event timeline.

    Returns 404 if the call isn't on this user's scope (also the case
    for a call_id that doesn't exist at all — we don't distinguish
    "wrong user" from "wrong id" to avoid the cross-user existence
    oracle).
    """

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.call_store import get_call, list_events_for_call

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")
    call = await get_call(conn, call_id=call_id, user_id=uid)
    if call is None:
        raise HTTPException(status_code=404, detail="call not found")
    events = await list_events_for_call(
        conn, call_id=call_id, user_id=uid,
    )
    return {
        "call": call.to_dict(),
        "events": [e.to_dict() for e in events],
    }


@router.get("/calls/{call_id}/livekit_token")
async def get_livekit_token(
    request: Request, call_id: str,
) -> dict[str, Any]:
    """Mint a LiveKit room access JWT for an active call.

    Used by the LiveKit media plane (see
    ``docs/superpowers/specs/2026-06-06-livekit-media-plane-design.md``).
    The client calls this after ``EVENT_ACCEPT`` arrives for calls
    where ``media == "livekit"``, then plugs the response into
    ``room.connect(url, token)`` via the livekit-client SDK.

    Response shape::

        {
            "token":      "<jwt>",
            "url":        "wss://<host>:7880",
            "room":       "call_<call_id>",
            "expires_at": <unix-ts>,
        }

    Returns 404 when the call isn't on this user's scope (same
    indistinguishable-from-not-found policy as ``get_call_detail`` so
    the endpoint isn't a cross-user existence oracle). Returns 503
    when LiveKit is unreachable — the caller should fall back to the
    P2P plane (the invite-time decision tree would normally have
    already routed to P2P in this case, but the check is repeated
    here as a safety belt for races where LiveKit dropped between
    invite and accept).
    """

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.call_store import get_call
    from augmentum.connect.livekit_tokens import (
        livekit_reachable,
        mint_call_token,
    )

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")
    call = await get_call(conn, call_id=call_id, user_id=uid)
    if call is None:
        raise HTTPException(status_code=404, detail="call not found")

    if not await livekit_reachable():
        raise HTTPException(status_code=503, detail="livekit unreachable")

    user_did = _user_did_for(uid)
    bundle = mint_call_token(
        call_id=call_id,
        user_did=user_did,
    )
    return {
        "token": bundle.token,
        "url": bundle.url,
        "room": bundle.room,
        "expires_at": bundle.expires_at,
    }


@router.post("/calls/{call_id}/rate")
async def rate_call(
    request: Request, call_id: str,
) -> dict[str, Any]:
    """Post-call quality rating. Body: ``{"rating": 1|0|-1, "notes": str}``."""

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.call_store import set_quality_rating

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="json object required")
    rating = body.get("rating")
    if rating not in (1, 0, -1, None):
        raise HTTPException(
            status_code=400,
            detail="rating must be 1, 0, -1, or null",
        )
    notes = str(body.get("notes") or "")
    updated = await set_quality_rating(
        conn, call_id=call_id, user_id=uid,
        rating=rating, notes=notes,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="call not found")
    return {"ok": True}


@router.delete("/threads/{thread_id}/messages")
async def clear_thread_messages_route(
    request: Request, thread_id: str,
) -> dict[str, Any]:
    """Hard-delete every message in ``thread_id`` for the caller. Peer
    instance is unaffected — this is a local-view wipe matching
    iMessage / WhatsApp "Clear Chat History" semantics. The thread row
    is preserved so the conversation can resume; only the message rows
    + tail-snapshot fields are reset.
    """

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.message_store import clear_thread_for_user

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")
    removed = await clear_thread_for_user(
        conn, user_id=uid, thread_id=thread_id,
    )
    return {"ok": True, "removed": removed}


@router.post("/threads/{thread_id}/mark-read")
async def mark_thread_read_route(
    request: Request, thread_id: str,
) -> dict[str, Any]:
    """Clear the user's unread counter for one thread."""

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.message_store import mark_thread_read

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")
    body: dict[str, Any] = {}
    try:
        parsed = await request.json()
        if isinstance(parsed, dict):
            body = parsed
    except Exception:
        # Empty bodies are fine — fall through with the default of
        # "mark every unread message as read".
        body = {}
    last = str(body.get("last_read_message_id") or "")
    marked = await mark_thread_read(
        conn, thread_id=thread_id, user_id=uid,
        last_read_message_id=last,
    )
    return {"marked": marked}


@router.patch("/threads/{thread_id}")
async def update_thread_flags_route(
    request: Request, thread_id: str,
) -> dict[str, Any]:
    """Persist per-thread preferences (pin / mute / archive) for the caller.

    Body: any subset of ``{"pinned": bool, "muted": bool, "archived": bool}``.
    Each provided flag is applied via ``set_thread_flag`` (which is user-scoped
    by (thread_id, user_id)); omitted flags are left untouched. The peer's copy
    is unaffected — these are local-view preferences, same isolation model as
    unread/clear-history.
    """

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.message_store import set_thread_flag

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")

    try:
        body = await request.json()
    except Exception:
        body = None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="expected a JSON object")

    _ALLOWED = ("pinned", "muted", "archived")
    updates = {k: bool(body[k]) for k in _ALLOWED if k in body and body[k] is not None}
    if not updates:
        raise HTTPException(
            status_code=400,
            detail="provide at least one of: pinned, muted, archived",
        )

    # Applied in a stable order; a missing thread row yields existed=False so the
    # caller can tell "no such thread for me" from a silent no-op.
    existed = False
    for flag, value in updates.items():
        if await set_thread_flag(
            conn, thread_id=thread_id, user_id=uid, flag=flag, value=value,
        ):
            existed = True
    if not existed:
        raise HTTPException(status_code=404, detail="thread not found")
    return {"ok": True, "thread_id": thread_id, "updated": updates}


@router.post("/threads/{thread_id}/messages/{message_id}/react")
async def react_to_message(
    request: Request, thread_id: str, message_id: str,
) -> dict[str, Any]:
    """Add or remove an emoji reaction on a message.

    Body: ``{"peer_did": "...", "emoji": "👍", "action": "add"|"remove"}``.

    Both perspectives' rows are upserted/deleted; an
    ``EVENT_TEXT_REACT`` is routed to the peer if they're online.
    Returns the routing count and the message_id for client confirmation.

    HTTP path exists for parity with send/mark-read so the UI can
    react via REST when the signaling WS is dropped — the reaction
    still lands on both sides and the peer sees it on next reconnect.
    """

    if not _connect_enabled():
        raise HTTPException(status_code=503, detail="connect disabled")

    from augmentum.connect.contacts import local_did_for
    from augmentum.connect.message_routing import _handle_react
    from augmentum.connect.protocol import MSG_TEXT_REACT

    uid = _user_id(request)
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="persistence unavailable")

    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="json body required")
    peer_did = str(body.get("peer_did") or "")
    emoji = str(body.get("emoji") or "")
    action = str(body.get("action") or "add")
    if not peer_did:
        raise HTTPException(status_code=400, detail="peer_did required")
    if not emoji:
        raise HTTPException(status_code=400, detail="emoji required")

    sender_did = local_did_for(uid)
    target_user_id = peer_did.split("@", 1)[0]

    # Guest ACL gate — this REST path calls _handle_react directly and so skips
    # the gate baked into handle_message_envelope; without it a guest could
    # react in a thread with a non-host peer. Mirror the send/WS enforcement.
    from augmentum.connect.guest_grant_store import guest_scope_blocked
    if await guest_scope_blocked(
        conn, sender_user_id=uid,
        sender_role=getattr(request.scope.get("user"), "role", ""),
        target_user_id=target_user_id,
    ):
        raise HTTPException(status_code=403, detail="guest_scope_violation")

    env = ConnectEnvelope(
        kind="msg",
        verb=MSG_TEXT_REACT,
        peer=peer_did,
        data={
            "thread_id": thread_id,
            "message_id": message_id,
            "emoji": emoji,
            "action": action,
        },
    )
    result = await _handle_react(
        conn=conn,
        connect_hub=_get_hub(request),
        env=env,
        sender_user_id=uid,
        sender_did=sender_did,
        # Same-instance for Phase 1 — fabric peers deferred.
        target_user_id=target_user_id,
    )
    if result.error_code:
        raise HTTPException(status_code=400, detail=result.error_code)
    return {
        "message_id": message_id,
        "emoji": emoji,
        "action": action,
        "routed": result.routed,
    }


# ── WebSocket ────────────────────────────────────────────────────


@router.websocket("/signaling")
async def signaling_ws(websocket: WebSocket) -> None:
    """Connect signaling channel.

    Lifecycle:
        accept → welcome (with TURN creds) → register with hub →
        receive loop → detach on close.

    Stable contract today; routing of invite/offer/answer/ICE/hangup
    to remote peers lands in the next iteration. The welcome envelope
    + ping/pong is enough for the UI to wire up its WS layer and
    confirm end-to-end auth + TURN config.
    """

    if not _connect_enabled():
        await websocket.close(code=1008, reason="connect disabled")
        return

    user = websocket.scope.get("user")
    if user is None:
        await websocket.close(code=1008, reason="auth required")
        return

    await websocket.accept()

    hub = getattr(websocket.app.state, "connect_hub", None)
    if hub is None:
        hub = ConnectHub()
        websocket.app.state.connect_hub = hub
        _wire_hub_to_fabric(websocket.app, connect_hub=hub)

    user_did = _user_did_for(user.id)
    att = await hub.attach(ws=websocket, user_id=user.id, user_did=user_did)
    # Per-connection party_id (Matrix MSC2746 pattern). The same user
    # may have multiple devices online; party_id lets routing
    # distinguish them when SELECT_ANSWER fires.
    party_id = new_party_id()

    # Welcome envelope: identify the peer to itself + ship TURN creds
    # so the UI doesn't need a separate HTTP fetch before opening a
    # peer connection.
    try:
        creds = mint_ephemeral(user.id)
        welcome = ConnectEnvelope(
            kind="event",
            verb=EVENT_WELCOME,
            data={
                "user_did": user_did,
                "party_id": party_id,
                "turn": {
                    **creds.as_ice_server(_turn_url()),
                    "expires_at": creds.expires_at,
                },
                "server_time": int(time.time()),
            },
        )
        await websocket.send_text(serialise_envelope(welcome))
    except Exception as exc:
        log.warning(
            "connect_welcome_send_failed",
            connection_id=att.connection_id,
            user_id=user.id,
            error=str(exc)[:160],
        )

    try:
        while True:
            raw = await websocket.receive_text()
            # Frame-size cap. Sized to fit large SDPs (5-20KB) with
            # headroom for batched ICE arrays. Beyond this is almost
            # certainly malicious or buggy — close with policy-
            # violation so misbehaving clients back off rather than
            # retry. Mirrors the 64KB ceiling sendBeacon uses
            # elsewhere in Augmentum.
            if len(raw) > MAX_ENVELOPE_BYTES:
                log.warning(
                    "connect_envelope_oversized",
                    connection_id=att.connection_id,
                    user_id=user.id,
                    bytes=len(raw),
                )
                await websocket.close(code=1009, reason="envelope too large")
                break
            env = deserialise_envelope(raw)
            if env is None:
                # Tolerant of garbage — drop silently. Clients can
                # roll out new verbs ahead of server support without
                # tearing down healthy connections.
                continue
            # Per-envelope isolation. Handling one envelope can fail for
            # reasons that say nothing about the health of the socket —
            # most commonly a transient ``database is locked`` when a
            # background job (media library scan, migration) is holding
            # SQLite. Letting that escape to the outer handler exits the
            # receive loop and detaches the user in ``finally``, taking
            # them OFFLINE mid-session: subsequent invites have nowhere
            # to route and the peer's phone never rings, with no error
            # shown on either end. The transport is still fine here, so
            # log and keep serving; if the socket really is dead the next
            # ``receive_text`` raises WebSocketDisconnect and we exit
            # through the normal path.
            try:
                await _handle_inbound(
                    websocket,
                    hub=hub,
                    env=env,
                    user_did=user_did,
                    user_id=user.id,
                    user_role=getattr(user, "role", ""),
                    party_id=party_id,
                    connection_id=att.connection_id,
                )
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                log.warning(
                    "connect_envelope_handler_failed",
                    connection_id=att.connection_id,
                    user_id=user.id,
                    verb=getattr(env, "verb", ""),
                    error=str(exc)[:160],
                )
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.warning(
            "connect_signaling_ws_error",
            connection_id=att.connection_id,
            user_id=user.id,
            error=str(exc)[:160],
        )
    finally:
        await hub.detach(att.connection_id)


async def _handle_inbound(
    websocket: WebSocket, *,
    hub: ConnectHub,
    env: ConnectEnvelope,
    user_did: str,
    user_id: str,
    party_id: str,
    connection_id: str = "",
    user_role: str = "",
) -> None:
    """Dispatch one inbound envelope.

    Pings get the fast-path pong. Every call-related verb goes
    through ``handle_signaling_envelope`` which does peer routing,
    persistence, and (for invite) notification publishing.
    """

    if env.verb == MSG_PING:
        await websocket.send_text(serialise_envelope(ConnectEnvelope(
            kind="event",
            verb=EVENT_PONG,
            corr_id=env.corr_id,
            data={"server_time": int(time.time())},
        )))
        return

    # Rate-limit every non-ping verb. The HTTP middleware skips WS,
    # so without this check a misbehaving client can flood the hub.
    limiter = _get_rate_limiter(websocket.app)
    allowed, category, retry_after = limiter.check(
        user_id=user_id, verb=env.verb,
    )
    if not allowed:
        log.warning(
            "connect_ws_rate_limited",
            user_id=user_id, verb=env.verb,
            category=category, retry_after=retry_after,
        )
        await websocket.send_text(serialise_envelope(ConnectEnvelope(
            kind="event",
            verb=EVENT_ERROR,
            corr_id=env.corr_id,
            data={
                "code": "rate_limited",
                "message": (
                    f"too many {category} requests "
                    f"(retry in {retry_after}s)"
                ),
                "ref_verb": env.verb,
                "retry_after": retry_after,
            },
        )))
        return

    conn = _get_conn(websocket)
    if conn is None:
        await websocket.send_text(serialise_envelope(ConnectEnvelope(
            kind="event",
            verb=EVENT_ERROR,
            corr_id=env.corr_id,
            data={
                "code": "no_persistence",
                "message": "signaling requires SQLite backend",
                "ref_verb": env.verb,
            },
        )))
        return

    notification_hub = _get_notification_hub(websocket.app)

    # Text-messaging verbs route through a separate dispatcher — they
    # share the WS transport with call signaling but their persistence
    # path is entirely different (connect_threads / connect_messages).
    if env.verb in (
        MSG_TEXT_SEND, MSG_TEXT_READ, MSG_TEXT_DELETE, MSG_TEXT_EDIT,
        MSG_TYPING_START, MSG_TYPING_STOP, MSG_TEXT_REACT,
        MSG_TEXT_DELIVERED,
    ):
        text_result = await handle_message_envelope(
            conn=conn,
            connect_hub=hub,
            notification_hub=notification_hub,
            env=env,
            sender_user_id=user_id,
            sender_did=user_did,
            sender_role=user_role,
            fabric_coordinator=getattr(
                websocket.app.state, "fabric_coordinator", None,
            ),
            fabric_identity=getattr(
                websocket.app.state, "fabric_identity", None,
            ),
            our_attachment_base_url=_local_url_base(websocket),
        )
        if text_result.error_code:
            await websocket.send_text(serialise_envelope(ConnectEnvelope(
                kind="event",
                verb=EVENT_ERROR,
                corr_id=env.corr_id,
                data={
                    "code": text_result.error_code,
                    "message": text_result.error_message,
                    "ref_verb": env.verb,
                },
            )))
            return
        ack_data: dict[str, Any] = {"routed": text_result.routed}
        if text_result.thread_id:
            ack_data["thread_id"] = text_result.thread_id
        if text_result.message_id:
            ack_data["message_id"] = text_result.message_id
        if text_result.notification_id:
            ack_data["notification_id"] = text_result.notification_id
        await websocket.send_text(serialise_envelope(ConnectEnvelope(
            kind="event",
            verb="routed",
            corr_id=env.corr_id,
            data=ack_data,
        )))
        return

    result = await handle_signaling_envelope(
        conn=conn,
        connect_hub=hub,
        notification_hub=notification_hub,
        env=env,
        sender_user_id=user_id,
        sender_did=user_did,
        sender_party_id=party_id,
        sender_role=user_role,
        fabric_coordinator=getattr(
            websocket.app.state, "fabric_coordinator", None,
        ),
        sender_connection_id=connection_id,
    )

    if result.error_code:
        await websocket.send_text(serialise_envelope(ConnectEnvelope(
            kind="event",
            verb=EVENT_ERROR,
            corr_id=env.corr_id,
            data={
                "code": result.error_code,
                "message": result.error_message,
                "ref_verb": env.verb,
            },
        )))
        return

    # Success — tell the sender we routed. The routed_count tells the
    # UI whether the peer was online at the time (zero = the peer is
    # offline; the recipient still has the notification on next read).
    ack_data: dict[str, Any] = {"routed": result.routed}
    if result.call_id:
        ack_data["call_id"] = result.call_id
    if result.notification_id:
        ack_data["notification_id"] = result.notification_id
    await websocket.send_text(serialise_envelope(ConnectEnvelope(
        kind="event",
        verb="routed",
        corr_id=env.corr_id,
        data=ack_data,
    )))
