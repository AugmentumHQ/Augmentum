"""Guest comms portal — the invited-guest experience.

Self-contained surface (kept apart from the shared account-claim flow):

  POST /api/portal/register/{token}    public: invitee registers from an
                                       external_guest invite -> PENDING.
  GET  /api/portal/me                  guest: am I confirmed + reachable?
  GET  /api/portal/pending             admin: registrations awaiting me.
  POST /api/portal/registrations/{id}/confirm   admin: final step ->
                                       allowlist IP + mint guest grant.
  POST /api/portal/registrations/{id}/deny      admin: reject.
  GET  /api/invite/{token}/qr.png      public: QR of the join link (scan
                                       face-to-face, or tap the link).

The gate (pending -> admin confirm -> IP allowlist) lives in
``connect/guest_portal.py``; these routes are thin orchestration over it
plus account creation + the existing guest-grant ACL.
"""

from __future__ import annotations

import io

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from augmentum.connect import guest_portal as gp
from augmentum.connect.rate_limit import KeyedRateLimiter
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["portal"])

_USERNAME_OK = __import__("re").compile(r"^[a-zA-Z0-9_]{3,32}$")

# The portal surface is fully public (register/gateway/env are reachable over
# the ephemeral tunnel with no session). Throttle each door: register bounds
# guest-account creation; gateway/env bound the per-request crypto work
# (AES-GCM + Ed25519) so an anonymous caller can't turn /env into a CPU-DoS.
_REGISTER_LIMITER = KeyedRateLimiter(limit=10, window_s=60.0)
_GATEWAY_LIMITER = KeyedRateLimiter(limit=60, window_s=60.0)
_ENV_LIMITER = KeyedRateLimiter(limit=120, window_s=60.0)


def _rl_key(request: Request) -> str:
    """Rate-limit key: the tunnel edge sets ``Cf-Connecting-Ip``; otherwise the
    real socket peer. Deliberately NOT the raw ``X-Forwarded-For`` — a guest can
    set that header freely and would otherwise rotate the key to skip the limit.
    """
    cf = request.headers.get("cf-connecting-ip", "").strip()
    if cf:
        return cf
    return request.client.host if request.client else "unknown"


def _portal_too_many(
    request: Request, limiter: KeyedRateLimiter, *, extra: str = "",
) -> JSONResponse | None:
    """Return a 429 when the caller is over ``limiter``, else None. ``extra``
    sub-buckets the key (e.g. by device_id for /env)."""
    key = _rl_key(request)
    if extra:
        key = f"{key}|{extra}"
    allowed, retry = limiter.check(key)
    if allowed:
        return None
    return JSONResponse(
        {"error": "Too many requests. Please slow down."},
        status_code=429, headers={"Retry-After": str(retry)},
    )


def _conn(request: Request):
    sm = getattr(request.app.state, "state_manager", None)
    return getattr(getattr(sm, "backend", None), "conn", None) if sm else None


def _sm(request: Request):
    return getattr(request.app.state, "session_manager", None)


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else ""


def _user(request: Request):
    return request.scope.get("user")


# ── public: register from an invite ──────────────────────────────────


@router.post("/api/portal/register/{token}")
async def portal_register(token: str, request: Request):
    """Register from an external_guest invite. Creates the guest account +
    a PENDING registration the inviting host must confirm. Does NOT log the
    guest in — they return once the admin confirms (and their IP is
    allowlisted)."""
    if (limited := _portal_too_many(request, _REGISTER_LIMITER)) is not None:
        return limited
    sm = _sm(request)
    conn = _conn(request)
    if not sm or conn is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)

    body = await request.json()
    username = str(body.get("username", "") or "")
    password = str(body.get("password", "") or "")
    display_name = str(body.get("display_name", "") or "").strip()
    if not _USERNAME_OK.match(username):
        return JSONResponse({"error": "Pick a username: 3–32 letters, numbers or _"}, status_code=400)
    if len(password) < 8:
        return JSONResponse({"error": "Choose a password of at least 8 characters"}, status_code=400)

    from augmentum.auth.invite_store import consume_invite, mark_claimed, preview_invite
    preview = await preview_invite(conn, token)
    # invite_status() emits active|expired|used|revoked — never "valid". The old
    # check compared against "valid", so this 410'd for EVERY register attempt
    # (portal onboarding was silently dead). Same class bug as the portal.js
    # frontend check; fixed in both places. consume_invite() below is still the
    # authoritative atomic gate — this is just the early, friendly rejection.
    if preview is None or preview.get("status") != "active":
        return JSONResponse({"error": "This invite isn’t valid anymore — ask for a fresh one."}, status_code=410)
    if await sm.get_user_by_username(username):
        return JSONResponse({"error": "That username is taken — try another."}, status_code=409)

    invite = await consume_invite(conn, token)
    if invite is None:
        return JSONResponse({"error": "This invite isn’t valid anymore — ask for a fresh one."}, status_code=410)

    try:
        guest = await sm.create_user(
            username, password, role="guest",
            display_name=display_name or username, email=invite.get("invitee_email", ""),
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    await mark_claimed(conn, token_hash=invite["token_hash"], claimed_user_id=guest.id,
                       claimed_ip=_client_ip(request))
    # Scopes the inviter granted ride on the invite (handle_hint reuse) or
    # default to text+call. Recorded on the registration for the host to see.
    scopes = str(body.get("scopes") or invite.get("handle_hint") or "text,call")
    await gp.register_pending(
        conn, inviter_user_id=invite["inviter_user_id"], guest_user_id=guest.id,
        display_name=display_name or username, requested_ip=_client_ip(request),
        scopes=scopes, invite_token_hash=invite["token_hash"],
        # The guest's web-device identity (so confirm can trust the device,
        # enabling IP-independent reconnection from any network).
        device_id=str(body.get("device_id", "") or ""),
        device_public_key=str(body.get("device_public_key", "") or ""),
    )
    log.info("portal_guest_registered_pending", guest=guest.id, inviter=invite["inviter_user_id"])
    # Do NOT close the public door here. Registration is only step one — the
    # guest still has to be APPROVED and then SIGN IN, both of which need the
    # door open (for a cloudflared-tier guest it's their only transport). Closing
    # it on register/fully-used stranded them ("wrong password" on a dead
    # origin). The door closes on deny / revoke / TTL-reap instead. Standing
    # tiers (funnel/tailnet/lan) never had a door to close.
    return JSONResponse({
        "status": "pending",
        "display_name": display_name or username,
        "message": "You're registered! Your host just needs to confirm you — "
                   "you'll be able to message and call them once they do.",
    }, status_code=201)


@router.get("/api/portal/me")
async def portal_me(request: Request):
    """Guest self-check: confirmed by the host AND reachable from this IP?"""
    user = _user(request)
    conn = _conn(request)
    if user is None:
        return JSONResponse({"error": "Not signed in"}, status_code=401)
    if conn is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)
    confirmed = await gp.is_confirmed(conn, guest_user_id=user.id)
    # Device-trust model: a confirmed guest is reachable from ANY network
    # (their session is device-bound, not IP-bound). IP is a logged signal,
    # not a gate. Both failover URLs are returned so the portal can prefer
    # the fast LAN address when near the host and the remote one otherwise.
    from augmentum.proxy.mobile_pair_routes import (
        _configured_remote_url,
        _request_root_url,
    )
    server_url = _request_root_url(request)
    remote_url = _configured_remote_url(request)
    if remote_url == server_url:
        remote_url = ""
    return JSONResponse({
        "confirmed": confirmed,
        "ready": confirmed,
        "state": "ready" if confirmed else "waiting",
        "server_url": server_url,   # this network (fast/LAN)
        "remote_url": remote_url,   # works from anywhere
    })


# ── admin: review + confirm ──────────────────────────────────────────


@router.get("/api/portal/pending")
async def portal_pending(request: Request):
    """Registrations awaiting the signed-in host's confirmation."""
    user = _user(request)
    conn = _conn(request)
    if user is None:
        return JSONResponse({"error": "Not signed in"}, status_code=401)
    if conn is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)
    pend = await gp.list_pending(conn, inviter_user_id=user.id)
    return JSONResponse({"pending": [
        {"registration_id": p.registration_id, "display_name": p.display_name,
         "requested_ip": p.requested_ip, "scopes": p.scopes,
         "guest_user_id": p.guest_user_id}
        for p in pend
    ]})


@router.post("/api/portal/registrations/{registration_id}/confirm")
async def portal_confirm(registration_id: str, request: Request):
    """The host's final step: allowlist the guest's IP, mint the guest grant
    (so text/call scopes apply), and let them in."""
    user = _user(request)
    conn = _conn(request)
    if user is None:
        return JSONResponse({"error": "Not signed in"}, status_code=401)
    if conn is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        body = {}
    extra_ips = body.get("extra_ips") if isinstance(body.get("extra_ips"), list) else []

    try:
        reg = await gp.confirm(conn, registration_id=registration_id,
                               admin_user_id=user.id, extra_ips=extra_ips)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    # Mint the guest grant so the existing ACL (who they may reach) applies.
    try:
        from augmentum.connect.contacts import local_did_for
        from augmentum.connect.guest_grant_store import create_grant
        await create_grant(
            conn, host_user_id=user.id, host_did=local_did_for(user.id),
            guest_user_id=reg.guest_user_id, guest_did=local_did_for(reg.guest_user_id),
            scopes=reg.scopes,
        )
    except Exception:
        log.warning("portal_confirm_grant_failed", exc_info=True)

    # Register the guest's web device as TRUSTED (reuses the Android mobile-
    # device model). This is what makes the guest's later session
    # device-bound + IP-independent — they reconnect from any network, and
    # the host can revoke this one device. Best-effort: the grant above is
    # the functional gate; the device record adds reconnection + revoke.
    if reg.device_id:
        try:
            from types import SimpleNamespace

            from augmentum.auth.mobile_pairing import TrustedMobileDeviceStore
            sm = _sm(request)
            await TrustedMobileDeviceStore(sm._db).upsert_from_pair(SimpleNamespace(
                user_id=reg.guest_user_id, device_id=reg.device_id,
                label=f"{reg.display_name}'s line", platform="web", app_version="",
                public_key=reg.device_public_key, key_alg="ed25519",
                scopes=[s for s in reg.scopes.split(",") if s], capabilities=[],
            ))
        except Exception:
            log.warning("portal_confirm_device_register_failed", exc_info=True)

    # Do NOT close the door on confirm. The guest still has to SIGN IN after
    # approval, and for a cloudflared-tier guest the tunnel is their only
    # transport — closing it here stranded them. The door closes on deny /
    # revoke / TTL-reap. (Standing funnel/tailnet/lan tiers have no door anyway.)
    return JSONResponse({"status": "confirmed", "guest_user_id": reg.guest_user_id})


@router.post("/api/portal/registrations/{registration_id}/deny")
async def portal_deny(registration_id: str, request: Request):
    user = _user(request)
    conn = _conn(request)
    if user is None:
        return JSONResponse({"error": "Not signed in"}, status_code=401)
    if conn is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)
    # Release BEFORE the deny flips status (the hash lookup is status-agnostic,
    # but keeping the order deterministic makes the log read correctly).
    await _release_registration_tunnel(conn, registration_id)
    ok = await gp.deny(conn, registration_id=registration_id, admin_user_id=user.id)
    return JSONResponse({"denied": ok})


async def _release_registration_tunnel(conn, registration_id: str) -> None:
    """Close the ephemeral tunnel held by a registration's invite (best-effort)."""
    try:
        from augmentum.auth.invite_store import tunnel_ref_for_hash
        from augmentum.proxy.auth_routes import release_invite_reach_by_ref
        token_hash = await gp.invite_hash_for_registration(conn, registration_id=registration_id)
        if token_hash:
            await release_invite_reach_by_ref(tunnel_ref_for_hash(token_hash))
    except Exception:
        log.warning("portal_tunnel_release_failed", exc_info=True)


# ── guest gateway: signed seal-key bundle + enveloped dispatch ────────
#
# Trust chain (2026-07-16 spec): the invite QR pins the instance Ed25519
# identity → /gateway hands out the X25519 seal key SIGNED by that identity →
# every guest request after claim is an envelope sealed to the seal key and
# signed by the guest's device key (bound to the account at host confirm).
# The envelope IS the auth — the "browser-side VPN authenticated through the
# account they create".


@router.get("/api/portal/gateway")
async def portal_gateway(request: Request):
    """The instance's PUBLIC gateway bundle. The portal verifies ``sig``
    against the identity key pinned from the invite QR before sealing
    anything to ``seal_pub``. No secrets; public by design."""
    if (limited := _portal_too_many(request, _GATEWAY_LIMITER)) is not None:
        return limited
    store = getattr(request.app.state, "settings_store", None)
    if store is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)
    try:
        from augmentum.connect.guest_gateway import get_gateway_keys
        keys = await get_gateway_keys(store)
        return JSONResponse(keys.bundle())
    except Exception:
        log.warning("portal_gateway_bundle_failed", exc_info=True)
        return JSONResponse({"error": "gateway unavailable"}, status_code=503)


@router.post("/api/portal/env")
async def portal_env(request: Request):
    """Enveloped guest dispatch — verify, decrypt, run, seal the response.

    The Ed25519 device signature (device registered TRUSTED at host confirm)
    authenticates the caller; the inner request is re-dispatched in-process
    as that guest user (per-boot secret header the auth middleware trusts —
    see ``AuthMiddleware._env_dispatch_user``). The guest deny-by-default
    route lists still apply underneath; failures all return a uniform 400 so
    the error channel doesn't oracle the crypto.
    """
    # Bound the per-request crypto work (AES-GCM + Ed25519) before we do any of
    # it, so an anonymous caller can't turn /env into a CPU-DoS. IP-keyed; the
    # replay guard + device-sig gate the correctness, this gates the volume.
    if (limited := _portal_too_many(request, _ENV_LIMITER)) is not None:
        return limited
    sm = _sm(request)
    conn = _conn(request)
    store = getattr(request.app.state, "settings_store", None)
    if not sm or conn is None or store is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)

    from augmentum.connect.guest_gateway import (
        ENVELOPE_CONTENT_TYPE,
        EnvelopeError,
        ReplayGuard,
        dispatch_allowed,
        get_gateway_keys,
        open_envelope,
        parse_device_public_key,
        seal_to_device,
    )

    try:
        envelope = await request.json()
    except Exception:
        return JSONResponse({"error": "bad envelope"}, status_code=400)
    device_id = str((envelope or {}).get("device_id", "") or "")
    nonce_b64 = str((envelope or {}).get("nonce", "") or "")
    if not device_id or not nonce_b64:
        return JSONResponse({"error": "bad envelope"}, status_code=400)

    # Device lookup — un-revoked trusted device, its registered key record.
    cur = await conn.execute(
        "SELECT user_id, public_key FROM trusted_mobile_devices "
        "WHERE device_id = ? AND revoked_at = ''",
        (device_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    if not row:
        return JSONResponse({"error": "bad envelope"}, status_code=400)
    user_id, key_record_raw = str(row[0]), str(row[1] or "")
    key_record = parse_device_public_key(key_record_raw)
    if key_record is None:
        # Legacy device (registered pre-gateway, empty key) — plain API only.
        return JSONResponse({"error": "bad envelope"}, status_code=400)

    guard: ReplayGuard = getattr(request.app.state, "guest_env_replay", None) or ReplayGuard()
    request.app.state.guest_env_replay = guard
    if not guard.check_and_record(device_id, nonce_b64):
        return JSONResponse({"error": "bad envelope"}, status_code=400)

    try:
        keys = await get_gateway_keys(store)
        inner = open_envelope(envelope, keys=keys, device_sign_pub_b64=key_record["sign_pub"])
    except EnvelopeError as exc:
        log.warning("portal_env_refused", device=device_id, reason=str(exc))
        return JSONResponse({"error": "bad envelope"}, status_code=400)

    method = str(inner.get("m", "") or "").upper()
    path = str(inner.get("p", "") or "")
    if not dispatch_allowed(method, path):
        return JSONResponse({"error": "bad envelope"}, status_code=400)

    user = await sm.get_user_by_id(user_id)
    if user is None or not user.is_active:
        return JSONResponse({"error": "bad envelope"}, status_code=400)

    import secrets as _secrets
    env_secret = getattr(request.app.state, "guest_env_secret", "")
    if not env_secret:
        env_secret = _secrets.token_hex(32)
        request.app.state.guest_env_secret = env_secret

    import base64 as _b64

    import httpx
    body_bytes = b""
    if inner.get("b"):
        try:
            body_bytes = _b64.b64decode(str(inner["b"]))
        except Exception:
            return JSONResponse({"error": "bad envelope"}, status_code=400)
    headers = {
        "x-augmentum-env-secret": env_secret,
        "x-augmentum-env-user": user_id,
        # Use the trusted edge IP (tunnel's Cf-Connecting-Ip or socket peer),
        # NOT _client_ip's raw X-Forwarded-For — the guest controls their own
        # XFF, and forwarding it here would feed guest-spoofed IPs to any inner
        # handler that trusts XFF for audit/geo.
        "x-forwarded-for": _rl_key(request),
    }
    ct = str(inner.get("ct", "") or "")
    if ct:
        headers["content-type"] = ct
    transport = httpx.ASGITransport(app=request.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://guest-env.internal") as client:
        resp = await client.request(method, path, content=body_bytes or None, headers=headers)

    inner_resp = {
        "s": resp.status_code,
        "ct": resp.headers.get("content-type", ""),
        "b": _b64.b64encode(resp.content).decode("ascii"),
        "ts": int(__import__("time").time()),
    }
    sealed = seal_to_device(
        inner_resp, keys=keys, device_seal_pub_b64=key_record["seal_pub"], device_id=device_id,
    )
    return JSONResponse(sealed, media_type=ENVELOPE_CONTENT_TYPE)


# ── public: invite QR (scan face-to-face, or tap the link) ───────────


@router.get("/api/invite/{token}/qr.png")
async def invite_qr_png(token: str, request: Request):
    """Render the invite's join URL as a scannable QR PNG — so sharing is a
    scan when face-to-face or a tap when remote. Public (holding the token
    already lets you decode it)."""
    host = request.headers.get("host", "") or ""
    scheme = request.headers.get("x-forwarded-proto", "https")
    # External-guest invites land on the portal; others on connect-join.
    landing = "connect-join"
    external_guest = False
    join_base = ""
    conn = _conn(request)
    if conn is not None:
        try:
            from augmentum.auth.invite_store import get_join_base, preview_invite
            preview = await preview_invite(conn, token)
            if preview and preview.get("kind") == "external_guest":
                landing = "portal"
                external_guest = True
            # The mint-time public base (e.g. the ephemeral tunnel URL) wins
            # over this request's own Host header — the QR must encode the
            # SAME bundle the mint response returned, or a QR scanned by an
            # external recipient points at a LAN address (the original bug).
            join_base = await get_join_base(conn, token)
        except Exception:
            log.warning("invite_qr_kind_lookup_failed", exc_info=True)
    base = join_base or (f"{scheme}://{host}" if host else "")
    join_url = f"{base}/ui/{landing}/?token={token}" if base else token
    # Guest-gateway bundle: pin the instance identity in the FRAGMENT (never
    # sent on the wire; travels only in the QR pixels). See 2026-07-16 spec.
    if external_guest and join_url != token:
        try:
            store = getattr(request.app.state, "settings_store", None)
            if store is not None:
                from augmentum.connect.guest_gateway import get_gateway_keys
                keys = await get_gateway_keys(store)
                join_url = f"{join_url}#k={keys.identity.public_key_b64}"
        except Exception:
            log.warning("invite_qr_pin_failed", exc_info=True)
    try:
        import qrcode
        img = qrcode.make(join_url)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as exc:
        log.warning("invite_qr_png_failed", exc_info=True)
        return JSONResponse({"error": f"qr generation failed: {exc}"}, status_code=500)
