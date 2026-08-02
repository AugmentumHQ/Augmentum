"""Authentication REST endpoints."""

from __future__ import annotations

import re

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from augmentum.auth.passwords import verify_dummy, verify_password
from augmentum.connect.rate_limit import KeyedRateLimiter
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Username: 3-32 chars, alphanumeric + underscore
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,32}$")

# Per-IP throttle for the PUBLIC, unauthenticated invite endpoints (preview +
# claim). Token entropy guards correctness; this clips a flood. ~30/min/IP is far
# above any human use of an invite link.
_INVITE_PUBLIC_LIMITER = KeyedRateLimiter(limit=30, window_s=60.0)


def _too_many(request: Request, limiter: KeyedRateLimiter) -> JSONResponse | None:
    """Return a 429 JSONResponse when ``request``'s IP is over the limit, else None."""
    allowed, retry = limiter.check(_claim_ip(request))
    if allowed:
        return None
    return JSONResponse(
        {"error": "Too many requests. Please slow down."},
        status_code=429, headers={"Retry-After": str(retry)},
    )


def _get_sm(request: Request):
    """Get SessionManager from app state."""
    return getattr(request.app.state, "session_manager", None)


def _get_user(request: Request):
    """Get authenticated user from request scope (set by AuthMiddleware)."""
    return request.scope.get("user")


def _get_ip(request: Request) -> str:
    """Get client IP for lockout/audit keying.

    Only honour X-Forwarded-For when ``settings.auth_trust_forwarded_for``
    is on (i.e., the operator confirmed they're behind a proxy that sets
    the header reliably). Trusting it by default would let an attacker
    rotate IPs by spoofing the header — defeats the IP-based lockout.
    """
    from augmentum.config import settings as _settings
    if _settings.auth_trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _cookie_domain(request: Request | None = None) -> str | None:
    """Domain attribute for the session cookie.

    Widens the session cookie to the gate domain ONLY when the login is
    actually happening on that domain (its apex or a ``<service>.<gate>``
    subdomain) — so the gate's forward_auth can validate it across subdomains.
    A login via the bare IP (or any other host) keeps a **host-only** cookie,
    so bare-IP access never breaks and the cookie is never widened for
    non-gate traffic. Returns None (host-only) when no gate domain is set or
    the request isn't on it. Cookie stays HttpOnly+Secure and is stripped at
    the proxy edge before any upstream sees it. See
    docs/superpowers/specs/2026-06-19-front-gate-identity-aware-proxy-design.md
    """
    from augmentum.config import settings
    gd = (settings.gate_domain or "").strip().lower()
    if not gd:
        return None
    host = ""
    if request is not None:
        raw = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
        host = raw.split(",")[0].split(":")[0].strip().lower()
    return gd if (host == gd or host.endswith("." + gd)) else None


def _set_session_cookie(response: JSONResponse, token: str, *, request: Request | None = None) -> JSONResponse:
    """Set the auth session cookie.

    SameSite=Lax. Augmentum is a SPA where every legitimate API call is
    a same-origin fetch, so we don't need cross-site form submissions
    to carry the cookie — what Strict adds over Lax (block GET on
    cross-site nav) doesn't materially harden us. What it DOES break
    in practice is iOS Safari, which intermittently drops Strict
    cookies on subsequent fetches during bootstrap after tab restore
    / backgrounding — observed as scattered 401s on a fresh page load
    despite a valid session row. Lax preserves CSRF protection against
    the realistic threat (cross-site POST) while restoring iOS
    reliability. The Secure flag only applies under HTTPS — the browser
    drops Secure cookies on plain HTTP, which would lock the operator
    out of localhost installs.
    """
    is_https = False
    if request:
        proto = request.headers.get("x-forwarded-proto", "")
        is_https = proto == "https" or request.url.scheme == "https"
    response.set_cookie(
        "augmentum_session",
        token,
        httponly=True,
        secure=is_https,
        samesite="lax",
        max_age=30 * 86400,  # 30 days
        path="/",
        domain=_cookie_domain(request),
    )
    return response


@router.get("/status")
async def auth_status(request: Request):
    """Check auth status. No auth required."""
    sm = _get_sm(request)
    if not sm:
        # Database is unavailable — don't claim setup is required (it would
        # send the user to the setup wizard which also can't work without a DB).
        degraded = getattr(request.app.state, "persistence_degraded", False)
        return JSONResponse({
            "setup_required": False,
            "authenticated": False,
            "user": None,
            "db_error": True,
            "error": "Database unavailable — running in memory-only mode. "
                     "Restart the server to retry the database connection."
                     if degraded else
                     "Auth system not initialized. Check server logs.",
        })
    # Durable latch, not a bare user count — so the UI doesn't drop back to
    # the setup wizard if the users table is ever transiently/legitimately
    # emptied on an already-provisioned install.
    setup_required = not await sm.setup_completed()
    # This endpoint is public (no middleware auth), so validate token manually
    user = _get_user(request)
    if not user:
        # Try extracting token from cookie/header
        token = None
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        else:
            for part in request.headers.get("cookie", "").split(";"):
                part = part.strip()
                if part.startswith("augmentum_session="):
                    token = part[18:].strip()
                    break
        if token:
            user = await sm.validate_token(token)
    return JSONResponse({
        "setup_required": setup_required,
        "authenticated": user is not None,
        "user": user.to_public_dict() if user else None,
    })


@router.post("/setup")
async def auth_setup(request: Request):
    """First-run admin creation — first user to register becomes admin.

    Endpoint is open while ``user_count == 0``; the first POST creates the
    admin and ``count == 0`` becomes false, locking the endpoint for any
    subsequent requests. No setup token: zero-friction onboarding for the
    dominant localhost case. Users who need network-exposed installs locked
    down should set up auth before exposing the port.
    """
    sm = _get_sm(request)
    if not sm:
        degraded = getattr(request.app.state, "persistence_degraded", False)
        msg = ("Database unavailable — the server is running in memory-only mode. "
               "Check Docker volume permissions and restart."
               if degraded else "Auth system not initialized. Check server logs.")
        return JSONResponse({"error": msg}, status_code=503)

    # Cheap pre-check so the obvious "already set up" case fails fast without
    # a write attempt. Uses the DURABLE latch (not a bare user count) so an
    # emptied users table can't re-arm admin creation. The real concurrency
    # defense still lives in ``create_first_admin`` — its atomic INSERT
    # ... WHERE NOT EXISTS (users) AND NOT EXISTS (latch) lets only one of N
    # racing requests succeed regardless of what this check observes.
    if await sm.setup_completed():
        return JSONResponse({"error": "Setup already completed"}, status_code=403)

    body = await request.json()
    username = body.get("username", "")
    password = body.get("password", "")

    # Validate username
    if not _USERNAME_RE.match(username):
        return JSONResponse(
            {"error": "Username must be 3-32 characters (letters, numbers, underscore)"},
            status_code=400,
        )
    if username.lower().startswith("fabric_peer_") or username.lower().startswith("fabric:"):
        return JSONResponse(
            {"error": "Usernames starting with 'fabric_peer_' or 'fabric:' are reserved"},
            status_code=400,
        )

    # Validate password
    if len(password) < 8:
        return JSONResponse({"error": "Password must be at least 8 characters"}, status_code=400)

    # Atomic admin creation. Returns None if a concurrent request
    # already populated the users table while we were validating.
    user = await sm.create_first_admin(username, password)
    if user is None:
        return JSONResponse({"error": "Setup already completed"}, status_code=403)

    # Backfill existing data to this admin
    await sm.backfill_user_id(user.id)

    # Create session
    ip = _get_ip(request)
    ua = request.headers.get("user-agent", "")
    session_token = await sm.create_session(
        user.id,
        ip_address=ip,
        user_agent=ua,
        source="web",
    )

    response = JSONResponse({"user": user.to_public_dict()})
    return _set_session_cookie(response, session_token, request=request)


@router.post("/login")
async def auth_login(request: Request):
    """Login with username/password."""
    sm = _get_sm(request)
    if not sm:
        return JSONResponse({"error": "Auth not initialized"}, status_code=500)

    body = await request.json()
    username = body.get("username", "")
    password = body.get("password", "")
    ip = _get_ip(request)

    # Check lockout BEFORE expensive argon2 verify
    retry_after = await sm.check_lockout(username, ip)
    if retry_after:
        return JSONResponse(
            {"error": "Too many failed attempts. Try again later.", "retry_after": retry_after},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )

    # Lookup user
    pw_hash = await sm.get_password_hash(username)
    if not pw_hash:
        # Unknown username — still run argon2 for constant time
        verify_dummy(password)
        await sm.record_failed_attempt(username, ip)
        return JSONResponse({"error": "Invalid username or password"}, status_code=401)

    # Verify password
    if not verify_password(pw_hash, password):
        await sm.record_failed_attempt(username, ip)
        return JSONResponse({"error": "Invalid username or password"}, status_code=401)

    # Get full user
    user = await sm.get_user_by_username(username)
    if not user or not user.is_active:
        return JSONResponse({"error": "Account is disabled"}, status_code=403)

    # Clear failed attempts on success
    await sm.clear_failed_attempts(username)

    # Guest comms-portal gate (device-trust model): a portal guest may sign
    # in once their host has CONFIRMED them — and then from ANY network,
    # because their session is DEVICE-bound, not IP-bound (so they reconnect
    # from home WiFi / cellular without re-approval). The IP is recorded as a
    # logged "seen location" for the host's awareness, not a gate. Only
    # applies to portal guests ('none' => other guest type, untouched).
    guest_device_id = ""
    if getattr(user, "role", "") == "guest":
        conn = _get_conn(request)
        if conn is not None:
            from augmentum.connect.guest_portal import (
                allow_ip,
                device_for_guest,
                registration_state,
            )
            state = await registration_state(conn, guest_user_id=user.id)
            if state in ("pending", "denied"):
                return JSONResponse(
                    {"error": "Your host hasn’t confirmed you yet — hang tight.",
                     "guest_state": "waiting"}, status_code=403)
            if state == "confirmed":
                guest_device_id = await device_for_guest(conn, guest_user_id=user.id)
                try:  # record the location for the host; never blocks
                    await allow_ip(conn, guest_user_id=user.id, ip=ip, added_by="login")
                except Exception:
                    log.warning("guest_login_ip_log_failed", exc_info=True)

    # Create session
    ua = request.headers.get("user-agent", "")
    session_token = await sm.create_session(
        user.id,
        ip_address=ip,
        user_agent=ua,
        source="web",
        # Portal guests get a device-bound session so the host can revoke
        # that one device; empty for everyone else (unchanged behavior).
        source_device_id=guest_device_id,
    )

    response = JSONResponse({"user": user.to_public_dict()})
    return _set_session_cookie(response, session_token, request=request)


@router.post("/logout")
async def auth_logout(request: Request):
    """Logout and revoke session."""
    sm = _get_sm(request)
    user = _get_user(request)
    if not sm or not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    # Extract token to revoke it
    token = None
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    else:
        for cookie_part in request.headers.get("cookie", "").split(";"):
            cookie_part = cookie_part.strip()
            if cookie_part.startswith("augmentum_session="):
                token = cookie_part[18:].strip()
                break

    if token:
        await sm.revoke_session(token)

    response = JSONResponse({"ok": True})
    # Clear both the host-only cookie and (if logging out on the gate domain)
    # the domain-scoped one, so logout works regardless of how they logged in.
    response.delete_cookie("augmentum_session", path="/")
    dom = _cookie_domain(request)
    if dom:
        response.delete_cookie("augmentum_session", path="/", domain=dom)
    return response


@router.post("/ws-ticket")
async def auth_ws_ticket(request: Request):
    """Get a short-lived WebSocket ticket."""
    sm = _get_sm(request)
    user = _get_user(request)
    if not sm or not user:
        # Log which leg of the gate failed — operators were seeing
        # repeated "Failed to get auth ticket" on the terminal/voice UI
        # with no signal at all on the server side. The two causes have
        # very different fixes: sm=None means the auth subsystem never
        # initialized (DB issue at startup); user=None means the request
        # arrived without a valid session cookie (cookie expired, was
        # set on a different host, or CSRF middleware rejected upstream
        # — though CSRF should have 403'd before we got here).
        log.warning(
            "ws_ticket_unauthenticated",
            reason="no_session_manager" if not sm else "no_user_in_scope",
            origin=request.headers.get("origin", "")[:200],
            referer=request.headers.get("referer", "")[:200],
            host=request.headers.get("host", "")[:100],
        )
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    ticket = sm.create_ws_ticket(user.id)
    return JSONResponse({"ticket": ticket})


@router.get("/me")
async def auth_me(request: Request):
    """Get current user profile."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    return JSONResponse({
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "created_at": user.created_at,
        "quota_bytes": user.quota_bytes,
    })


@router.put("/me/password")
async def auth_change_password(request: Request):
    """Change current user's password."""
    sm = _get_sm(request)
    user = _get_user(request)
    if not sm or not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    body = await request.json()
    current_password = body.get("current_password", "")
    new_password = body.get("new_password", "")

    # Rate-limit current-password verification the same way /login does.
    # A stolen session token would otherwise let an attacker brute-force
    # the password endlessly via this endpoint — each attempt is one
    # argon2 verify, slow but unbounded. Reuse the per-username +
    # per-IP lockout that login already implements.
    ip_address = _get_ip(request)
    lockout = await sm.check_lockout(user.username, ip_address)
    if lockout is not None:
        return JSONResponse(
            {"error": "Too many failed attempts. Try again later.",
             "retry_after_seconds": lockout},
            status_code=429,
        )

    # Verify current password
    pw_hash = await sm.get_password_hash(user.username)
    if not pw_hash or not verify_password(pw_hash, current_password):
        await sm.record_failed_attempt(user.username, ip_address)
        return JSONResponse({"error": "Current password is incorrect"}, status_code=401)

    # Clear lockout counter on a successful verify (matches the
    # successful-login path) so a user who flubbed their password once
    # doesn't carry the count into the next sensitive operation.
    await sm.clear_failed_attempts(user.username)

    if len(new_password) < 8:
        return JSONResponse({"error": "New password must be at least 8 characters"}, status_code=400)

    # Update password
    await sm.update_password(user.id, new_password)

    # Revoke all OTHER sessions (keep current)
    token = None
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    else:
        for cookie_part in request.headers.get("cookie", "").split(";"):
            cookie_part = cookie_part.strip()
            if cookie_part.startswith("augmentum_session="):
                token = cookie_part[18:].strip()
                break

    if token:
        await sm.revoke_all_sessions(user.id, except_token=token)

    await sm.write_audit(
        actor=user, target=user, action="password_change_self",
        ip_address=_get_ip(request),
    )
    return JSONResponse({"ok": True})


# ------------------------------------------------------------------
# API keys (inbound — for external OpenAI-compatible clients)
# ------------------------------------------------------------------

def _get_akm(request: Request):
    """Get ApiKeyManager from app state. May be None when DB is down."""
    return getattr(request.app.state, "api_key_manager", None)


@router.get("/keys")
async def list_api_keys(request: Request):
    """List the current user's inbound API keys (metadata only)."""
    user = _get_user(request)
    akm = _get_akm(request)
    if not user or not akm:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    keys = await akm.list_for_user(user.id)
    return JSONResponse({"keys": keys})


@router.post("/keys")
async def create_api_key(request: Request):
    """Mint a new API key. The raw value is returned exactly once.

    Body: ``{"name": "OpenWebUI laptop", "scope": "chat"}``
    Response: ``{"key": "sk-aug-...", "id": "...", ...}``
    """
    user = _get_user(request)
    akm = _get_akm(request)
    if not user or not akm:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    # Scoped, non-member accounts (fabric 'peer' service users and external
    # 'guest' accounts) must not mint API keys. They already authenticate every
    # call by signed envelope / scoped session; a persistent sk-aug- token would
    # survive unpairing/revocation and defeat it. The middleware allow/deny lists
    # block this surface at the door (FabricPeerMiddleware for peers,
    # _GUEST_DENIED_PREFIXES for guests), but this in-handler guard keeps the
    # rule intact if a future middleware change ever loosens either list.
    if user.role in ("peer", "guest"):
        return JSONResponse(
            {"error": "This account type cannot mint API keys"},
            status_code=403,
        )

    try:
        body = await request.json()
    except Exception:
        body = {}
    name = (body.get("name") or "").strip()[:100]
    scope = (body.get("scope") or "chat").strip().lower()
    if scope not in ("chat", "admin"):
        scope = "chat"
    # Only admins can mint admin-scope keys.
    if scope == "admin" and not user.is_admin:
        return JSONResponse({"error": "Admin scope requires admin user"}, status_code=403)

    raw, meta = await akm.create(user.id, name=name, scope=scope)
    sm = _get_sm(request)
    if sm:
        await sm.write_audit(
            actor=user, target=user, action="api_key_created",
            ip_address=_get_ip(request),
        )
    return JSONResponse({"key": raw, **meta})


@router.delete("/keys/{key_id}")
async def revoke_api_key(request: Request, key_id: str):
    """Revoke one of the current user's API keys."""
    user = _get_user(request)
    akm = _get_akm(request)
    if not user or not akm:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    deleted = await akm.revoke(key_id, user.id)
    if not deleted:
        return JSONResponse({"error": "Key not found"}, status_code=404)
    sm = _get_sm(request)
    if sm:
        await sm.write_audit(
            actor=user, target=user, action="api_key_revoked",
            ip_address=_get_ip(request),
        )
    return JSONResponse({"ok": True})


# ------------------------------------------------------------------
# Admin endpoints
# ------------------------------------------------------------------

@router.get("/users")
async def admin_list_users(request: Request):
    """List all users (admin only). Admin check done by middleware."""
    sm = _get_sm(request)
    user = _get_user(request)
    if not sm or not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    users = await sm.list_users()
    return JSONResponse({"users": [u.to_public_dict() for u in users]})


@router.post("/users")
async def admin_create_user(request: Request):
    """Create a new user (admin only)."""
    sm = _get_sm(request)
    user = _get_user(request)
    if not sm or not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    body = await request.json()
    username = body.get("username", "")
    password = body.get("password", "")
    role = body.get("role", "user")

    if not _USERNAME_RE.match(username):
        return JSONResponse(
            {"error": "Username must be 3-32 characters (letters, numbers, underscore)"},
            status_code=400,
        )
    # Reserve the fabric_peer_<hex> namespace for the receiver-side
    # service users that FabricPeerMiddleware provisions. An admin
    # creating a colliding username would let a real account squat the
    # slot a future peer needs, breaking that peer's authentication
    # (UNIQUE collision in get_or_create_fabric_peer_user → None →
    # signed request falls through unauthenticated).
    if username.lower().startswith("fabric_peer_") or username.lower().startswith("fabric:"):
        return JSONResponse(
            {"error": "Usernames starting with 'fabric_peer_' or 'fabric:' are reserved"},
            status_code=400,
        )
    if len(password) < 8:
        return JSONResponse({"error": "Password must be at least 8 characters"}, status_code=400)
    if role not in ("user", "admin"):
        return JSONResponse({"error": "Role must be 'user' or 'admin'"}, status_code=400)

    # Check uniqueness
    existing = await sm.get_user_by_username(username)
    if existing:
        return JSONResponse({"error": "Username already taken"}, status_code=409)

    new_user = await sm.create_user(username, password, role=role)
    await sm.write_audit(
        actor=user, target=new_user, action="user_create",
        detail=f"role={role}", ip_address=_get_ip(request),
    )
    return JSONResponse({"user": new_user.to_public_dict()}, status_code=201)


@router.put("/users/{user_id}")
async def admin_update_user(user_id: str, request: Request):
    """Update a user (admin only)."""
    sm = _get_sm(request)
    user = _get_user(request)
    if not sm or not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    body = await request.json()

    # Cannot demote self
    if user_id == user.id and body.get("role") and body["role"] != "admin":
        return JSONResponse({"error": "Cannot demote yourself"}, status_code=400)

    # Cannot deactivate self
    if user_id == user.id and body.get("is_active") is False:
        return JSONResponse({"error": "Cannot deactivate yourself"}, status_code=400)

    # Last-admin guard: refuse demote/deactivate if it would leave the
    # system with zero active admins.
    new_role = body.get("role") if "role" in body else None
    new_active = body.get("is_active") if "is_active" in body else None
    if await sm._would_remove_last_admin(user_id, new_role=new_role, new_active=new_active):
        return JSONResponse(
            {"error": "Cannot remove the last active admin. Promote another user first."},
            status_code=400,
        )

    target_before = await sm.get_user_by_id(user_id)
    updated = await sm.update_user(user_id, **body)
    if not updated:
        return JSONResponse({"error": "User not found or no valid fields"}, status_code=404)

    # If deactivated, revoke all their sessions
    if body.get("is_active") is False:
        await sm.revoke_all_sessions(user_id)

    # Audit each meaningful field that changed
    changes = []
    if "role" in body and target_before and target_before.role != body["role"]:
        changes.append(f"role: {target_before.role}→{body['role']}")
    if "is_active" in body and target_before and bool(target_before.is_active) != bool(body["is_active"]):
        changes.append(f"active: {bool(target_before.is_active)}→{bool(body['is_active'])}")
    if "display_name" in body:
        changes.append("display_name updated")
    if "quota_bytes" in body:
        changes.append(f"quota_bytes={body['quota_bytes']}")
    if "content_level" in body and target_before and target_before.content_level != body["content_level"]:
        changes.append(f"content_level: {target_before.content_level}→{body['content_level']}")
    if changes:
        await sm.write_audit(
            actor=user, target=target_before, action="user_update",
            detail="; ".join(changes), ip_address=_get_ip(request),
        )

    return JSONResponse({"ok": True})


@router.put("/users/{user_id}/password")
async def admin_reset_password(user_id: str, request: Request):
    """Admin-initiated password reset for another user. Revokes all the
    target's sessions on success."""
    sm = _get_sm(request)
    user = _get_user(request)
    if not sm or not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    body = await request.json()
    new_password = body.get("new_password", "")
    if len(new_password) < 8:
        return JSONResponse({"error": "Password must be at least 8 characters"}, status_code=400)

    target = await sm.get_user_by_id(user_id)
    if not target:
        return JSONResponse({"error": "User not found"}, status_code=404)

    await sm.update_password(user_id, new_password)
    # Revoke ALL of the target's sessions — they must sign in with the new password
    await sm.revoke_all_sessions(user_id)

    await sm.write_audit(
        actor=user, target=target, action="password_reset_admin",
        ip_address=_get_ip(request),
    )
    return JSONResponse({"ok": True})


@router.delete("/users/{user_id}")
async def admin_delete_user(user_id: str, request: Request):
    """Delete a user (admin only). Requires X-Confirm-Delete header."""
    sm = _get_sm(request)
    user = _get_user(request)
    if not sm or not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    # Cannot delete self
    if user_id == user.id:
        return JSONResponse({"error": "Cannot delete yourself"}, status_code=400)

    # Require confirmation header
    if request.headers.get("x-confirm-delete") != "true":
        return JSONResponse({"error": "Missing X-Confirm-Delete: true header"}, status_code=400)

    if await sm._would_remove_last_admin(user_id, deleting=True):
        return JSONResponse(
            {"error": "Cannot delete the last active admin. Promote another user first."},
            status_code=400,
        )

    target = await sm.get_user_by_id(user_id)
    deleted = await sm.delete_user(user_id)
    if not deleted:
        return JSONResponse({"error": "User not found"}, status_code=404)

    await sm.write_audit(
        actor=user, target=target, action="user_delete",
        ip_address=_get_ip(request),
    )
    return JSONResponse({"ok": True})


@router.get("/audit")
async def admin_list_audit(request: Request, limit: int = 100):
    """Return recent auth audit entries (admin only)."""
    sm = _get_sm(request)
    user = _get_user(request)
    if not sm or not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    entries = await sm.list_audit(limit=limit)
    return JSONResponse({"entries": entries})


# ─── Invites (Connect open-access onboarding, Phase 1) ──────────────────────
#
# Admin management lives under ``/api/auth/invites`` (plural) — gated by the
# middleware ``_ADMIN_PREFIXES`` admin check. The PUBLIC preview + claim live
# under ``/api/auth/invite/`` (singular) — exempt in the middleware
# ``_PUBLIC_PREFIXES`` because the claimant has no account yet; the high-entropy
# token IS the credential (same model as cast guest invites).


def _get_conn(request: Request):
    """Shared app DB connection (the same one SessionManager writes through)."""
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None)
    return getattr(backend, "conn", None)


async def _build_invite_link(
    request: Request, token: str, scope_str: str,
    *, ttl_hours: int = 168, allowed_ips: list[str] | None = None,
) -> tuple[str, str, dict]:
    """Plan the least-exposing reachable link for an invite.

    Uses the adaptive reachability ladder (LAN → tailnet → Funnel → cloudflared):
    picks the most-private tier that can reach the chosen ``recipient_scope``,
    builds the join URL from it, and returns reach metadata (tier, note, any
    privacy downgrade) for the UI. Tunnel tiers that aren't provisioned yet fall
    back to the best static host with an honest note — so LAN/tailnet links work
    today and public tiers light up once a live tunnel engine is registered.

    ``allowed_ips`` (IPs/CIDRs) pins a public tunnel to specific addresses — used
    for IP-whitelisted re-access so a leaked URL is useless from any other IP.
    The tunnel's open window is tied to ``ttl_hours`` (the invite's own expiry),
    NOT a fixed week, so the public door closes when the link does.
    """
    from augmentum.config import settings as _settings
    from augmentum.connect.reachability import (
        EngineUnavailable,
        detect_capabilities,
        get_engine,
        parse_recipient_scope,
        plan_reachability,
        tailnet_ip_from_sans,
    )

    path = f"/ui/connect-join/?token={token}"
    resolver = getattr(request.app.state, "public_host_resolver", None)
    resolver_host = ""
    if resolver is not None:
        try:
            resolver_host = resolver.public_host(request) or ""
        except Exception:
            resolver_host = ""

    # The resolver only ever sees ONE host — whichever address the admin's
    # browser connected on (usually the LAN IP). So a "My tailnet" invite minted
    # from a LAN browser found no tailnet host and fell all the way back to the
    # LAN base_url. Feed the machine's own Tailscale IP (auto-detected from the
    # TLS SANs) as an extra candidate, carrying the same port we were reached on,
    # so the tailnet tier actually has a host to offer.
    extra_hosts: tuple[str, ...] = ()
    ts_ip = tailnet_ip_from_sans()
    if ts_ip:
        port = ""
        if ":" in resolver_host:
            port = resolver_host.rsplit(":", 1)[1].strip()
        elif request.url.port:
            port = str(request.url.port)
        extra_hosts = (f"{ts_ip}:{port}" if port else ts_ip,)

    caps = detect_capabilities(
        resolver_host=resolver_host,
        configured_host=(_settings.augmentum_public_host or "").strip(),
        extra_hosts=extra_hosts,
    )
    scope = parse_recipient_scope(scope_str)
    plan = plan_reachability(scope, caps)

    allowed_ips = [ip for ip in (allowed_ips or []) if ip]
    from augmentum.connect.reachability import ReachTier as _RT
    # Honesty: a durable tier (LAN/TAILNET/FUNNEL) survives restarts; the
    # cloudflared last-resort URL is per-session (changes on restart). And when
    # we had to fall to cloudflared but a ts.net host exists, the operator could
    # upgrade to a private, durable Funnel address — surface that.
    durable = bool(plan.tier and plan.tier in (_RT.LAN, _RT.TAILNET, _RT.TS_FUNNEL))
    ts_name = ""
    from augmentum.connect.reachability import _bare_tsnet_name
    for h in (caps.tailnet_host, ts_ip):
        ts_name = _bare_tsnet_name(h)
        if ts_name:
            break
    upgrade_available = bool(
        plan.tier == _RT.CLOUDFLARED and ts_name and not caps.funnel_url,
    )
    meta: dict = {
        "scope": scope.name.lower(),
        "tier": plan.tier.name.lower() if plan.tier is not None else None,
        "needs_tunnel": plan.needs_tunnel,
        "privacy_downgrade": plan.privacy_downgrade,
        "note": plan.note,
        "public": bool(plan.tier and plan.tier.is_public),
        "durable": durable,
        "ip_locked": bool(allowed_ips and plan.tier and plan.tier.is_public),
        "allowed_ips": allowed_ips,
        "upgrade_available": upgrade_available,
        "tailnet_hostname": ts_name,
    }
    if upgrade_available:
        meta["upgrade_hint"] = (
            "This is a temporary anonymous address that changes on restart. For a "
            "private, durable link, enable Tailscale Funnel (set "
            "AUGMENTUM_CONNECT_FUNNEL=1 + AUGMENTUM_CONNECT_FUNNEL_URL / port)."
        )

    base = ""
    if plan.reachable:
        # Tiers with a live lifecycle register their manager lazily the first
        # time they're actually chosen (idempotent). LAN/TAILNET and config-mode
        # funnel need no live process.
        if plan.tier.name == "CLOUDFLARED":
            from augmentum.connect.tunnel_manager import get_or_register_cloudflared_manager
            get_or_register_cloudflared_manager()
        elif plan.tier.name == "TS_FUNNEL" and getattr(_settings, "augmentum_connect_funnel_live", False):
            from augmentum.connect.funnel_manager import get_or_register_funnel_manager
            get_or_register_funnel_manager(
                funnel_port=(_settings.augmentum_connect_funnel_port or "").strip() or None,
            )
        try:
            from augmentum.auth.invite_store import tunnel_ref_for_token
            base = await get_engine(plan.tier).ensure(
                invite_id=tunnel_ref_for_token(token), plan=plan,
                ttl_seconds=max(1, int(ttl_hours)) * 3600,
                allowed_ips=allowed_ips,
            )
        except EngineUnavailable as exc:
            # Tunnel tier selected but its live manager isn't wired (the bones).
            # Fall back to the best static host we have, and say so honestly.
            log.info("invite_reach_tier_unprovisioned", tier=meta["tier"], reason=str(exc))
            meta["note"] = (
                f"{plan.tier.name.lower()} reach isn't provisioned on this host yet — "
                "the link will only work on your local network / tailnet for now."
            )
            meta["public"] = False
            from augmentum.connect.reachability import _static_url
            base = _static_url(caps.tailnet_host or caps.lan_host)

    if not base:
        # No planned host at all → fall back to the request's own origin so the
        # admin at least gets a same-origin link (works where they're standing).
        try:
            base = str(request.base_url).rstrip("/")
        except Exception:
            base = ""
        if not meta["note"]:
            meta["note"] = "Couldn't determine a reachable host — this link may only work locally."

    return path, (f"{base}{path}" if base else path), meta


def _claim_ip(request: Request) -> str:
    """The recipient's real IP at claim time, for later IP-whitelisted re-access.

    Through the cloudflared tunnel the visitor IP is in ``Cf-Connecting-Ip``;
    fall back to the first ``X-Forwarded-For`` hop, then the socket peer. This is
    recorded (``claimed_ip``) so the admin can pin a reconnect link to it."""
    cf = request.headers.get("cf-connecting-ip", "").strip()
    if cf:
        return cf
    xff = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
    if xff:
        return xff
    return request.client.host if request.client else ""


def _parse_allowed_ips(body: dict) -> tuple[list[str], str]:
    """Validate the invite's IP allowlist from the request body.

    Reads ``allowed_ips`` (list or comma-separated string) and ``recipient_ip``
    (single). Each entry must be a valid IP or CIDR; an invalid one is a 400 so
    a typo can't silently leave the door open. Returns ``(ips, "")`` on success
    or ``([], error)``.
    """
    import ipaddress

    raw = body.get("allowed_ips")
    items: list[str] = []
    if isinstance(raw, str):
        items = [p.strip() for p in raw.split(",")]
    elif isinstance(raw, list):
        items = [str(p).strip() for p in raw]
    single = (body.get("recipient_ip") or "").strip()
    if single:
        items.append(single)

    out: list[str] = []
    for entry in items:
        if not entry:
            continue
        try:
            if "/" in entry:
                ipaddress.ip_network(entry, strict=False)
            else:
                ipaddress.ip_address(entry)
        except ValueError:
            return [], f"Invalid IP or CIDR: {entry!r}"
        if entry not in out:
            out.append(entry)
    return out, ""


async def _release_invite_reach(token: str) -> None:
    """Release any public-tunnel exposure held for an invite (best-effort).

    Keyed by the invite's HASH prefix (``tunnel_ref_for_token``) — the same
    ref the mint used — so lifecycle hooks that only hold the hash (portal
    confirm/deny, admin revoke) can release too. Safe to call for every
    claim — it no-ops for LAN/tailnet invites and engines that aren't tracking
    this id. A failure here must never sink an otherwise-successful claim.
    """
    from augmentum.auth.invite_store import tunnel_ref_for_token
    await release_invite_reach_by_ref(tunnel_ref_for_token(token))


async def release_invite_reach_by_ref(ref: str) -> None:
    """Release an EPHEMERAL public-tunnel ref by its hash-derived key (best-effort).

    Shared with portal_routes (deny / revoke). Only the ephemeral cloudflared
    tunnel is released — TS_FUNNEL is a STANDING, durable door (stable URL, the
    guest's ongoing transport) and must never be torn down per-invite; LAN/tailnet
    have no door. Releasing funnel here would kill every guest's access.
    """
    if not ref:
        return
    try:
        from augmentum.connect.reachability import ReachTier, get_engine
        await get_engine(ReachTier.CLOUDFLARED).release(invite_id=ref)
    except Exception:
        log.warning("invite_reach_release_failed", exc_info=True)


@router.post("/invites")
async def admin_create_invite(request: Request):
    """Mint an invite link (admin only). Returns the raw token ONCE."""
    sm = _get_sm(request)
    user = _get_user(request)
    conn = _get_conn(request)
    if not sm or not user or conn is None:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    body = await request.json()
    role = body.get("role", "user")
    kind = body.get("kind", "account_claim")
    email = (body.get("email") or "").strip()
    handle_hint = (body.get("handle_hint") or "").strip()
    try:
        max_uses = int(body.get("max_uses", 1) or 1)
        ttl_hours = int(body.get("expires_in_hours", 168) or 0)
    except (TypeError, ValueError):
        return JSONResponse({"error": "max_uses / expires_in_hours must be integers"}, status_code=400)
    if role not in ("user", "guest"):
        return JSONResponse({"error": "Role must be 'user' or 'guest'"}, status_code=400)
    if kind not in ("account_claim", "external_guest"):
        return JSONResponse({"error": "Invalid invite kind"}, status_code=400)

    # IP allowlist for a public link — pins the tunnel to specific addresses so
    # the recipient can regain access while a leaked URL is dead from any other
    # IP. Accepts ``allowed_ips`` (list or comma string) and/or ``recipient_ip``.
    allowed_ips, ip_err = _parse_allowed_ips(body)
    if ip_err:
        return JSONResponse({"error": ip_err}, status_code=400)

    from augmentum.auth.invite_store import create_invite
    try:
        invite = await create_invite(
            conn, inviter_user_id=user.id, kind=kind, role=role,
            invitee_email=email, handle_hint=handle_hint,
            max_uses=max_uses, ttl_hours=ttl_hours,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    scope_str = body.get("recipient_scope") or "lan"
    path, url, reach = await _build_invite_link(
        request, invite["token"], scope_str,
        ttl_hours=ttl_hours or 168, allowed_ips=allowed_ips,
    )
    # External guests land on the installable comms portal; account-claim
    # invites keep the existing connect-join page.
    if kind == "external_guest":
        path = f"/ui/portal/?token={invite['token']}"
        url = (url or "").replace("/ui/connect-join/?token=", "/ui/portal/?token=")
    invite["join_path"] = path
    invite["join_url"] = url

    # Guest gateway (external_guest): the bundle carries the trust anchor —
    # this instance's Ed25519 identity — in the URL FRAGMENT. Fragments never
    # travel on the wire (not to the tunnel edge, not in Referer), so the
    # pinned key rides only inside the link text / QR pixels. The portal pins
    # it on first load and verifies the gateway seal key against it. See the
    # 2026-07-16 guest-gateway spec.
    if kind == "external_guest":
        try:
            store = getattr(request.app.state, "settings_store", None)
            if store is not None:
                from augmentum.connect.guest_gateway import get_gateway_keys
                keys = await get_gateway_keys(store)
                frag = f"#k={keys.identity.public_key_b64}"
                invite["join_path"] = f"{path}{frag}"
                if url:
                    invite["join_url"] = f"{url}{frag}"
        except Exception:
            # The bundle stays shareable without the pin (legacy mode); the
            # portal shows "no E2E" rather than the mint failing outright.
            log.warning("invite_bundle_pin_failed", exc_info=True)
        # Persist the mint-time base so the QR endpoint reproduces the SAME
        # bundle even when the QR request arrives under a different host.
        try:
            base = url[: -len(path)] if url and path and url.endswith(path) else ""
            if not base and url:
                from urllib.parse import urlsplit
                parts = urlsplit(url)
                base = f"{parts.scheme}://{parts.netloc}" if parts.netloc else ""
            if base:
                from augmentum.auth.invite_store import hash_token, set_join_base
                await set_join_base(conn, token_hash=hash_token(invite["token"]), join_base=base)
        except Exception:
            log.warning("invite_join_base_persist_failed", exc_info=True)

    # Both ways to share: tap the link, or scan the QR face-to-face.
    invite["qr_path"] = f"/api/invite/{invite['token']}/qr.png"
    invite["reach"] = reach
    return JSONResponse({"invite": invite}, status_code=201)


@router.get("/invites")
async def admin_list_invites(request: Request):
    """List invites (admin only). Raw tokens are NOT recoverable."""
    user = _get_user(request)
    conn = _get_conn(request)
    if not user or conn is None:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    from augmentum.auth.invite_store import list_invites
    invites = await list_invites(conn, inviter_user_id=None)
    return JSONResponse({"invites": invites})


@router.delete("/invites/{invite_id}")
async def admin_revoke_invite(invite_id: str, request: Request):
    """Revoke an invite (admin only)."""
    user = _get_user(request)
    conn = _get_conn(request)
    if not user or conn is None:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    from augmentum.auth.invite_store import (
        revoke_invite,
        token_hash_for_id,
        tunnel_ref_for_hash,
    )
    token_hash = await token_hash_for_id(conn, invite_id)
    ok = await revoke_invite(conn, invite_id=invite_id, inviter_user_id=None)
    if not ok:
        return JSONResponse({"error": "Invite not found or already revoked"}, status_code=404)
    # A revoked invite's public door closes immediately (best-effort; the TTL
    # reaper is the backstop).
    await release_invite_reach_by_ref(tunnel_ref_for_hash(token_hash))
    return JSONResponse({"revoked": True})


# Registered BEFORE ``/invite/{token}`` so the literal ``check-username`` segment
# wins over the path param (Starlette matches in declaration order). Parallel to
# ``/invite/{token}/claim`` — the live invite token gates the lookup so the
# endpoint can't be used to enumerate the user table without holding an invite.
@router.get("/invite/{token}/check-username")
async def public_check_username(token: str, request: Request):
    """PUBLIC: live availability + validity check for a username during onboarding.

    Returns ``{available, reason}`` where ``reason`` is one of ``ok`` / ``invalid``
    / ``reserved`` / ``taken``. Gated by a live invite token (same public surface
    as preview/claim) and rate-limited so it can't be ground for enumeration.
    """
    if (limited := _too_many(request, _INVITE_PUBLIC_LIMITER)) is not None:
        return limited
    sm = _get_sm(request)
    conn = _get_conn(request)
    if not sm or conn is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)

    # The token must resolve to a still-claimable invite — knowledge of a valid
    # invite is the price of admission to probing usernames.
    from augmentum.auth.invite_store import preview_invite
    preview = await preview_invite(conn, token)
    if preview is None or preview.get("status") not in (None, "active"):
        return JSONResponse({"error": "Invite not found"}, status_code=404)

    candidate = (request.query_params.get("u") or "").strip()
    if not _USERNAME_RE.match(candidate):
        return JSONResponse({"available": False, "reason": "invalid"})
    from augmentum.auth.models import is_reserved_username
    if is_reserved_username(candidate):
        return JSONResponse({"available": False, "reason": "reserved"})
    if await sm.get_user_by_username(candidate):
        return JSONResponse({"available": False, "reason": "taken"})
    return JSONResponse({"available": True, "reason": "ok"})


@router.get("/invite/{token}")
async def public_preview_invite(token: str, request: Request):
    """PUBLIC: preview an invite by token (who invited you, to which instance).

    Non-consuming. Returns 404 for an unknown token; otherwise a public-safe
    payload including the invite's lifecycle ``status`` so the join page can
    show "expired / already used / revoked" without attempting a claim.
    """
    if (limited := _too_many(request, _INVITE_PUBLIC_LIMITER)) is not None:
        return limited
    conn = _get_conn(request)
    if conn is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)
    from augmentum.auth.invite_store import preview_invite
    preview = await preview_invite(conn, token)
    if preview is None:
        return JSONResponse({"error": "Invite not found"}, status_code=404)
    return JSONResponse({"invite": preview})


@router.post("/invite/{token}/claim")
async def public_claim_invite(token: str, request: Request):
    """PUBLIC: claim an invite — create an account with a self-chosen password.

    On success: provisions the account (role/email from the invite), stamps the
    invite claimed, auto-adds the inviter ↔ new user as mutual contacts, opens a
    session, and returns the user with the session cookie set (auto-login).
    """
    if (limited := _too_many(request, _INVITE_PUBLIC_LIMITER)) is not None:
        return limited
    sm = _get_sm(request)
    conn = _get_conn(request)
    if not sm or conn is None:
        return JSONResponse({"error": "Service unavailable"}, status_code=503)

    body = await request.json()
    username = body.get("username", "")
    password = body.get("password", "")
    display_name = (body.get("display_name") or "").strip()

    if not _USERNAME_RE.match(username):
        return JSONResponse(
            {"error": "Username must be 3-32 characters (letters, numbers, underscore)"},
            status_code=400,
        )
    from augmentum.auth.models import is_reserved_username
    if is_reserved_username(username):
        return JSONResponse({"error": "That username is reserved"}, status_code=400)
    if len(password) < 8:
        return JSONResponse({"error": "Password must be at least 8 characters"}, status_code=400)
    if await sm.get_user_by_username(username):
        return JSONResponse({"error": "Username already taken"}, status_code=409)

    # Atomically consume one use BEFORE creating the account, so a race on a
    # 1-use invite can't provision two accounts. If account creation then fails,
    # the use is forfeit (acceptable — the operator can re-mint).
    from augmentum.auth.invite_store import consume_invite, mark_claimed
    invite = await consume_invite(conn, token)
    if invite is None:
        return JSONResponse(
            {"error": "This invite is invalid, expired, used up, or revoked."},
            status_code=410,
        )

    # An external_guest invite always provisions a scoped guest account, so the
    # durable guest-pass surface + ACL gate apply regardless of the role field.
    is_guest_claim = invite["kind"] == "external_guest"
    role = "guest" if is_guest_claim else (
        invite["role"] if invite["role"] in ("user", "guest") else "user"
    )
    try:
        new_user = await sm.create_user(
            username, password, role=role,
            display_name=display_name, email=invite["invitee_email"],
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    await mark_claimed(
        conn, token_hash=invite["token_hash"], claimed_user_id=new_user.id,
        claimed_ip=_claim_ip(request),
    )

    # Auto-add the inviter ↔ new user as mutual contacts so the thread is one
    # tap away on both sides. Best-effort: a contact-write failure must not sink
    # an otherwise-successful claim.
    try:
        from augmentum.connect.contact_store import ensure_contact
        from augmentum.connect.contacts import local_did_for
        inviter_id = invite["inviter_user_id"]
        await ensure_contact(
            conn, user_id=new_user.id, peer_did=local_did_for(inviter_id),
            discovery_source="invite",
        )
        await ensure_contact(
            conn, user_id=inviter_id, peer_did=local_did_for(new_user.id),
            discovery_source="invite",
        )
    except Exception:
        log.warning("invite_claim_auto_contact_failed", exc_info=True)

    # Persist the invitee's directory-visibility choice. ``discoverable=false``
    # means "only people who already have me as a contact can reach me" — i.e.
    # just the inviter (auto-added above); the invitee won't appear in the
    # same-instance directory or search for anyone else. ``true`` exposes them to
    # everyone on the instance. Guests are non-discoverable by role, so only full
    # users carry this. Omitted field → leave unset (server default = visible).
    if not is_guest_claim and "discoverable" in body:
        settings_store = getattr(request.app.state, "settings_store", None)
        if settings_store is not None:
            try:
                val = "true" if body.get("discoverable") else "false"
                await settings_store.set_user(
                    new_user.id, "ui.connectDiscoverableSameInstance", val,
                )
            except Exception:
                log.warning("invite_claim_discoverable_persist_failed", exc_info=True)

    # External guest → issue the durable grant + surface token (returned ONCE so
    # the saved PWA can re-establish a scoped session later). Default scope is
    # text + call so an invited guest can do both out of the box (the common
    # "reach me" case); the host narrows to text-only from the Guests list.
    guest_grant_token = ""
    if is_guest_claim:
        try:
            from augmentum.connect.contacts import local_did_for
            from augmentum.connect.guest_grant_store import create_grant
            inviter_id = invite["inviter_user_id"]
            grant = await create_grant(
                conn, host_user_id=inviter_id, host_did=local_did_for(inviter_id),
                guest_user_id=new_user.id, guest_did=local_did_for(new_user.id),
                scopes="text,call",
            )
            guest_grant_token = grant["token"]
        except Exception:
            log.warning("invite_claim_guest_grant_failed", exc_info=True)

    # The invite did its job — release any public tunnel ref it was holding so
    # the door closes the moment it's claimed (ref-counted, so a shared tunnel
    # stays up for other live invites). No-op for LAN/tailnet invites and when
    # no tunnel manager is registered. Best-effort.
    await _release_invite_reach(token)

    # Auto-login.
    session_token = await sm.create_session(
        new_user.id,
        ip_address=_get_ip(request),
        user_agent=request.headers.get("user-agent", ""),
        source="web",
    )
    payload: dict = {"user": new_user.to_public_dict()}
    if guest_grant_token:
        # The surface stores this to re-establish a scoped session on relaunch.
        payload["guest_grant_token"] = guest_grant_token
    response = JSONResponse(payload, status_code=201)
    return _set_session_cookie(response, session_token, request=request)
