"""Raw ASGI authentication middleware.

Uses raw ASGI (not BaseHTTPMiddleware) to support WebSocket upgrades.
Same pattern as _SecurityHeadersMiddleware in server.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.connect.reachability import tunnel_request_blocked
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.auth.session_manager import SessionManager

log = get_logger(__name__)


def _scope_host(scope: dict) -> str:
    """The RAW ``Host`` header of an ASGI request — deliberately ignoring
    ``X-Forwarded-Host``.

    This feeds the tunnel guard's is-this-tunnel-traffic check. cloudflared
    pins every tunneled request's REAL Host header to the sentinel
    (``--http-host-header``, see connect/tunnel_manager.py) — while
    ``X-Forwarded-Host`` is a plain request header the VISITOR controls
    and Cloudflare passes through untouched. Preferring it (the old
    behavior) let anyone on the tunnel send ``X-Forwarded-Host:
    other.host`` to dodge the sentinel match and reach the full app
    surface (/login, the API) through a tunnel scoped to one invite door.
    """
    for name, value in scope.get("headers", []):
        if name == b"host" and value:
            return value.decode("latin-1", errors="ignore").split(",", 1)[0].strip()
    return ""


def _scope_client_ip(scope: dict) -> str:
    """Originating client IP. Through cloudflared the real visitor IP is in
    ``Cf-Connecting-Ip``; fall back to the first ``X-Forwarded-For`` hop, then
    the raw ASGI client. Used to enforce the tunnel IP allowlist."""
    cf = xff = ""
    for name, value in scope.get("headers", []):
        if name == b"cf-connecting-ip" and value and not cf:
            cf = value.decode("latin-1", errors="ignore").strip()
        elif name == b"x-forwarded-for" and value and not xff:
            xff = value.decode("latin-1", errors="ignore").split(",", 1)[0].strip()
    if cf or xff:
        return cf or xff
    client = scope.get("client")
    return client[0] if isinstance(client, tuple | list) and client else ""

# Paths that don't require authentication
_PUBLIC_PATHS = {
    "/",
    "/ui",
    "/api/auth/login",
    "/api/auth/setup",
    "/api/auth/status",
    # Front-gate forward_auth verdict. Called by Caddy (not a logged-in
    # browser) ahead of the upstream; it validates the forwarded session
    # cookie itself, so it MUST be unauthenticated here or the middleware
    # 401s before the handler can issue its 200/302/403. See gate_routes.py.
    "/api/gate/verify",
    "/api/version",
    # UI shell content digest — the native Android app compares this to its
    # baked bundle to decide whether to serve the shell from the APK. Exposes
    # only a SHA-256 of the already-public /ui/ files (no secrets), and must be
    # reachable for the version handshake even before a session exists.
    "/api/ui-version",
    "/favicon.ico",
    "/.well-known/security.txt",
    # Push service worker — registered from the origin root so it can be
    # the canonical SW for Web Push delivery. The browser fetches this
    # without sending the session cookie during register/update, so it
    # MUST be unauthenticated or the SW never installs. The file is the
    # same static JS that lives at /ui/notification-sw.js; the root path
    # is just a re-route. No secrets — it's a thin push-event handler
    # that calls showNotification on the encrypted payload the OS push
    # service already authenticated via VAPID.
    "/notification-sw.js",
    # Fabric pair bootstrap — the caller is another augmentum node, not
    # a logged-in browser. Same pattern as /api/cast/pair/start: the
    # cryptographic check IS the auth (signed PairRequest envelope with
    # Ed25519 signature + fingerprint hint + timestamp window). Gating
    # this with a session cookie would 401 every legitimate pair attempt
    # because the remote peer has no session. The handler in
    # fabric_routes.py:fabric_pair validates the envelope before
    # persisting anything to fabric_nodes.
    "/api/fabric/pair",
    # Fabric discovery probe — returns this node's public identity
    # envelope (node_id, fingerprint, public key). No secrets; the
    # operator confirms the fingerprint out-of-band before pairing.
    # Public for the same reason as /pair: the LAN sweep that finds
    # this endpoint has no session cookie.
    "/api/fabric/hello",
    # Fabric persistent peer WebSocket. The handler in fabric_routes
    # accepts the upgrade and immediately demands a signed hello
    # envelope as the first frame — the Ed25519 signature against the
    # pinned pubkey IS the auth. Routing this WS through the ticket
    # check would timeout every peer reconnect because fabric peers
    # don't have session tickets (they're servers talking to servers,
    # not browsers). See ``fabric_routes.fabric_connect`` docstring.
    "/api/fabric/connect",
    # Community install preview — entry point for "Open in Augmentum" links
    # from augmentumhq.com. The handler does its own auth check and
    # redirects to /login?next=... when no session exists. Without this
    # exemption, the cross-origin nav from augmentumhq.com gets 401'd
    # before the handler ever runs. The POST endpoint that performs the
    # actual install (/api/community/install) is NOT exempt — it remains
    # auth-gated like every other state-changing API.
    "/community-install",
}

_PUBLIC_PREFIXES = (
    "/ui/",
    # NOTE: /api/auth/status is intentionally NOT a prefix here — it lives in
    # _PUBLIC_PATHS as an EXACT match. A startswith prefix would also exempt a
    # future "/api/auth/status_secret" route from auth (a silent landmine), and
    # no real "/api/auth/status/<sub>" path exists. Keep new auth-status-like
    # public paths as exact entries unless a true sub-tree needs exemption.
    # Vendored 3rd-party static asset mount (SillyTavern BVH motion pack).
    # Public-domain content used by the scene-test mockup's animation
    # library — no auth required since /ui/ already serves the page that
    # consumes it without auth.
    "/bvh-library/",
    # Cast blob endpoint — TVs/speakers fetch from here without auth
    # cookies. Access is gated by short-lived single-user-bound tokens
    # validated inside the handler (see augmentum/devices/cast_tokens.py),
    # so this prefix being public is part of the design, not a gap.
    "/api/cast/blob/",
    # Cast render-output endpoint — receiver apps (TVs, future cast
    # surfaces) fetch the rendered output bytes. Gated by short-lived
    # tokens issued at render time (see augmentum/cast/output_store.py).
    "/api/cast/render-output/",
    # Cast pair bootstrap — the receiver page (often pre-auth, e.g. a
    # TV browser without a session cookie) creates a pair record and
    # polls until a phone scans the QR. The QR image itself is public
    # (knowledge of the code is the only auth surface — anyone with
    # the code can already see the receiver's pair page). /approve/
    # is NOT exempt — claiming a pair requires the phone-side session
    # cookie, which is where the actual authorization happens.
    "/api/cast/pair/start",
    "/api/cast/pair/poll/",
    "/api/cast/pair/qr/",
    # Mobile pair bootstrap. The authenticated web UI creates/approves
    # the pair, but the Android app starts without a session and needs
    # to claim, poll, and redeem the one-shot grant.
    "/api/auth/pair/claim/",
    "/api/auth/pair/poll/",
    "/api/auth/pair/finish",
    "/api/auth/pair/qr/",
    # Invite preview + claim (Connect open-access onboarding). The claimant
    # has no account yet, so these must be public; the high-entropy token in
    # the URL IS the credential (same model as cast guest invites). SINGULAR
    # ``/api/auth/invite/`` — distinct from the admin-gated plural
    # ``/api/auth/invites`` management routes. The claim handler does its own
    # validation (token consume is atomic, username reserved/unique checks).
    "/api/auth/invite/",
    # Connect guest-pass session bootstrap ONLY. The saved PWA presents its
    # durable grant token (the credential) to mint a scoped guest session — it
    # has no cookie yet on a fresh launch. Scoped to ``/session`` exactly: other
    # guest endpoints (e.g. /guest/ping) need the resolved guest session and so
    # must stay auth-gated. Distinct from the plural ``/api/connect/guests``
    # host-management routes.
    "/api/connect/guest/session",
    # Guest comms portal registration — an external_guest invitee has NO
    # account yet, so this MUST be public; the high-entropy invite token in
    # the path IS the credential (same model as /api/auth/invite/). The
    # handler (portal_routes.py:portal_register) validates + consumes the
    # invite atomically before creating the PENDING guest account. Only the
    # ``/register/`` sub-path is public — the admin review routes
    # (/api/portal/pending, /registrations/{id}/confirm|deny) stay auth-gated,
    # and the guest self-check (/api/portal/me) is reachable only to a
    # logged-in guest via _GUEST_ALLOWED_PREFIXES below.
    "/api/portal/register/",
    # Guest gateway (see 2026-07-16 spec). /gateway returns the instance's
    # PUBLIC seal-key bundle, signed by the identity the invite QR pinned —
    # no secrets, and the portal must fetch it pre-auth to build its first
    # envelope. /env is the enveloped-dispatch endpoint: the Ed25519 device
    # signature on every envelope IS the auth (same model as /api/fabric/pair),
    # verified in the handler against the device key the host confirmed.
    "/api/portal/gateway",
    "/api/portal/env",
    # Invite QR image — GET /api/invite/{token}/qr.png renders the join link
    # as a scannable PNG. Public for the same reason as /api/auth/invite/:
    # holding the token already lets you decode it, and the scanner (a phone
    # camera, face-to-face) has no session cookie. Token is high-entropy and
    # in the path; the handler only emits a QR of the public join URL.
    "/api/invite/",
    # Cast invite QR — receiver fetches by token. No auth needed: the
    # token is the credential, only someone who already knows it can
    # decode the matching QR. Mirrors /api/cast/pair/qr/ shape.
    "/api/cast/invite/qr/",
    # Cast guest-join page (UI surface). Guests scan the QR on the TV
    # and land here without an Augmentum account; the page reads its
    # token from the URL and opens the WS via the join_token path.
    "/ui/cast-guest-join/",
    # Cast couch co-op guest identify / claim. No auth — the invite
    # token in the request body IS the credential, scoping the call
    # to one host's data. See cast_routes.cast_guest_identify and
    # cast_guest_claim.
    "/api/cast/guest/identify",
    "/api/cast/guest/claim",
    "/api/cast/guest/forget-device",
    # Establish-session redeems a ws_token (issued by an authenticated
    # phone-side /approve/) for a real auth_sessions row delivered via
    # Set-Cookie. The ws_token itself is the credential here; this
    # endpoint sits *before* the cookie exists, so it must be public.
    "/api/cast/pair/establish-session",
    # Stream-auth redeem: rendering container's Chrome boots cookie-
    # less, follows a one-shot redeem URL, server validates the
    # token + sets the cookie + 303s to the actual surface. The
    # redeem token IS the credential, single-use, ~30s TTL.
    "/api/cast/stream-auth/redeem",
    # Android TV self-update: the receiver APK polls version + downloads
    # via plain HttpsURLConnection at app launch, BEFORE the WebView has
    # any session cookie (cookies live in the WebView's CookieManager,
    # not shared with HttpsURLConnection). Public so the receiver kiosk
    # can update itself without a paired session. Integrity guard is
    # Android's package-signature check on install — an APK that doesn't
    # match the installed app's signing key is rejected by the OS.
    "/api/cast/pair/android-tv/",
    # Surface receivers are public TV/browser pages. The handler validates
    # a short-lived token and only grants the scopes embedded in it.
    "/api/surface-public/",
    # Cast input container-side WS. The in-container
    # cast-input-bridge.py daemon dials this with ?token=<x> matching
    # the session row's cast_input_token. No session cookie reaches
    # the container; the bridge_token IS the credential, validated
    # via hmac.compare_digest inside the handler. Same pattern as
    # /api/cast/blob/, /api/cast/render-output/, /api/cast/pair/.
    "/api/cast/input/container-ws/",
    # Cast origin-proxy — receiver-side TV browser fetches the
    # rewritten game assets. The path-embedded ``cgp_*`` token is
    # the credential, validated against ProxySessionStore inside the
    # handler. Same pattern as /api/cast/blob/ + render-output —
    # cross-origin iframed pages can't carry our session cookie
    # reliably anyway, so the token has to be the auth surface.
    "/api/cast/game-proxy/",
)

# Admin-only path prefixes
_ADMIN_PREFIXES = (
    "/api/auth/users",
    "/api/auth/audit",
    # Invite MANAGEMENT (mint / list / revoke). Note the trailing form: this
    # is the plural ``/api/auth/invites`` — it does NOT match the PUBLIC
    # singular ``/api/auth/invite/`` preview+claim routes below, which carry
    # their own credential (the high-entropy token).
    "/api/auth/invites",
)

# Guest surface gate. A ``role='guest'`` account is a comms-only invitee — the
# whole point of the role is that their session can drive ONLY the Connect
# surface (text/call the host), not the rest of the app. The /ui/ static shell
# is public and inert, so enforcement lives here at the data layer: a guest may
# reach the allow-listed prefixes and NOTHING else (deny-by-default). Everything
# the full app needs — /api/coder, /api/chat, /api/config, /api/library, /ws/voice,
# cast, image gen … — is refused, so a guest's cookie can't masquerade as a member.
_GUEST_ALLOWED_PREFIXES = (
    "/api/auth/",     # status / logout / ws-ticket / me — session plumbing
    "/api/connect/",  # messaging, calling, signaling, guest session/ping
    "/api/notify/",   # incoming-call wake
    "/api/health",    # boot/health probes (inert)
    # Guest portal self-check ONLY ("am I confirmed + reachable yet?"). The
    # logged-in guest's portal page polls this. Scoped to ``/me`` exactly so
    # the host-only review routes (/api/portal/pending, /registrations/...)
    # remain off-limits to a guest cookie — deny-by-default holds.
    "/api/portal/me",
)
# Carve-outs INSIDE the allowed prefixes that a guest still may not reach:
# browsing/searching the member directory, or the host-only guest-management
# routes. The routing-layer ACL already stops a guest messaging anyone but the
# host; this also denies the read surface.
_GUEST_DENIED_PREFIXES = (
    "/api/connect/directory",
    "/api/connect/search",
    "/api/connect/guests",  # plural = host management (NOT /guest/ singular)
    # Under the allowed /api/auth/ umbrella but OFF-LIMITS to a comms-only
    # guest: minting a persistent API key (a credential that outlives the
    # revocable session) and pairing mobile/devices (extends the foothold).
    "/api/auth/keys",
    "/api/auth/pair",
)


class AuthMiddleware:
    """ASGI middleware that validates auth tokens and attaches user to scope."""

    def __init__(self, app, session_manager: SessionManager | None = None) -> None:
        self.app = app
        self._session_manager = session_manager

    def _is_public(self, path: str) -> bool:
        """Check if path is public (no auth required)."""
        if path in _PUBLIC_PATHS:
            return True
        for prefix in _PUBLIC_PREFIXES:
            if path.startswith(prefix):
                return True
        return False

    async def _env_dispatch_user(self, scope: dict, sm):
        """The guest user for an in-process guest-gateway dispatch, or None.

        Trusts ``x-augmentum-env-user`` ONLY when ``x-augmentum-env-secret``
        matches the per-boot random secret minted by the /api/portal/env
        handler (``app.state.guest_env_secret``). Constant-time compare;
        any mismatch or absence = None (normal auth continues).
        """
        headers = dict(scope.get("headers") or [])
        secret = headers.get(b"x-augmentum-env-secret", b"").decode("latin-1")
        user_id = headers.get(b"x-augmentum-env-user", b"").decode("latin-1")
        if not secret or not user_id:
            return None
        app = scope.get("app")
        expected = getattr(getattr(app, "state", None), "guest_env_secret", "")
        if not expected:
            return None
        import hmac
        if not hmac.compare_digest(secret, expected):
            log.warning("guest_env_secret_mismatch")
            return None
        user = await sm.get_user_by_id(user_id)
        if user is None or not user.is_active:
            return None
        return user

    def _guest_denied(self, user, path: str) -> bool:
        """True if a ``role='guest'`` user must be refused ``path``.

        Deny-by-default outside the Connect comms surface — the enforcement that
        makes the guest role real rather than cosmetic. No-op for any other role.
        """
        if getattr(user, "role", "") != "guest":
            return False
        if any(path.startswith(p) for p in _GUEST_DENIED_PREFIXES):
            return True
        return not any(path.startswith(p) for p in _GUEST_ALLOWED_PREFIXES)

    def _extract_token(self, scope: dict) -> str | None:
        """Extract auth token from headers or cookies."""
        from http.cookies import SimpleCookie

        headers = dict(scope.get("headers", []))

        auth = headers.get(b"authorization", b"").decode("latin-1", errors="ignore")
        api_key = headers.get(b"x-api-key", b"").decode("latin-1", errors="ignore").strip()

        # 1. x-api-key has highest priority WHEN it's an Augmentum-shaped
        # key (sk-aug-*). Reason: Anthropic-shape clients like Claude Code
        # send BOTH ``Authorization: Bearer <cached-oauth-from-their-login>``
        # AND ``x-api-key: <user-set-key>``. If we picked Bearer first, the
        # cached OAuth token wins and 401s every request. Preferring an
        # sk-aug x-api-key over any Bearer fixes CC without changing
        # behavior for OpenWebUI / Cursor / etc. (which only send Bearer).
        if api_key.startswith("sk-aug-"):
            return api_key

        # 2. Authorization: Bearer <token>
        if auth.startswith("Bearer "):
            return auth[7:].strip()

        # 3. x-api-key fallback for non-sk-aug values (rare; permissive
        # in case a forward setup proxies through unusual keys).
        if api_key:
            return api_key

        # 4. Cookie: augmentum_session=<token>. Use SimpleCookie for proper
        # parsing — quoted values, escaped chars, and multiple cookies in
        # one header all worked in the hand-rolled startswith() parser
        # only for the common case; SimpleCookie handles the corner cases.
        cookie_header = headers.get(b"cookie", b"").decode("latin-1", errors="ignore")
        if cookie_header:
            try:
                jar = SimpleCookie()
                jar.load(cookie_header)
                morsel = jar.get("augmentum_session")
                if morsel and morsel.value:
                    return morsel.value.strip()
            except Exception:
                # Malformed Cookie header: drop to None rather than 500.
                return None

        return None

    def _ws_origin_allowed(self, scope: dict) -> bool:
        """Validate that a WebSocket Origin matches the request Host.

        Cookie-fallback WS branches (preview proxy, cast receiver, voice
        fanout) accept the session cookie as auth. Without an Origin check
        a malicious page can open a WS to our server in the user's browser
        and ride their cookie — same risk shape as CSRF for HTTP. Same-
        origin policy is enforced by matching the Origin scheme+host+port
        against the request's Host header. Missing Origin (non-browser
        clients) is allowed because those callers can't be CSRF-victimized.
        """
        headers = dict(scope.get("headers", []))
        origin = headers.get(b"origin", b"").decode("latin-1", errors="ignore").strip()
        if not origin:
            # Non-browser client (or browser stripped Origin per policy).
            # Browsers always send Origin on WS — non-browser callers
            # aren't subject to CSRF, so allow.
            return True
        host = headers.get(b"host", b"").decode("latin-1", errors="ignore").strip()
        if not host:
            # No Host to compare against — be conservative and reject.
            return False
        from urllib.parse import urlparse
        try:
            parsed = urlparse(origin)
        except Exception:
            return False
        origin_host = parsed.netloc or ""
        return origin_host == host

    def _extract_ticket(self, query_string: str) -> str | None:
        """Extract WS ticket from query string."""
        return self._extract_query_value(query_string, "ticket")

    @staticmethod
    def _extract_query_value(query_string: str, key: str) -> str | None:
        """Return the value of ``key`` from a raw query string, or None.

        Tolerant of multiple occurrences (takes the first) and of
        missing values. Doesn't URL-decode — callers operate on
        opaque tokens where decoding is a no-op.
        """
        if not query_string or not key:
            return None
        needle = f"{key}="
        for part in query_string.split("&"):
            if part.startswith(needle):
                return part[len(needle):]
        return None

    async def __call__(self, scope, receive, send):
        """ASGI entry point."""
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # If a prior middleware already authenticated this request
        # (FabricPeerMiddleware for cross-peer dispatch), honour its
        # decision and pass through. The outer middleware is responsible
        # for verifying the peer signature + looking up the User.
        # Without this check, AuthMiddleware would still demand a
        # Bearer token from the peer request and 401 even though the
        # peer signature was valid.
        if scope.get("user") is not None:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")

        # Public invite-tunnel guard — runs BEFORE the public-path bypass. When a
        # request arrives via the ephemeral public tunnel (the sentinel Host),
        # it may reach ONLY the invite door; everything else (even /login) 404s,
        # so an internet visitor can't see anything but the invite the link was
        # for. A no-op when no tunnel is active. See connect/reachability.py.
        if tunnel_request_blocked(_scope_host(scope), path, _scope_client_ip(scope)):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008, "reason": "Not found"})
            else:
                await self._send_404(send)
            return

        sm = self._session_manager
        if not sm:
            app = scope.get("app")
            if app and hasattr(app, "state"):
                sm = getattr(app.state, "session_manager", None)

        # Public paths (login, setup, status, static) always bypass auth —
        # auth_status needs to work when the DB is down so the UI can report it.
        if self._is_public(path):
            await self.app(scope, receive, send)
            return

        # Non-public path but no session manager = degraded mode (DB failure
        # at startup, or SM never initialized). Fail closed: deny rather than
        # fall through, which would expose user-scoped endpoints as if auth
        # were disabled and return all unscoped rows (user_id="").
        if not sm:
            log.warning("auth_unavailable_denied", path=path, scheme=scope["type"])
            if scope["type"] == "websocket":
                await send({
                    "type": "websocket.close",
                    "code": 1011,
                    "reason": "Auth unavailable",
                })
            else:
                await self._send_503(send)
            return

        # Guest-gateway internal dispatch (HTTP only). /api/portal/env has
        # already authenticated the caller cryptographically (Ed25519 device
        # signature on the envelope, device bound to the guest at confirm)
        # and re-dispatches the inner request in-process, carrying a per-boot
        # random secret that never leaves the process — unforgeable from
        # outside. The guest deny-by-default lists still apply: the envelope
        # widens nothing, it only replaces the cookie as the credential.
        if scope["type"] == "http":
            env_user = await self._env_dispatch_user(scope, sm)
            if env_user is not None:
                if self._guest_denied(env_user, path):
                    await self._send_403(send)
                    return
                scope["user"] = env_user
                await self.app(scope, receive, send)
                return

        # WebSocket — validate ticket
        if scope["type"] == "websocket":
            qs = scope.get("query_string", b"").decode("latin-1", errors="ignore")
            ticket = self._extract_ticket(qs)
            if ticket:
                user_id = sm.validate_ws_ticket(ticket)
                if user_id:
                    user = await sm.get_user_by_id(user_id)
                    if user and user.is_active:
                        if self._guest_denied(user, path):
                            await send({
                                "type": "websocket.close", "code": 1008,
                                "reason": "Forbidden",
                            })
                            return
                        scope["user"] = user
                        await self.app(scope, receive, send)
                        return
            # API-key auth for NON-BROWSER WS clients (the ACP editor bridge).
            # Browsers cannot set an Authorization/x-api-key header on a
            # WebSocket, so an sk-aug key here can only come from a real CLI/
            # server client — no cookie is in play, so this is NOT CSRF-prone
            # the way the cookie fallback below is. We accept ONLY api-key-shaped
            # tokens: a browser's session cookie is not sk-aug-shaped and still
            # requires a ws-ticket, so the ticket-only contract is preserved.
            ws_token = self._extract_token(scope)
            if ws_token:
                from augmentum.auth.api_keys import is_api_key
                if is_api_key(ws_token):
                    akm = getattr(
                        getattr(scope.get("app"), "state", None),
                        "api_key_manager", None,
                    )
                    key_user = await akm.validate(ws_token) if akm is not None else None
                    if key_user and key_user.is_active:
                        if self._guest_denied(key_user, path):
                            await send({
                                "type": "websocket.close", "code": 1008,
                                "reason": "Forbidden",
                            })
                            return
                        scope["authed_via_api_key"] = True
                        scope["user"] = key_user
                        await self.app(scope, receive, send)
                        return
            # Same-origin cookie fallback for the coder dev-server preview
            # proxy. The dev server's HMR client opens its own WS to a
            # path it picks (/__hmr, /sockjs, etc.) and has no way to
            # mint a ticket — but cookies travel with same-origin WS
            # automatically. Restricted to the preview prefix so the
            # ticket-only contract still holds for terminal/etc.
            #
            # Same fallback for cast receivers: the browser-tab receiver
            # is loaded as a static page under /ui/cast-receiver/ and
            # opens its WS straight to /api/cast/receiver/ws. A separate
            # ticket roundtrip on top of the existing session cookie is
            # pure friction. Native TV shells will land via a distinct
            # pairing-token flow, not this endpoint, so this fallback
            # doesn't leak ticket-only guarantees elsewhere.
            cookie_fallback_prefixes = (
                "/api/coder/preview/",
                "/api/cast/receiver/",
                # Voice fanout subscribers: cast-vrm surface (running
                # in the receiver iframe) opens a WS here to receive
                # TTS audio + visemes. Same-origin cookie carries auth.
                "/api/voice/sessions/",
            )
            if any(path.startswith(p) for p in cookie_fallback_prefixes):
                # CSRF defense: the cookie-fallback branch accepts the
                # browser's session cookie as auth, which means a third-
                # party origin could open this WS with the user's cookie
                # in flight. Require Origin == Host (or no Origin —
                # non-browser clients). The ticket branch above is
                # ticket-only and not vulnerable.
                if not self._ws_origin_allowed(scope):
                    log.warning(
                        "ws_cookie_fallback_origin_rejected", path=path,
                    )
                    await send({
                        "type": "websocket.close", "code": 4001,
                        "reason": "Origin not allowed",
                    })
                    return
                token = self._extract_token(scope)
                if token:
                    user = await sm.validate_token(token)
                    if user and user.is_active:
                        # Cookie-fallback WS prefixes are coder/cast/voice — a
                        # guest's cookie must never ride these.
                        if self._guest_denied(user, path):
                            await send({
                                "type": "websocket.close", "code": 1008,
                                "reason": "Forbidden",
                            })
                            return
                        scope["user"] = user
                        await self.app(scope, receive, send)
                        return
            # Pair-token fallback for cast receivers. Receivers that
            # can't carry a session cookie (TV browsers, native
            # shells) bootstrap via the QR pairing flow — the phone-
            # side approve issues a single-use wsp_* token, the
            # receiver passes it as ?token=<wsp_*>. We consume the
            # token here; the corresponding user becomes the WS owner.
            if path.startswith("/api/cast/receiver/"):
                pair_token = self._extract_query_value(qs, "token")
                if pair_token and pair_token.startswith("wsp_"):
                    app_obj = scope.get("app")
                    pair_store = getattr(getattr(app_obj, "state", None),
                                          "pair_store", None) if app_obj else None
                    if pair_store is not None:
                        record = pair_store.consume_token(pair_token)
                        if record is not None:
                            user = await sm.get_user_by_id(record.user_id)
                            if user and user.is_active:
                                scope["user"] = user
                                await self.app(scope, receive, send)
                                return
            # Invite-token fallback for cast couch co-op. Guest phones
            # scan a QR on the TV → land on the cast-guest-join page
            # → open ``/api/cast/input/ws?join_token=wsi_*``. The token
            # encodes the host's session and user; resolve, set the
            # user as the host (the guest's input legally counts as
            # the host's). Route handler does the slot bookkeeping.
            # Single-use semantics are NOT applied here — the route
            # claims the slot via cast_invite_store.claim instead, so
            # one token can admit ``max_slots`` distinct guests.
            if path == "/api/cast/input/ws":
                join_token = self._extract_query_value(qs, "join_token")
                if join_token and join_token.startswith("wsi_"):
                    app_obj = scope.get("app")
                    invite_store = getattr(
                        getattr(app_obj, "state", None),
                        "cast_invite_store", None,
                    ) if app_obj else None
                    if invite_store is not None:
                        record = invite_store.get(join_token)
                        if record is not None:
                            user = await sm.get_user_by_id(record.host_user_id)
                            if user and user.is_active:
                                scope["user"] = user
                                # Stash the resolved invite on the scope
                                # so the route handler can claim+attach
                                # without re-looking it up.
                                scope["augmentum.cast.invite"] = record
                                await self.app(scope, receive, send)
                                return
            # Reject WS — send close frame
            await send({"type": "websocket.close", "code": 4001, "reason": "Unauthorized"})
            return

        # HTTP — validate token. ``sk-aug-`` keys come from external
        # OpenAI-compatible clients (no browser session); everything
        # else is treated as an opaque session token.
        token = self._extract_token(scope)
        if not token:
            await self._send_401(send)
            return

        from augmentum.auth.api_keys import is_api_key

        user = None
        if is_api_key(token):
            akm = None
            app = scope.get("app")
            if app and hasattr(app, "state"):
                akm = getattr(app.state, "api_key_manager", None)
            if akm is not None:
                user = await akm.validate(token)
            # Mark the auth method so the OpenAI/Ollama surfaces can default
            # external tools (Open WebUI, Cursor, …) to clean DIRECT passthrough:
            # they authenticate with an sk-aug key (not a browser session) and
            # shouldn't route through Augmentum's mode classifier + memory
            # harness unless they explicitly opt in via a model prefix.
            if user is not None:
                scope["authed_via_api_key"] = True
        else:
            user = await sm.validate_token(token)

        if not user:
            await self._send_401(send)
            return

        # Admin-only check
        for prefix in _ADMIN_PREFIXES:
            if path.startswith(prefix) and not user.is_admin:
                await self._send_403(send)
                return

        # Guest surface gate — a comms-only invitee can't reach the full app.
        if self._guest_denied(user, path):
            await self._send_403(send)
            return

        # Attach user to scope (accessed as request.state.user in handlers)
        scope["user"] = user
        await self.app(scope, receive, send)

    async def _send_401(self, send):
        """Send 401 Unauthorized response."""
        body = b'{"error":"Unauthorized"}'
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode()],
            ],
        })
        await send({"type": "http.response.body", "body": body})

    async def _send_404(self, send):
        """Send 404 — used to hide non-invite paths on a public tunnel host."""
        body = b'{"error":"Not found"}'
        await send({
            "type": "http.response.start",
            "status": 404,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode()],
            ],
        })
        await send({"type": "http.response.body", "body": body})

    async def _send_403(self, send):
        """Send 403 Forbidden response."""
        body = b'{"error":"Forbidden"}'
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode()],
            ],
        })
        await send({"type": "http.response.body", "body": body})

    async def _send_503(self, send):
        """Send 503 when the auth system is unavailable (DB down at startup)."""
        body = b'{"error":"Auth system unavailable"}'
        await send({
            "type": "http.response.start",
            "status": 503,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode()],
            ],
        })
        await send({"type": "http.response.body", "body": body})
