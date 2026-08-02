"""LiveKit room access tokens + reachability probe.

The LiveKit media plane is the default for Connect calls (see spec
``docs/superpowers/specs/2026-06-06-livekit-media-plane-design.md``).
A peer joins a LiveKit room by presenting a JWT scoped to that room;
this module mints the JWT and ships the URL + room name the client
needs to call ``room.connect(url, token)``.

It also exposes a cheap reachability probe so the invite-time
decision tree in ``call_routing`` can pick the P2P fallback when
LiveKit is down (rolling 30s cache so a busy invite path doesn't
hammer the SFU's health port).

Why this lives alongside ``turn_credentials.py``: same architectural
slot — short-lived credential minted server-side, handed to the
browser, validated by an external service. The two are independent
(LiveKit doesn't need TURN creds; the JWT carries its own auth) but
the codebase shape rhymes.

Env vars:
    LIVEKIT_URL          — wss://host:7880 the browser connects to
    LIVEKIT_API_KEY      — public key identifier
    LIVEKIT_API_SECRET   — HMAC secret used to sign the JWT
    LIVEKIT_HEALTH_URL   — http://host:7880 (HTTP for the probe;
                           optional — derived from LIVEKIT_URL when
                           absent)

For local dogfooding the dev defaults match ``compose.calling.yaml``
so the system works out of the box; production .env MUST override
the API key/secret.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# 2 hours is the JWT TTL ceiling. Most calls are <60min; the buffer
# absorbs late-joiners reusing a stale token and clock skew between
# the proxy and LiveKit. Tunable per-call but rarely needs to be.
DEFAULT_TOKEN_TTL_SECONDS = 2 * 60 * 60

# How long a reachability probe result is reused before re-checking.
# Long enough that a burst of invites in the same minute pays one
# probe; short enough that a freshly-restarted LiveKit is picked up
# within half a minute.
_REACHABILITY_CACHE_SECONDS = 30

# Health probe timeout. The probe is on the critical path of invite
# routing — keep it tight so a slow/dead LiveKit doesn't hang the
# whole invite handshake. 100ms is enough on loopback / LAN.
_REACHABILITY_TIMEOUT_SECONDS = 0.1

# Dev defaults — see header. Mirror the ``compose.calling.yaml`` set.
_DEFAULT_LIVEKIT_URL = "ws://localhost:7880"
_DEFAULT_API_KEY = "devkey"
_DEFAULT_API_SECRET = "augmentum-livekit-dev-secret-change-in-env"

# Room-name prefix for Connect calls. Keeps Connect rooms namespaced
# so future LiveKit consumers (game-stream cast surface, voice agents)
# can share the same SFU without colliding on call_ids.
ROOM_PREFIX = "call_"


@dataclass(frozen=True)
class LiveKitToken:
    """Bundle the UI plugs into ``room.connect(url, token)``."""

    token: str
    url: str
    room: str
    expires_at: int


# ── Env access ────────────────────────────────────────────────────


def livekit_url() -> str:
    return os.environ.get("LIVEKIT_URL", _DEFAULT_LIVEKIT_URL)


def livekit_api_key() -> str:
    return os.environ.get("LIVEKIT_API_KEY", _DEFAULT_API_KEY)


def livekit_api_secret() -> str:
    return os.environ.get("LIVEKIT_API_SECRET", _DEFAULT_API_SECRET)


def livekit_health_url() -> str:
    """Derive the HTTP base for the health probe.

    Override via ``LIVEKIT_HEALTH_URL`` for unusual topologies (e.g.
    a separate health port). Otherwise convert the ``wss://`` /
    ``ws://`` signaling URL to its ``http(s)://`` sibling.
    """

    explicit = os.environ.get("LIVEKIT_HEALTH_URL")
    if explicit:
        return explicit
    url = livekit_url()
    if url.startswith("wss://"):
        return "https://" + url[len("wss://"):]
    if url.startswith("ws://"):
        return "http://" + url[len("ws://"):]
    return url


def room_name_for(call_id: str) -> str:
    """Canonical LiveKit room name for a Connect call_id."""

    return f"{ROOM_PREFIX}{call_id}"


# ── JWT minting ────────────────────────────────────────────────────


def mint_call_token(
    *,
    call_id: str,
    user_did: str,
    display_name: str = "",
    ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS,
    api_key: str | None = None,
    api_secret: str | None = None,
    url: str | None = None,
) -> LiveKitToken:
    """Mint a JWT that grants this user join+publish+subscribe on the
    call's LiveKit room.

    The room name is derived from ``call_id`` so both peers compute
    the same value without server coordination.

    ``user_did`` becomes the LiveKit ``identity`` (visible to the
    other peer via ``room.remoteParticipants``). It is the canonical
    Connect identifier; we don't invent a separate LiveKit identity.

    ``api_key`` / ``api_secret`` / ``url`` are test overrides —
    production paths read from env. We don't take a ``now`` override
    because the SDK reads ``time.time()`` internally when computing
    the JWT ``exp`` claim, and divergence between the bundle's
    ``expires_at`` field and the JWT's actual exp would surface as
    UI showing "good for 2h" while the SFU rejects after a few
    minutes.
    """

    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")
    if not call_id:
        raise ValueError("call_id is required")
    if not user_did:
        raise ValueError("user_did is required")

    # Imported lazily so the module can be referenced without the
    # livekit-api dependency installed (e.g. on smoke paths that
    # only need the room-name helper).
    from livekit import api as lk_api  # type: ignore[import-not-found]

    effective_key = api_key if api_key is not None else livekit_api_key()
    effective_secret = api_secret if api_secret is not None else livekit_api_secret()
    effective_url = url if url is not None else livekit_url()
    if not effective_key or not effective_secret:
        raise ValueError("LiveKit API key and secret are required")
    # Refuse to mint a real call token using the in-source dev defaults —
    # anyone reaching the LiveKit port could otherwise forge a join JWT for
    # any room. setup.{sh,bat} auto-rotate both keys on fresh installs;
    # upgrade-in-place deployments missing .env entries would otherwise
    # silently fall through to these. Tests pass api_key+api_secret
    # explicitly, so the test path is unaffected.
    if effective_key == _DEFAULT_API_KEY or effective_secret == _DEFAULT_API_SECRET:
        raise ValueError(
            "LiveKit API key/secret are at the in-source dev defaults. "
            "Run setup.sh / setup.bat (or set LIVEKIT_API_KEY and "
            "LIVEKIT_API_SECRET in .env to fresh random values) before "
            "starting Connect calls.",
        )

    room = room_name_for(call_id)
    expires_at = int(time.time()) + int(ttl_seconds)

    grants = lk_api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )
    builder = lk_api.AccessToken(effective_key, effective_secret)
    builder = builder.with_identity(user_did)
    if display_name:
        builder = builder.with_name(display_name)
    # The SDK's AccessToken accepts a ttl on the builder via
    # ``with_ttl(timedelta)`` in 1.x; pass it through so the JWT
    # ``exp`` matches the value we report back to callers.
    from datetime import timedelta
    builder = builder.with_ttl(timedelta(seconds=int(ttl_seconds)))
    builder = builder.with_grants(grants)
    jwt = builder.to_jwt()

    return LiveKitToken(
        token=jwt,
        url=effective_url,
        room=room,
        expires_at=expires_at,
    )


# ── Reachability probe ────────────────────────────────────────────


# Module-level cache. (timestamp, ok) tuple. ``timestamp`` is the
# unix-second the probe ran; ``ok`` is its result. Looked up on every
# call to ``livekit_reachable`` so the invite path takes one cheap
# membership check 99% of the time.
_reachability_cache: tuple[float, bool] | None = None


async def livekit_reachable(
    *,
    health_url: str | None = None,
    now: float | None = None,
) -> bool:
    """Return True iff a recent probe says LiveKit is responding.

    Hits ``GET <health_url>/`` (LiveKit ships a built-in health
    endpoint that returns 200/OK on the HTTP port). Cached for
    ``_REACHABILITY_CACHE_SECONDS`` so the invite hot path doesn't
    pay an HTTP round-trip per call.

    Errors of any kind (timeout, refused, 5xx) flip the cache to
    ``False`` for the cache window — the invite handler will then
    decide P2P fallback.
    """

    global _reachability_cache
    current = now if now is not None else time.time()

    cached = _reachability_cache
    if cached is not None and (current - cached[0]) < _REACHABILITY_CACHE_SECONDS:
        return cached[1]

    target = (health_url or livekit_health_url()).rstrip("/")
    ok = False
    try:
        async with httpx.AsyncClient(timeout=_REACHABILITY_TIMEOUT_SECONDS) as client:
            resp = await client.get(target + "/")
            # LiveKit's HTTP port serves 200 on / when healthy. Any
            # non-2xx is treated as down — we don't want partial
            # availability handing out tokens for a SFU that can't
            # actually accept publishes.
            ok = 200 <= resp.status_code < 300
    except (TimeoutError, httpx.HTTPError, OSError):
        ok = False

    _reachability_cache = (current, ok)
    return ok


def _reset_reachability_cache() -> None:
    """Test-only: drop the cache so the next probe runs fresh."""

    global _reachability_cache
    _reachability_cache = None


# ── Room teardown ─────────────────────────────────────────────────


async def delete_room(
    *,
    call_id: str,
    api_key: str | None = None,
    api_secret: str | None = None,
    url: str | None = None,
) -> bool:
    """Tear down the LiveKit room for a hung-up call.

    Idempotent — deleting a non-existent room is treated as success
    so the hangup handler can call this unconditionally without
    needing to know if the room ever got created (e.g. invite was
    declined before either side connected).

    Returns ``True`` if the room was deleted or didn't exist;
    ``False`` if the API call errored for another reason.
    """

    if not call_id:
        return False

    from livekit import api as lk_api  # type: ignore[import-not-found]

    effective_key = api_key if api_key is not None else livekit_api_key()
    effective_secret = api_secret if api_secret is not None else livekit_api_secret()
    effective_url = url if url is not None else livekit_url()
    # The Room Service API speaks HTTP; convert the wss/ws URL.
    if effective_url.startswith("wss://"):
        effective_url = "https://" + effective_url[len("wss://"):]
    elif effective_url.startswith("ws://"):
        effective_url = "http://" + effective_url[len("ws://"):]

    room = room_name_for(call_id)
    client: Any | None = None
    try:
        client = lk_api.LiveKitAPI(
            url=effective_url,
            api_key=effective_key,
            api_secret=effective_secret,
        )
        await client.room.delete_room(lk_api.DeleteRoomRequest(room=room))
        return True
    except Exception as exc:  # noqa: BLE001
        # LiveKit returns NotFound for "room doesn't exist". Treating
        # any exception with "not found" / "does not exist" as
        # success — there's no shared exception type to catch in the
        # SDK across versions.
        msg = str(exc).lower()
        if "not found" in msg or "does not exist" in msg or "no such room" in msg:
            return True
        log.warning("livekit_delete_room_failed", call_id=call_id, err=str(exc))
        return False
    finally:
        if client is not None:
            try:  # noqa: SIM105 — cleanup, errors here are not actionable
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass
