"""HMAC-SHA1 ephemeral TURN credentials for the bundled coturn.

coturn supports a "shared-secret" credential mode (``--use-auth-secret``
+ ``--static-auth-secret=<SECRET>``) in which the username is
``<unix_expiry>:<opaque_metadata>`` and the password is
``base64(HMAC-SHA1(secret, username))``. The relay validates the pair
by recomputing the HMAC; no per-user database is needed.

This is the canonical way to hand short-lived credentials to peers
without sharing a long-lived ``user:pass`` everywhere. We use it for
both game-stream (per-session browser viewer) and Connect (per-call
peer signaling).

Why HMAC over static creds:

* Static ``augmentum:augmentumstream`` gets shipped to every browser
  the proxy talks to; once leaked, anyone on the network can relay
  through us indefinitely.
* Ephemeral creds expire (default 24h) so a leaked cred has a bounded
  blast radius.
* The opaque-metadata tail of the username is logged by coturn — we
  encode a stable identity hint (``user_id`` truncated) so the relay
  log is greppable for "which user used which allocation".

The secret is configured via ``AUGMENTUM_TURN_SECRET``. Both the proxy
(this module) and the coturn container (compose.calling.yaml) read it
from the same env var so they stay in sync.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from dataclasses import dataclass

# 24h is the standard ephemeral-credential TTL. Long enough that
# a single game-stream / Connect call almost never outlives its
# allocation; short enough that a leaked cred can't be replayed
# next week. Tunable by caller; this is just the default.
DEFAULT_TURN_CRED_TTL_SECONDS = 24 * 60 * 60

# When the operator hasn't set AUGMENTUM_TURN_SECRET we mirror
# the static-creds default posture (compose.game-stream.yaml used
# augmentum:augmentumstream): use a fixed dev value so dogfooding
# works out of the box, and document the override path. Production
# deployments MUST override via .env.
_DEFAULT_DEV_SECRET = "augmentum-turn-dev-secret-change-in-env"


@dataclass(frozen=True)
class TurnCredentials:
    """Ephemeral TURN cred pair.

    The browser/peer plugs ``username`` + ``password`` into its
    ``RTCIceServer`` config. ``expires_at`` is a unix timestamp the
    UI can use to schedule re-mint before allocations would
    otherwise drop on ICE restart.
    """

    username: str
    password: str
    expires_at: int

    def as_ice_server(self, turn_url: str) -> dict[str, str | list[str]]:
        """Render the pair as an ``RTCIceServer`` dict.

        ``turn_url`` is the full ``turn:host:port?transport=udp`` URI
        the relay listens on. Returned dict matches what browsers
        accept directly (no additional massaging).
        """

        return {
            "urls": [turn_url],
            "username": self.username,
            "credential": self.password,
        }


def turn_secret_from_env() -> str:
    """Read the shared secret from the environment.

    Falls back to the dev default when unset so local dogfooding
    works without operator setup. The compose file applies the same
    fallback so the proxy and coturn agree.
    """

    return os.environ.get("AUGMENTUM_TURN_SECRET", _DEFAULT_DEV_SECRET)


def mint_ephemeral(
    identity_hint: str,
    *,
    ttl_seconds: int = DEFAULT_TURN_CRED_TTL_SECONDS,
    secret: str | None = None,
    now: int | None = None,
) -> TurnCredentials:
    """Mint a fresh credential pair.

    ``identity_hint`` becomes the opaque tail of the username — coturn
    logs it but doesn't validate it. Pass a stable per-user or
    per-session string (a user_id, session_id, or call_id) so the
    relay log is greppable.

    ``ttl_seconds`` controls how long the cred is valid. The default
    is 24h; callers can shorten for short-lived flows.

    ``secret`` overrides ``AUGMENTUM_TURN_SECRET`` (mainly for tests).

    ``now`` overrides ``time.time()`` (also mainly for tests).
    """

    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")

    effective_secret = secret if secret is not None else turn_secret_from_env()
    if not effective_secret:
        raise ValueError("turn secret is empty")

    current = int(now if now is not None else time.time())
    expiry = current + int(ttl_seconds)

    # Sanitize the hint: coturn's username is whitespace-delimited in
    # its log format, and ':' is the separator between expiry and tail.
    # Drop characters that would confuse either.
    safe_hint = "".join(
        c for c in identity_hint if c.isalnum() or c in "-_"
    )[:32] or "anon"

    username = f"{expiry}:{safe_hint}"
    digest = hmac.new(
        effective_secret.encode("utf-8"),
        username.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    password = base64.b64encode(digest).decode("ascii")

    return TurnCredentials(
        username=username,
        password=password,
        expires_at=expiry,
    )
