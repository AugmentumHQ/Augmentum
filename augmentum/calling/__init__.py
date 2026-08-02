"""Calling substrate — shared WebRTC TURN/signaling infrastructure.

Owns the pieces that BOTH Connect (user-to-user calls/messages) and
game-stream (browser-to-container low-latency stream) need:

* TURN credential minting (``turn_credentials``) — HMAC-SHA1 ephemeral
  user/password pairs that coturn validates via ``--use-auth-secret``.
  No per-user state on the relay; secret is a single env var shared
  between the proxy and coturn.

Future additions land here as Connect grows:

* Signaling endpoint scaffold (Task #53 — ``signaling.py``).
* TURN-server health / capacity probes.
* Per-call lifecycle helpers shared by Connect + game-stream.

See ``docs/superpowers/specs/2026-06-01-connect-and-os-positioning-design.md``
for the broader architecture this slots into.
"""

from __future__ import annotations

from .turn_credentials import (
    DEFAULT_TURN_CRED_TTL_SECONDS,
    TurnCredentials,
    mint_ephemeral,
    turn_secret_from_env,
)

__all__ = [
    "DEFAULT_TURN_CRED_TTL_SECONDS",
    "TurnCredentials",
    "mint_ephemeral",
    "turn_secret_from_env",
]
