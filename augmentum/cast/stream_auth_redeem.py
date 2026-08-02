"""One-shot redemption tokens for server-side rendering Chrome.

Each ``browser-stream`` cast spawns a Chrome kiosk inside a Docker
container. That Chrome has a fresh user-data-dir → zero cookies. The
rendering route mints a real ``auth_sessions`` row for the casting
user, hands the raw session token to this store keyed by a short
single-use redeem token, and points Chrome at
``/api/cast/stream-auth/redeem?t=<redeem>``. The handler validates
+ consumes the redeem token, returns ``Set-Cookie augmentum_session=…``
plus a 302 to the original ``next`` URL.

After that one round-trip Chrome holds the cookie and the rendering
surface can hit ``/api/avatar/for-session``, ``/api/voice/...``, the
companion runtime — everything that needs auth — exactly like the
phone-side UI does.

Why a separate store (not pair_store):

  * Different lifecycle: pair tokens live ~120s and are scoped to
    receiver bootstrap; redeem tokens are even shorter (~30s, just
    enough for Chrome to follow the redirect on container start).
  * Different threat shape: pair binds a receiver to a user; redeem
    smuggles a user session into a rendering container. Distinct
    invariants → distinct module.
  * Smaller blast radius if we ever rip one out.

In-memory only — same rationale as PairStore. A restart kills every
rendering container anyway, so cross-restart redemption buys nothing.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# 30 seconds is plenty for Chrome cold-boot → first HTTP fetch. Longer
# windows just widen the redeem-leak risk.
_DEFAULT_TTL_S: float = 30.0
_MAX_ACTIVE: int = 64


@dataclass(slots=True)
class _RedeemRecord:
    session_token: str
    user_id: str
    next_url: str
    expires_at: float
    consumed: bool = False


class StreamAuthRedeemStore:
    """Process-local store of one-shot redemption tokens."""

    def __init__(
        self,
        *,
        default_ttl_s: float = _DEFAULT_TTL_S,
        max_active: int = _MAX_ACTIVE,
    ) -> None:
        self._records: dict[str, _RedeemRecord] = {}
        self._ttl = max(5.0, float(default_ttl_s or _DEFAULT_TTL_S))
        self._max_active = max(1, int(max_active or _MAX_ACTIVE))

    def mint(
        self, *, session_token: str, user_id: str, next_url: str,
    ) -> str:
        """Create a redeem token wrapping ``session_token``."""
        self._prune()
        token = f"rd_{secrets.token_hex(16)}"
        self._records[token] = _RedeemRecord(
            session_token=session_token,
            user_id=user_id,
            next_url=next_url,
            expires_at=time.time() + self._ttl,
        )
        return token

    def redeem(self, token: str) -> _RedeemRecord | None:
        """Validate + mark consumed. Returns the record on success,
        None on unknown / expired / already-consumed.
        """
        if not token:
            return None
        record = self._records.get(token)
        if record is None:
            return None
        if record.consumed:
            return None
        if record.expires_at <= time.time():
            self._records.pop(token, None)
            return None
        record.consumed = True
        log.info("cast_stream_auth_redeemed", user_id=record.user_id)
        return record

    def _prune(self) -> None:
        now = time.time()
        expired = [
            t for t, r in self._records.items()
            if r.consumed or r.expires_at <= now
        ]
        for t in expired:
            self._records.pop(t, None)
        if len(self._records) > self._max_active:
            ordered = sorted(self._records.items(), key=lambda kv: kv[1].expires_at)
            for t, _ in ordered[: len(self._records) - self._max_active]:
                self._records.pop(t, None)
