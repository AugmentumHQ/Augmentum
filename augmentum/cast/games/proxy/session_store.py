"""Per-cast proxy session token store.

Mirrors :mod:`augmentum.cast.invite_store` — short-lived, in-memory,
self-pruning. A session is the credential that authorizes proxy
requests; the token itself IS the auth (path-embedded, no cookies
needed because cross-origin embedded iframes don't carry our session
cookie reliably anyway).

Lifecycle:

  - Minted by ``OriginProxyStrategy.prepare`` at cast time.
  - Bound to (user_id, receiver_id, title_id, source_origin, expires_at).
  - Looked up by /api/cast/game-proxy/{token}/{path} to gate every
    request + scope the asset cache.
  - Expires after TTL; user logout / cast end revokes early.

Security invariants:

  - Token is opaque + cryptographically random (24 hex bytes).
  - Source_origin pin means a stolen token can only fetch from the
    origin it was minted for — can't pivot to internal IPs or other
    games.
  - User_id pin means cross-tenant token reuse is rejected.
  - Receiver_id pin means a token from one TV can't be reused on
    another.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_DEFAULT_TTL_S = 3600.0           # 1 hour — covers a typical play session
_MAX_ACTIVE_RECORDS = 256


def _generate_token() -> str:
    """Mint a ``cgp_<24hex>`` token. ``cgp`` = Cast Game Proxy."""
    return f"cgp_{secrets.token_hex(12)}"


def _normalise_origin(url: str) -> str:
    """Return scheme://host[:port] for ``url``. Empty on malformed."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
    except (ValueError, AttributeError):
        return ""
    if not parsed.scheme or not parsed.netloc:
        return ""
    if parsed.scheme not in ("http", "https"):
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


@dataclass(slots=True)
class ProxySession:
    """One active proxy binding.

    A session is the credential. Lookup is by token; the bound fields
    are *verified* at use time so a leaked token can't reach a different
    origin / title / user.
    """

    token: str
    user_id: str
    receiver_id: str
    title_id: str
    source_origin: str           # e.g. https://example.com
    source_base_url: str         # canonical entry URL (preserves path)
    expires_at: float
    revoked: bool = False

    def is_expired(self, *, now: float | None = None) -> bool:
        return (now or time.time()) >= self.expires_at

    def is_active(self, *, now: float | None = None) -> bool:
        return not self.revoked and not self.is_expired(now=now)


class ProxySessionStore:
    """In-memory ProxySession store. Single-event-loop access.

    Mirrors :class:`augmentum.cast.invite_store.InviteStore`. Capped
    record count + opportunistic eviction keep RAM bounded; expired
    sessions are reaped at every lookup.
    """

    def __init__(
        self,
        *,
        default_ttl_s: float = _DEFAULT_TTL_S,
        max_active: int = _MAX_ACTIVE_RECORDS,
    ) -> None:
        self._records: dict[str, ProxySession] = {}
        self._default_ttl = max(60.0, float(default_ttl_s or _DEFAULT_TTL_S))
        self._max_active = max(8, int(max_active or _MAX_ACTIVE_RECORDS))

    def mint(
        self,
        *,
        user_id: str,
        receiver_id: str,
        title_id: str,
        source_url: str,
        ttl_s: float | None = None,
    ) -> ProxySession:
        """Create + register a new ProxySession. Returns it.

        Raises ``ValueError`` if ``source_url`` doesn't carry a valid
        http(s) scheme + host."""
        origin = _normalise_origin(source_url)
        if not origin:
            raise ValueError(f"invalid source_url: {source_url!r}")
        if not user_id:
            raise ValueError("user_id is required")

        # Opportunistic prune.
        if len(self._records) >= self._max_active:
            self._reap_expired()
            if len(self._records) >= self._max_active:
                # Drop the oldest record by expiry to free a slot.
                oldest = min(self._records.values(), key=lambda r: r.expires_at)
                self._records.pop(oldest.token, None)

        session = ProxySession(
            token=_generate_token(),
            user_id=user_id,
            receiver_id=receiver_id or "",
            title_id=title_id or "",
            source_origin=origin,
            source_base_url=source_url,
            expires_at=time.time() + float(ttl_s or self._default_ttl),
        )
        self._records[session.token] = session
        log.info(
            "cast_proxy_session_minted",
            token_prefix=session.token[:8],
            user_id=user_id,
            title_id=title_id,
            origin=origin,
        )
        return session

    def get(self, token: str) -> ProxySession | None:
        """Lookup a session. Returns None when missing OR expired/revoked
        (callers MUST NOT branch on the difference — both mean "no")."""
        if not token:
            return None
        sess = self._records.get(token)
        if sess is None:
            return None
        if not sess.is_active():
            # Eagerly evict — keeps the dict from growing forever and
            # means subsequent lookups by the same token short-circuit.
            self._records.pop(token, None)
            return None
        return sess

    def revoke(self, token: str) -> bool:
        """Revoke + remove. Returns True iff a row existed."""
        sess = self._records.pop(token, None)
        if sess is None:
            return False
        sess.revoked = True
        log.info(
            "cast_proxy_session_revoked",
            token_prefix=token[:8],
            user_id=sess.user_id,
        )
        return True

    def revoke_for_user(self, user_id: str) -> int:
        """Drop every session for a user (logout path). Returns count."""
        if not user_id:
            return 0
        tokens = [t for t, s in self._records.items() if s.user_id == user_id]
        for token in tokens:
            self._records.pop(token, None)
        if tokens:
            log.info(
                "cast_proxy_sessions_revoked_for_user",
                user_id=user_id,
                count=len(tokens),
            )
        return len(tokens)

    def list_for_user(self, user_id: str) -> list[ProxySession]:
        if not user_id:
            return []
        return [s for s in self._records.values() if s.user_id == user_id]

    def _reap_expired(self) -> None:
        now = time.time()
        dead = [t for t, s in self._records.items() if not s.is_active(now=now)]
        for token in dead:
            self._records.pop(token, None)
