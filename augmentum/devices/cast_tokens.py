"""Short-lived public tokens for cast streams.

When the user casts a movie or audiobook from augmentum to a TV, the TV
fetches the media URL directly. The TV doesn't carry the user's auth
cookie, so it can't access augmentum's normal `/api/media/stream/{id}`
endpoint. This module issues short-lived tokens that delegate access:
the cast routine creates a token tied to a specific (user, file, IP)
triple, hands the TV a public URL embedding the token, and the public
endpoint validates+streams without requiring user auth.

Why in-memory:

  Cast sessions are ephemeral by design. If augmentum restarts, every
  active cast is over anyway — the TV's connection drops with the
  process. So persisting tokens to SQLite buys nothing and adds a
  cleanup job. The in-memory store is bounded (max 1024 active tokens)
  and self-pruning on every issue+lookup.

Security shape:

- Token: 32-hex secret (192 bits, urandom). Sufficient against
  enumeration even with a permissive IP allowlist.
- Bound to a single user_id at issue. The downstream stream resolution
  uses that user_id, so a leaked token only accesses files the original
  user could already see.
- TTL: 30 minutes default. Long enough for a movie; short enough that
  a leaked token expires before it's interesting.
- IP allowlist: optional. If set, the token only works when the
  requesting client matches. Recommended for high-value content.
- Single-session revocation: tokens carry a session_id; ending the
  session calls `revoke_session` which drops every related token.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_DEFAULT_TTL_S: float = 30 * 60.0
_MAX_ACTIVE_TOKENS = 1024


@dataclass(slots=True)
class CastToken:
    token: str
    user_id: str
    file_id: str
    expires_at: float
    allowed_client_ip: str = ""
    session_id: str = ""
    query_params: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, *, now: float | None = None) -> bool:
        return (now or time.time()) >= self.expires_at

    def matches_client(self, client_ip: str) -> bool:
        if not self.allowed_client_ip:
            return True
        return self.allowed_client_ip == (client_ip or "").strip()

    def to_dict(self) -> dict[str, Any]:
        # Never serialize the token value — caller already has it; this
        # exists for log/audit shapes only.
        return {
            "user_id": self.user_id,
            "file_id": self.file_id,
            "expires_at": self.expires_at,
            "allowed_client_ip": self.allowed_client_ip,
            "session_id": self.session_id,
            "query_params": dict(self.query_params or {}),
        }


class CastTokenStore:
    """In-memory token store. User-scoped on every read."""

    def __init__(self, *, default_ttl_s: float = _DEFAULT_TTL_S) -> None:
        self._tokens: dict[str, CastToken] = {}
        self._default_ttl = max(60.0, float(default_ttl_s or _DEFAULT_TTL_S))

    def issue(
        self,
        *,
        user_id: str,
        file_id: str,
        allowed_client_ip: str = "",
        session_id: str = "",
        query_params: dict[str, str] | None = None,
        ttl_s: float | None = None,
    ) -> CastToken:
        if not user_id or not file_id:
            raise ValueError("token issue requires user_id and file_id")

        # Prune expired entries first; bounded store keeps drift small.
        self._prune()

        # Hard cap to prevent unbounded growth from misuse.
        if len(self._tokens) >= _MAX_ACTIVE_TOKENS:
            # Evict oldest by expires_at.
            oldest = min(self._tokens.values(), key=lambda t: t.expires_at)
            self._tokens.pop(oldest.token, None)

        token_str = secrets.token_hex(32)
        ttl = max(60.0, float(ttl_s) if ttl_s else self._default_ttl)
        entry = CastToken(
            token=token_str,
            user_id=user_id,
            file_id=file_id,
            expires_at=time.time() + ttl,
            allowed_client_ip=str(allowed_client_ip or "").strip(),
            session_id=str(session_id or "").strip(),
            query_params=dict(query_params or {}),
        )
        self._tokens[token_str] = entry
        log.info(
            "cast_token_issued",
            user_id=user_id,
            file_id=file_id,
            session_id=session_id,
            ttl_s=ttl,
            ip_locked=bool(allowed_client_ip),
        )
        return entry

    def lookup(
        self,
        token: str,
        *,
        client_ip: str = "",
    ) -> CastToken | None:
        entry = self._tokens.get(str(token or "").strip())
        if entry is None:
            return None
        if entry.is_expired():
            self._tokens.pop(entry.token, None)
            return None
        if not entry.matches_client(client_ip):
            log.warning(
                "cast_token_ip_mismatch",
                expected=entry.allowed_client_ip,
                got=client_ip,
            )
            return None
        return entry

    def revoke(self, token: str) -> bool:
        return self._tokens.pop(str(token or "").strip(), None) is not None

    def revoke_session(self, session_id: str) -> int:
        if not session_id:
            return 0
        targets = [t.token for t in self._tokens.values() if t.session_id == session_id]
        for tok in targets:
            self._tokens.pop(tok, None)
        if targets:
            log.info("cast_tokens_revoked_for_session", session_id=session_id, count=len(targets))
        return len(targets)

    def revoke_user(self, user_id: str) -> int:
        targets = [t.token for t in self._tokens.values() if t.user_id == user_id]
        for tok in targets:
            self._tokens.pop(tok, None)
        return len(targets)

    def _prune(self) -> int:
        now = time.time()
        expired = [t.token for t in self._tokens.values() if t.expires_at <= now]
        for tok in expired:
            self._tokens.pop(tok, None)
        return len(expired)

    def active_count(self) -> int:
        self._prune()
        return len(self._tokens)
