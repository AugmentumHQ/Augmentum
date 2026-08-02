"""Short-lived public tokens for browser/TV surface receivers."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_TTL_S: float = 6 * 60 * 60.0
_MAX_ACTIVE_TOKENS = 2048


@dataclass(slots=True)
class SurfaceAccessToken:
    token: str
    user_id: str
    session_id: str
    expires_at: float
    file_id: str = ""
    scopes: tuple[str, ...] = ()
    allowed_client_ip: str = ""
    query_params: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, *, now: float | None = None) -> bool:
        return (now or time.time()) >= self.expires_at

    def matches_client(self, client_ip: str) -> bool:
        if not self.allowed_client_ip:
            return True
        return self.allowed_client_ip == (client_ip or "").strip()

    def allows(self, scope: str) -> bool:
        wanted = str(scope or "").strip()
        if not wanted:
            return True
        return "*" in self.scopes or wanted in self.scopes

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "file_id": self.file_id,
            "expires_at": self.expires_at,
            "scopes": list(self.scopes),
            "allowed_client_ip": self.allowed_client_ip,
            "query_params": dict(self.query_params or {}),
        }


class SurfaceAccessTokenStore:
    """In-memory delegation tokens for unauthenticated TV browsers.

    Tokens are tied to one authenticated user and one surface session.
    They deliberately do not authenticate the browser as the user; they
    only delegate the narrow scopes listed on the token.
    """

    def __init__(self, *, default_ttl_s: float = _DEFAULT_TTL_S) -> None:
        self._tokens: dict[str, SurfaceAccessToken] = {}
        self._default_ttl = max(60.0, float(default_ttl_s or _DEFAULT_TTL_S))

    def issue(
        self,
        *,
        user_id: str,
        session_id: str,
        file_id: str = "",
        scopes: list[str] | tuple[str, ...] | None = None,
        allowed_client_ip: str = "",
        query_params: dict[str, str] | None = None,
        ttl_s: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> SurfaceAccessToken:
        if not user_id or not session_id:
            raise ValueError("surface token issue requires user_id and session_id")

        self._prune()
        if len(self._tokens) >= _MAX_ACTIVE_TOKENS:
            oldest = min(self._tokens.values(), key=lambda t: t.expires_at)
            self._tokens.pop(oldest.token, None)

        normalized_scopes = tuple(
            sorted({
                str(s or "").strip()
                for s in (scopes or ())
                if str(s or "").strip()
            })
        )
        token_str = secrets.token_hex(32)
        ttl = max(60.0, float(ttl_s) if ttl_s else self._default_ttl)
        entry = SurfaceAccessToken(
            token=token_str,
            user_id=user_id,
            session_id=session_id,
            file_id=str(file_id or "").strip(),
            scopes=normalized_scopes,
            expires_at=time.time() + ttl,
            allowed_client_ip=str(allowed_client_ip or "").strip(),
            query_params=dict(query_params or {}),
            extra=dict(extra or {}),
        )
        self._tokens[token_str] = entry
        log.info(
            "surface_token_issued",
            user_id=user_id,
            session_id=session_id,
            file_id=entry.file_id,
            scopes=list(entry.scopes),
            ttl_s=ttl,
            ip_locked=bool(entry.allowed_client_ip),
        )
        return entry

    def lookup(
        self,
        token: str,
        *,
        client_ip: str = "",
        required_scope: str = "",
    ) -> SurfaceAccessToken | None:
        entry = self._tokens.get(str(token or "").strip())
        if entry is None:
            return None
        if entry.is_expired():
            self._tokens.pop(entry.token, None)
            return None
        if not entry.matches_client(client_ip):
            log.warning(
                "surface_token_ip_mismatch",
                expected=entry.allowed_client_ip,
                got=client_ip,
            )
            return None
        if required_scope and not entry.allows(required_scope):
            log.warning(
                "surface_token_scope_denied",
                session_id=entry.session_id,
                required_scope=required_scope,
                scopes=list(entry.scopes),
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
            log.info("surface_tokens_revoked_for_session", session_id=session_id, count=len(targets))
        return len(targets)

    def revoke_user(self, user_id: str) -> int:
        targets = [t.token for t in self._tokens.values() if t.user_id == user_id]
        for tok in targets:
            self._tokens.pop(tok, None)
        return len(targets)

    def active_count(self) -> int:
        self._prune()
        return len(self._tokens)

    def _prune(self) -> int:
        now = time.time()
        expired = [t.token for t in self._tokens.values() if t.expires_at <= now]
        for tok in expired:
            self._tokens.pop(tok, None)
        return len(expired)
