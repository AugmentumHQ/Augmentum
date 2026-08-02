"""Auth primitives for the isolated coder-preview origin.

The coder-preview iframe runs on a separate origin (different external
port, e.g. :6444) so the iframe content cannot reach Augmentum's main
API with the user's session cookies. Because cookies don't cross
origins, we need a non-cookie auth handoff: the main app mints a
one-time token, the iframe redeems it on first request, and the
isolated origin sets its OWN cookie scoped to its own origin for all
subsequent requests.

Two stores live here:

- :class:`PreviewTokenStore` — single-use ``pvt_*`` tokens. Minted by
  the main app on behalf of an authenticated user, consumed by the
  isolated proxy on first request. 60-second default TTL. Mirrors
  ``PairStore`` / ``StreamAuthRedeemStore`` shape.

- :class:`PreviewSessionStore` — sliding-TTL ``pvs_*`` cookie values.
  Minted on token redemption, validated on every subsequent in-iframe
  request (Vite assets, HMR WS, dev-server fetches). Default sliding
  TTL 30 min, hard cap 8 hours.

Both are process-local + in-memory + self-pruning. Cross-restart
persistence buys nothing because the iframe is gone too on restart;
the user just refreshes and a new token mints transparently.

Spec: docs/superpowers/specs/2026-05-27-preview-origin-isolation-design.md
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Sane defaults; the actual values come from settings at instantiation.
_DEFAULT_TOKEN_TTL_S: float = 60.0
_DEFAULT_SESSION_TTL_S: float = 1800.0  # 30 min sliding
_SESSION_HARD_CAP_S: float = 8 * 60 * 60  # 8 hours absolute
_MAX_ACTIVE_TOKENS: int = 256
_MAX_ACTIVE_SESSIONS: int = 1024


# Kind discriminator — what RESOURCE the token / session unlocks. The
# coder preview was the original consumer (kind="workspace"); content
# isolation extends the mechanism to knowledge packs, game bundles,
# emulator artifacts. The kind is validated in the listener gate so
# a workspace-kind token can't open a knowledge_pack path and vice
# versa. workspace_id is kept as the legacy field name on the record
# (loaded by every existing caller); for non-workspace kinds, the
# value is the resource identifier (pack_id, artifact_id, …).
_KIND_WORKSPACE = "workspace"


@dataclass(slots=True)
class _TokenRecord:
    user_id: str
    workspace_id: str
    expires_at: float
    kind: str = _KIND_WORKSPACE

    def is_expired(self, *, now: float | None = None) -> bool:
        return (now or time.time()) >= self.expires_at


@dataclass(slots=True)
class _SessionRecord:
    user_id: str
    workspace_id: str
    # Sliding TTL — extended on every successful get(). Capped by hard_expires_at.
    expires_at: float
    # Absolute expiry — never extends. Forces re-mint after 8h even
    # under heavy activity.
    hard_expires_at: float
    kind: str = _KIND_WORKSPACE

    def is_expired(self, *, now: float | None = None) -> bool:
        now = now or time.time()
        return now >= self.expires_at or now >= self.hard_expires_at


# ---------------------------------------------------------------------------
# Token store — one-time URL tokens for the iframe mount handshake
# ---------------------------------------------------------------------------


class PreviewTokenStore:
    """Single-use one-time tokens for preview-origin auth handoff.

    Mint server-side from an authenticated context (main session
    cookie). Consume on the isolated origin's first request. The
    second consume of the same token returns ``None``, even within
    the TTL — a leaked token can only bootstrap one preview session.
    """

    def __init__(
        self,
        *,
        default_ttl_s: float = _DEFAULT_TOKEN_TTL_S,
        max_active: int = _MAX_ACTIVE_TOKENS,
    ) -> None:
        self._records: dict[str, _TokenRecord] = {}
        self._default_ttl = max(5.0, float(default_ttl_s or _DEFAULT_TOKEN_TTL_S))
        self._max_active = max(8, int(max_active or _MAX_ACTIVE_TOKENS))

    def mint(
        self,
        *,
        user_id: str,
        workspace_id: str,
        ttl_s: float | None = None,
        kind: str = _KIND_WORKSPACE,
    ) -> tuple[str, float]:
        """Mint a fresh ``pvt_*`` token bound to ``(user_id, kind, workspace_id)``.

        Returns ``(token, expires_at_epoch)``. Caller passes ``token``
        to the browser as ``?_pvt=`` query param on the iframe URL.

        ``kind`` defaults to ``"workspace"`` for the original coder
        preview use case. For content isolation (knowledge packs, game
        bundles, emulator artifacts), the caller supplies the resource
        kind and the ``workspace_id`` field carries the resource id.
        Validation in the listener gate ensures a token of one kind
        can't open a path belonging to another kind.
        """
        if not user_id or not workspace_id:
            raise ValueError("preview token requires non-empty user_id + workspace_id")

        self._prune()
        ttl = float(ttl_s) if ttl_s and ttl_s > 0 else self._default_ttl
        token = f"pvt_{secrets.token_hex(16)}"  # 32 hex chars after prefix
        expires_at = time.time() + ttl
        self._records[token] = _TokenRecord(
            user_id=user_id,
            workspace_id=workspace_id,
            expires_at=expires_at,
            kind=kind or _KIND_WORKSPACE,
        )
        log.info(
            "preview_token_minted",
            user_id=user_id,
            workspace_id=workspace_id,
            kind=kind,
            ttl_s=round(ttl, 1),
        )
        return token, expires_at

    def consume(self, token: str) -> _TokenRecord | None:
        """Single-use redemption — returns the bound record + drops it.

        Returns ``None`` for unknown / expired / already-consumed tokens.
        Logs a warning on the unknown/expired path so operators can spot
        replay attempts.
        """
        if not token:
            return None
        record = self._records.pop(token, None)
        if record is None:
            log.warning("preview_token_invalid", reason="unknown_or_consumed")
            return None
        if record.is_expired():
            log.warning(
                "preview_token_invalid",
                reason="expired",
                user_id=record.user_id,
                workspace_id=record.workspace_id,
            )
            return None
        log.info(
            "preview_token_redeemed",
            user_id=record.user_id,
            workspace_id=record.workspace_id,
            token_age_ms=int((time.time() - (record.expires_at - self._default_ttl)) * 1000),
        )
        return record

    def _prune(self) -> None:
        now = time.time()
        expired = [t for t, r in self._records.items() if r.expires_at <= now]
        for t in expired:
            self._records.pop(t, None)
        if len(self._records) >= self._max_active:
            ordered = sorted(self._records.items(), key=lambda kv: kv[1].expires_at)
            overflow = len(self._records) - self._max_active + 1
            for t, _ in ordered[:overflow]:
                self._records.pop(t, None)


# ---------------------------------------------------------------------------
# Session store — cookie-backed sliding sessions on the isolated origin
# ---------------------------------------------------------------------------


class PreviewSessionStore:
    """Sliding-TTL session store for the isolated preview origin.

    On token redemption the proxy mints a session here and sets a
    ``preview_session`` cookie containing the ``pvs_*`` value. Each
    subsequent in-iframe request validates via :meth:`get`, which
    extends the sliding TTL. The hard absolute expiry caps at
    :data:`_SESSION_HARD_CAP_S` regardless of activity so a stolen
    cookie can't be refreshed forever.
    """

    def __init__(
        self,
        *,
        sliding_ttl_s: float = _DEFAULT_SESSION_TTL_S,
        hard_cap_s: float = _SESSION_HARD_CAP_S,
        max_active: int = _MAX_ACTIVE_SESSIONS,
    ) -> None:
        self._records: dict[str, _SessionRecord] = {}
        self._sliding_ttl = max(60.0, float(sliding_ttl_s or _DEFAULT_SESSION_TTL_S))
        self._hard_cap = max(self._sliding_ttl, float(hard_cap_s or _SESSION_HARD_CAP_S))
        self._max_active = max(16, int(max_active or _MAX_ACTIVE_SESSIONS))

    def mint(
        self, *, user_id: str, workspace_id: str,
        kind: str = _KIND_WORKSPACE,
    ) -> str:
        """Mint a fresh ``pvs_*`` cookie value. Returns the value to set.

        The proxy sets this as HttpOnly + Secure + SameSite=Lax on the
        isolated origin. The cookie's domain scope is the isolated
        origin's host:port, so it never reaches Augmentum's main origin.

        ``kind`` carries the same semantics as :meth:`PreviewTokenStore.mint`
        — see that docstring. The session record carries the kind so
        in-iframe requests can be validated against the path's kind.
        """
        if not user_id or not workspace_id:
            raise ValueError("preview session requires non-empty user_id + workspace_id")

        self._prune()
        cookie = f"pvs_{secrets.token_hex(16)}"
        now = time.time()
        self._records[cookie] = _SessionRecord(
            user_id=user_id,
            workspace_id=workspace_id,
            expires_at=now + self._sliding_ttl,
            hard_expires_at=now + self._hard_cap,
            kind=kind or _KIND_WORKSPACE,
        )
        log.info(
            "preview_session_minted",
            user_id=user_id,
            workspace_id=workspace_id,
            kind=kind,
            sliding_ttl_s=int(self._sliding_ttl),
            hard_cap_s=int(self._hard_cap),
        )
        return cookie

    def get(self, cookie: str) -> _SessionRecord | None:
        """Look up a session. Extends sliding TTL on success.

        Returns ``None`` on unknown / expired. Expired records are
        dropped on access so the table doesn't grow unboundedly even
        between prune cycles.
        """
        if not cookie:
            return None
        record = self._records.get(cookie)
        if record is None:
            return None
        if record.is_expired():
            self._records.pop(cookie, None)
            log.info(
                "preview_session_expired",
                user_id=record.user_id,
                workspace_id=record.workspace_id,
            )
            return None
        # Extend sliding TTL, capped by hard expiry.
        new_sliding = time.time() + self._sliding_ttl
        record.expires_at = min(new_sliding, record.hard_expires_at)
        return record

    def revoke(self, cookie: str) -> bool:
        """Drop a session explicitly. Returns True if it existed."""
        return self._records.pop(cookie, None) is not None

    def _prune(self) -> None:
        now = time.time()
        expired = [
            c for c, r in self._records.items()
            if r.expires_at <= now or r.hard_expires_at <= now
        ]
        for c in expired:
            self._records.pop(c, None)
        if len(self._records) >= self._max_active:
            ordered = sorted(self._records.items(), key=lambda kv: kv[1].expires_at)
            overflow = len(self._records) - self._max_active + 1
            for c, _ in ordered[:overflow]:
                self._records.pop(c, None)
