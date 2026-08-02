"""Strategy 2 — fetch through our origin so the universal adapter
reaches the inner realm.

The strategy delegates the actual proxy work to
:mod:`augmentum.cast.games.proxy` — its job is to:

  1. Mint a per-cast ProxySession (or no-op if already minted).
  2. Build a PreparedCast whose ``surface_url`` is the proxy entry
     point, ``input_chain`` is whatever the profile asks for.

cost_rank=2 — picked over the shim (cost_rank=1) only when the title
needs cross-origin coverage. ``can_handle`` returns True iff the title
has an embed_url that's safe to fetch (no internal IPs / wrong scheme).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from augmentum.cast.games.models import (
    STRATEGY_PROXY,
    CastProfile,
    HostCapabilities,
    PreparedCast,
)
from augmentum.cast.games.proxy.fetcher import is_url_safe
from augmentum.cast.games.proxy.session_store import ProxySessionStore
from augmentum.cast.games.strategies.base import CastStrategy
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class OriginProxyStrategy(CastStrategy):
    """Cross-origin → same-origin via /api/cast/game-proxy/."""

    id = STRATEGY_PROXY
    cost_rank = 2

    def __init__(self, session_store: ProxySessionStore | None = None) -> None:
        # session_store is optional at construct time — the classifier
        # may instantiate strategies before app.state finishes wiring.
        # prepare() resolves the live store via the request context at
        # call time (caller passes it through profile.quirks).
        self._store = session_store

    @property
    def session_store(self) -> ProxySessionStore | None:
        return self._store

    def attach_session_store(self, store: ProxySessionStore) -> None:
        """Late wiring — server.py calls this once both pieces exist."""
        self._store = store

    async def can_handle(
        self,
        title: dict[str, Any],
        host: HostCapabilities,
    ) -> bool:
        # No store attached means the proxy isn't wired (test app /
        # startup not yet complete). Defer to the cheaper shim — the
        # classifier picks it because we said no.
        if self._store is None:
            return False
        if not host.has_network_egress:
            return False
        embed = _embed_url(title)
        if not embed:
            return False
        return is_url_safe(embed)

    async def prepare(
        self,
        title: dict[str, Any],
        profile: CastProfile,
    ) -> PreparedCast:
        if self._store is None:
            raise RuntimeError(
                "OriginProxyStrategy has no ProxySessionStore attached",
            )
        embed = profile.embed_url or _embed_url(title)
        if not embed:
            raise ValueError("OriginProxyStrategy requires an embed_url")
        if not is_url_safe(embed):
            raise ValueError(f"OriginProxyStrategy: unsafe embed_url: {embed!r}")

        title_id = str(title.get("id") or title.get("title_id") or profile.title_id)
        user_id = str(title.get("user_id") or "")
        receiver_id = str(title.get("receiver_id") or "")

        session = self._store.mint(
            user_id=user_id,
            receiver_id=receiver_id,
            title_id=title_id,
            source_url=embed,
        )

        parsed = urlparse(embed)
        entry_path = parsed.path or "/"
        if parsed.query:
            entry_path = f"{entry_path}?{parsed.query}"
        surface_url = f"/api/cast/game-proxy/{session.token}{entry_path}"

        return PreparedCast(
            title_id=title_id,
            strategy=self.id,
            surface_url=surface_url,
            surface_kind="html.generic",
            input_chain=profile.input_chain or ("gamepad_api",),
            keymap=profile.keymap,
            session_token=session.token,
            notes=profile.notes,
        )


def _embed_url(title: dict[str, Any]) -> str:
    meta = title.get("metadata") if isinstance(title.get("metadata"), dict) else {}
    return str(meta.get("embed_url") or title.get("embed_url") or "")
