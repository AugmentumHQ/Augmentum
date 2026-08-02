"""Strategy 2 — origin proxy + universal adapter injection.

Renders any cross-origin game inside our origin so the universal input
adapter (gamepad / keyboard / touch / pointer) reaches the inner realm.

Three pieces:

  - :mod:`augmentum.cast.games.proxy.session_store` — minted per-cast
    token bound to (user_id, receiver_id, title_id, source_origin).
  - :mod:`augmentum.cast.games.proxy.fetcher` — async HTTP fetcher with
    a same-origin redirect policy + on-disk asset cache + header strip.
  - :mod:`augmentum.cast.games.proxy.rewriter` — HTML/CSS URL rewriter
    + adapter-loader + service-worker shim injection.
  - :mod:`augmentum.cast.games.proxy.routes` — the
    ``/api/cast/game-proxy/{token}/{path}`` surface + start endpoint.

See spec: ``docs/superpowers/specs/2026-06-04-universal-cast-pipeline-design.md``
"""

from __future__ import annotations

from augmentum.cast.games.proxy.fetcher import (
    BLOCKED_RESPONSE_HEADERS,
    AssetCache,
    FetchResult,
    ProxyFetcher,
)
from augmentum.cast.games.proxy.rewriter import (
    DEFAULT_CDN_ALLOWLIST,
    inject_adapter_loader,
    rewrite_css,
    rewrite_html,
)
from augmentum.cast.games.proxy.session_store import (
    ProxySession,
    ProxySessionStore,
)

__all__ = [
    "AssetCache",
    "BLOCKED_RESPONSE_HEADERS",
    "DEFAULT_CDN_ALLOWLIST",
    "FetchResult",
    "ProxyFetcher",
    "ProxySession",
    "ProxySessionStore",
    "inject_adapter_loader",
    "rewrite_css",
    "rewrite_html",
]
