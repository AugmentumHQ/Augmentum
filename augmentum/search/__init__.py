"""Search-side infrastructure shared across tools that hit SearXNG.

Currently houses the outbound-proxy manager used by the SearXNG container
to route around residential-IP blocking by upstream engines. Future
search-pipeline pieces (engine-toggle management, cache, etc.) can live
here too.
"""

from __future__ import annotations

from augmentum.search.healthcheck_loop import ProxyHealthcheckLoop
from augmentum.search.proxy_manager import (
    ProxyHealth,
    ProxyManagerStatus,
    SearxngProxyManager,
)

__all__ = [
    "ProxyHealth",
    "ProxyHealthcheckLoop",
    "ProxyManagerStatus",
    "SearxngProxyManager",
]
