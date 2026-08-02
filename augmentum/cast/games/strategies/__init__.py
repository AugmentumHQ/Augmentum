"""Built-in CastStrategy registrations.

Strategies are appended in cost-rank order so iteration naturally
prefers the cheaper option. Import this module (e.g. via
``augmentum.cast.games.classifier`` boot) for the side effect of
registering the defaults.

To add a new strategy, define it under this package + register it
here. No decorator magic — explicit imports are clearer in a small
plugin namespace.
"""

from __future__ import annotations

from augmentum.cast.games.strategies.base import (
    CastStrategy,
    StrategyRegistry,
    strategy_registry,
)
from augmentum.cast.games.strategies.origin_proxy import OriginProxyStrategy
from augmentum.cast.games.strategies.same_origin import SameOriginStrategy

__all__ = [
    "CastStrategy",
    "StrategyRegistry",
    "strategy_registry",
    "SameOriginStrategy",
    "OriginProxyStrategy",
    "register_default_strategies",
]


def register_default_strategies(registry: StrategyRegistry | None = None) -> None:
    """Idempotent registration of built-in strategies into ``registry``
    (or the module-level singleton when omitted).

    The OriginProxyStrategy is registered WITHOUT a session_store —
    the lifespan wiring in :mod:`augmentum.proxy.server` calls
    ``.attach_session_store()`` once the store exists. Until then the
    strategy's ``can_handle`` returns False so the classifier falls
    through to the shim.
    """
    target = registry or strategy_registry
    if not target.has("shim"):
        target.register(SameOriginStrategy())
    if not target.has("proxy"):
        target.register(OriginProxyStrategy())


# Auto-register defaults at import time, mirroring ProfileRegistry's
# convention in game_stream/profiles.py.
register_default_strategies()
