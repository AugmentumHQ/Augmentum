"""Universal cast pipeline — per-game strategy + adapter classification.

Public API for the cast-launch flow (``library2/cast-launch.js`` →
``/api/cast/games/{title_id}/profile``):

  - :class:`CastProfile`            (models.py) — declarative per-game profile
  - :class:`CastProfileRegistry`    (registry.py) — SQLite-backed store
  - :class:`CastClassifier`         (classifier.py) — strategy ranker
  - :class:`CastStrategy`           (strategies/base.py) — ABC for strategy plugins
  - built-in strategies under :mod:`augmentum.cast.games.strategies`

Route handlers live at :mod:`augmentum.proxy.cast_games_routes` and
:mod:`augmentum.proxy.cast_game_proxy_routes` to keep the route-map
scanner's ``proxy/*_routes.py`` convention.

See spec: ``docs/superpowers/specs/2026-06-04-universal-cast-pipeline-design.md``
"""

from __future__ import annotations

from augmentum.cast.games.classifier import CastClassifier
from augmentum.cast.games.models import (
    STRATEGY_CONTAINERIZED,
    STRATEGY_PROXY,
    STRATEGY_SHIM,
    CastProfile,
    CastStrategyKind,
    HostCapabilities,
    KeymapProfile,
    PreparedCast,
)
from augmentum.cast.games.registry import CastProfileRegistry
from augmentum.cast.games.strategies.base import CastStrategy, StrategyRegistry

__all__ = [
    "CastProfile",
    "CastStrategyKind",
    "HostCapabilities",
    "KeymapProfile",
    "PreparedCast",
    "STRATEGY_CONTAINERIZED",
    "STRATEGY_PROXY",
    "STRATEGY_SHIM",
    "CastProfileRegistry",
    "CastClassifier",
    "CastStrategy",
    "StrategyRegistry",
]
