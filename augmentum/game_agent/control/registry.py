"""ProfileRegistry — load, hold, and compose game/controller profiles.

The registry is the single source of truth at runtime. The shipped
default registry loads every profile under
``augmentum/game_agent/control/profiles/`` at module import; tests
construct empty :class:`ProfileRegistry` instances directly so they
don't share state.

Why one registry instead of per-call loaders
--------------------------------------------
Profiles are tiny, JSON-Schema-validated, and used on every agent
turn. Loading them on every session POST would do disk I/O + JSON
parsing on the hot path of session creation. Loading once at process
start, then sharing one immutable registry, is both faster and
simpler.
"""

from __future__ import annotations

import logging
from pathlib import Path

from augmentum.game_agent.control.profile import (
    ComposedProfile,
    ControllerProfile,
    GameProfile,
    ProfileLoadError,
    compose,
    load_controller_profile,
    load_game_profile,
)

log = logging.getLogger(__name__)


class ProfileRegistry:
    """In-memory store of validated controller / game profiles.

    Use when:
    - The route layer is constructing an :class:`Orchestrator` and
      needs to compose a controller + game pairing from request body
      string ids.

    Expects:
    - All profile JSON files in the provided directories pass schema
      validation. A bad profile raises :class:`ProfileLoadError` at
      load time, surfacing the issue at process start rather than on
      first session POST.

    Returns:
    - Sorted lists of registered ids via :meth:`controller_ids` /
      :meth:`game_ids`. Composed profiles via :meth:`compose`.
    """

    def __init__(self) -> None:
        self._controllers: dict[str, ControllerProfile] = {}
        self._games: dict[str, GameProfile] = {}

    # ── Loading ───────────────────────────────────────────────────────

    def load_directory(self, root: Path) -> None:
        """Load every JSON profile under ``<root>/controllers`` + ``<root>/games``.

        Files are loaded in lexicographic order so any cross-references
        between profiles resolve predictably. A single malformed file
        raises :class:`ProfileLoadError` and aborts the load -- we'd
        rather fail at startup than serve a half-configured registry.
        """

        controllers_dir = root / "controllers"
        if controllers_dir.is_dir():
            for path in sorted(controllers_dir.glob("*.json")):
                self.register_controller(load_controller_profile(path))
        games_dir = root / "games"
        if games_dir.is_dir():
            for path in sorted(games_dir.glob("*.json")):
                self.register_game(load_game_profile(path))

    def register_controller(self, profile: ControllerProfile) -> None:
        """Register a controller profile, overwriting any prior entry with the same id.

        Operator-supplied profiles (loaded after the bundled defaults)
        deliberately win on id collision so deployments can override
        in-tree definitions without forking the package.
        """

        if profile.id in self._controllers:
            log.info(
                "control.controller_profile_replaced",
                extra={"id": profile.id},
            )
        self._controllers[profile.id] = profile

    def register_game(self, profile: GameProfile) -> None:
        """Register a game profile, overwriting any prior entry with the same id."""

        if profile.id in self._games:
            log.info(
                "control.game_profile_replaced", extra={"id": profile.id},
            )
        self._games[profile.id] = profile

    # ── Read ──────────────────────────────────────────────────────────

    def controller_ids(self) -> list[str]:
        return sorted(self._controllers.keys())

    def game_ids(self) -> list[str]:
        return sorted(self._games.keys())

    def controller(self, id: str) -> ControllerProfile:
        if id not in self._controllers:
            raise ProfileLoadError(
                f"no controller profile registered with id {id!r}; "
                f"known: {self.controller_ids()}",
            )
        return self._controllers[id]

    def game(self, id: str) -> GameProfile:
        if id not in self._games:
            raise ProfileLoadError(
                f"no game profile registered with id {id!r}; "
                f"known: {self.game_ids()}",
            )
        return self._games[id]

    def compose(self, controller_id: str, game_id: str) -> ComposedProfile:
        """Compose a controller + game pair. Raises on invalid pairing."""

        return compose(self.controller(controller_id), self.game(game_id))


def _build_default_registry() -> ProfileRegistry:
    """Build the registry shipped with the package.

    Loads every profile under ``augmentum/game_agent/control/profiles``
    at first import. Operators add or override via ``app.state.profile_registry``
    in custom wiring.
    """

    registry = ProfileRegistry()
    bundled = Path(__file__).parent / "profiles"
    if bundled.is_dir():
        registry.load_directory(bundled)
    return registry


# Module-level shared registry for production use.
default_registry: ProfileRegistry = _build_default_registry()


__all__ = ["ProfileRegistry", "default_registry"]
