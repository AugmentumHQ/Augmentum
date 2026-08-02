"""Process-wide subagent registry.

Subagent classes register themselves at import time via
``SubagentRegistry.register(cls)``. The registry is gated by the
``companion_subagent_registry_active`` setting flag: when off,
``available()`` returns an empty tuple, so the dispatcher (Sprint 3)
sees no candidates and falls back to the legacy mode-router.

Registration is idempotent — re-importing a module does not duplicate
entries. The flag gate is intentionally read on every access so the
admin UI can flip it at runtime without a restart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.subagents.base import SubagentBase

log = get_logger(__name__)


class _SubagentRegistry:
    """Internal singleton. Do not instantiate — use the module-level
    :data:`SubagentRegistry`."""

    def __init__(self) -> None:
        self._classes: dict[str, type[SubagentBase]] = {}
        self._instances: dict[str, SubagentBase] = {}

    def register(self, cls: type[SubagentBase]) -> type[SubagentBase]:
        """Register a subagent class. Returns the class so it can be used
        as a decorator. Idempotent — re-registering the same name with
        the same class is a no-op; with a different class it overwrites
        and warns."""
        name = cls.name
        if not name:
            raise ValueError("subagent class has empty `name` attribute")
        existing = self._classes.get(name)
        if existing is cls:
            return cls
        if existing is not None:
            log.warning(
                "subagent_overwrite",
                name=name,
                old=f"{existing.__module__}.{existing.__name__}",
                new=f"{cls.__module__}.{cls.__name__}",
            )
        self._classes[name] = cls
        # Invalidate any cached instance for this name
        self._instances.pop(name, None)
        log.debug("subagent_registered", name=name)
        return cls

    def get(self, name: str) -> SubagentBase | None:
        """Return the singleton instance for ``name`` or None if absent
        OR if the registry is inactive."""
        if not self._is_active():
            return None
        if name not in self._classes:
            return None
        if name not in self._instances:
            self._instances[name] = self._classes[name]()
        return self._instances[name]

    def available(self) -> tuple[SubagentBase, ...]:
        """All registered subagents, instantiated lazily. Empty tuple
        when the registry flag is off."""
        if not self._is_active():
            return ()
        for name in self._classes:
            if name not in self._instances:
                self._instances[name] = self._classes[name]()
        return tuple(self._instances.values())

    def names(self) -> tuple[str, ...]:
        """Names of all registered subagents, regardless of flag state.
        Useful for diagnostics — `available()` would hide them when off."""
        return tuple(self._classes.keys())

    def _is_active(self) -> bool:
        # Imported lazily so that settings can be patched in tests.
        from augmentum.config import settings
        return bool(getattr(settings, "companion_subagent_registry_active", False))


SubagentRegistry = _SubagentRegistry()

__all__ = ["SubagentRegistry"]
