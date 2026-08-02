"""Process-wide primitive registry.

Mirrors :class:`SubagentRegistry` but gated by
``companion_primitive_registry_active`` (falls back to
``companion_subagent_registry_active`` if the dedicated flag is
absent — Unit F may not yet have shipped both).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.primitives.base import PrimitiveBase

log = get_logger(__name__)


class _PrimitiveRegistry:
    def __init__(self) -> None:
        self._classes: dict[str, type[PrimitiveBase]] = {}
        self._instances: dict[str, PrimitiveBase] = {}

    def register(self, cls: type[PrimitiveBase]) -> type[PrimitiveBase]:
        name = cls.name
        if not name:
            raise ValueError("primitive class has empty `name` attribute")
        existing = self._classes.get(name)
        if existing is cls:
            return cls
        if existing is not None:
            log.warning(
                "primitive_overwrite",
                name=name,
                old=f"{existing.__module__}.{existing.__name__}",
                new=f"{cls.__module__}.{cls.__name__}",
            )
        self._classes[name] = cls
        self._instances.pop(name, None)
        log.debug("primitive_registered", name=name)
        return cls

    def get(self, name: str) -> PrimitiveBase | None:
        if not self._is_active():
            return None
        if name not in self._classes:
            return None
        if name not in self._instances:
            self._instances[name] = self._classes[name]()
        return self._instances[name]

    def available(self) -> tuple[PrimitiveBase, ...]:
        if not self._is_active():
            return ()
        for name in self._classes:
            if name not in self._instances:
                self._instances[name] = self._classes[name]()
        return tuple(self._instances.values())

    def names(self) -> tuple[str, ...]:
        return tuple(self._classes.keys())

    def _is_active(self) -> bool:
        from augmentum.config import settings
        # Primary flag; fall back to the shared subagent flag so Unit F
        # can ship either or both. When neither exists the registry is
        # inert by default — matches the Sprint-1 "all flags False"
        # promise.
        if hasattr(settings, "companion_primitive_registry_active"):
            return bool(settings.companion_primitive_registry_active)
        return bool(getattr(settings, "companion_subagent_registry_active", False))


PrimitiveRegistry = _PrimitiveRegistry()

__all__ = ["PrimitiveRegistry"]
