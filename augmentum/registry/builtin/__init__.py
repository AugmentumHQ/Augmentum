"""Built-in setting registrations — the first 10 settings migrated
into the declarative substrate (Phase 1B).

Importing this package triggers registration of every setting declared
in the submodules. Registration is idempotent at the package level
(submodules guard against double-registration via the registry's
own duplicate-check).

Add a new subsystem by:
1. Creating ``augmentum/registry/builtin/<subsystem>.py`` exporting a
   ``register(registry)`` function.
2. Importing it below.
3. The Setting declarations validate at import time, so a malformed
   entry is a build break, not a runtime surprise.
"""

from __future__ import annotations

from augmentum.registry.registry import SettingsRegistry, get_registry


def _load_all(registry: SettingsRegistry) -> None:
    """Register every built-in subsystem's settings into ``registry``.

    Order is irrelevant — each Setting carries its own key and section,
    and the registry rejects duplicates outright.
    """
    from augmentum.registry.builtin import (
        companion,
        engine,
        experience,
        games,
        image,
        modes,
        narrative,
        providers,
        retrieval,
        security,
        voice,
        workspace,
    )

    voice.register(registry)
    narrative.register(registry)
    companion.register(registry)
    engine.register(registry)
    image.register(registry)
    modes.register(registry)
    retrieval.register(registry)
    security.register(registry)
    workspace.register(registry)
    experience.register(registry)
    games.register(registry)
    providers.register(registry)


def load_into_default_registry() -> None:
    """Convenience for the production entry path — loads every built-in
    setting into the process-wide singleton. Idempotent: a duplicate
    call no-ops because the registry raises on duplicate registration
    and we catch + skip if the first setting in the first module is
    already registered.

    Tests typically use ``SettingsRegistry()`` + ``_load_all(r)``
    directly for isolation.
    """
    registry = get_registry()
    # Cheap re-entry guard: if any of the first-bucket settings is
    # already registered, we've already loaded.
    if registry.has("tts_voice_style"):
        return
    _load_all(registry)


__all__ = ["load_into_default_registry", "_load_all"]
