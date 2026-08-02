# ruff: noqa: E402
# Verb submodules MUST be imported AFTER VerbRegistry is defined below
# (they register with it at import time). The whole module is exempt
# from E402 rather than per-import; the comment above each import
# block explains the ordering invariant.
"""Process-wide management-verb registry.

Verb modules in this package register a :class:`ManagementVerb` at
import time via ``VerbRegistry.register(verb)``. The runtime imports
this package once during ``CompanionRuntime.start()`` (after the
:class:`VerbDispatcher` has opened its bus subscription) and feeds the
registered verbs to the dispatcher.

Mirrors the SubagentRegistry / PrimitiveRegistry pattern — importing
the package is the registration trigger; the registry itself is dumb.

Idempotent: re-importing a module re-registers the same name with the
same verb instance, which is a no-op. Phase 3a/3b verbs live in
sibling modules (``tick_drive.py``, ``tick_energy.py``, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.event_bus import ManagementVerb

log = get_logger(__name__)


class _VerbRegistry:
    """Internal singleton. Use :data:`VerbRegistry`."""

    def __init__(self) -> None:
        self._verbs: dict[str, ManagementVerb] = {}

    def register(self, verb: ManagementVerb) -> ManagementVerb:
        """Register a verb. Returns the verb so the call chains. Re-registering
        the same name with the same instance is a no-op; with a different
        instance the new one wins and a warning is logged."""
        name = verb.name
        if not name:
            raise ValueError("verb has empty `name` attribute")
        existing = self._verbs.get(name)
        if existing is verb:
            return verb
        if existing is not None:
            log.warning(
                "verb_registry_overwrite",
                name=name,
                old_handler=getattr(existing.handler, "__qualname__", "<lambda>"),
                new_handler=getattr(verb.handler, "__qualname__", "<lambda>"),
            )
        self._verbs[name] = verb
        log.debug("verb_registry_added", name=name)
        return verb

    def all(self) -> tuple[ManagementVerb, ...]:
        """All registered verbs in registration order."""
        return tuple(self._verbs.values())

    def names(self) -> tuple[str, ...]:
        return tuple(self._verbs.keys())

    def clear(self) -> None:
        """Test hook only. Production code must never call this."""
        self._verbs.clear()


VerbRegistry = _VerbRegistry()


# Phase 3a/3b verbs — importing this package triggers registration via
# the @verb decorator at module-import time. Add new verb modules below
# as they ship; the runtime imports the package once at start.
from augmentum.companion_runtime.verbs import apply_signal as _apply_signal  # noqa: F401
from augmentum.companion_runtime.verbs import embody_event as _embody_event  # noqa: F401
from augmentum.companion_runtime.verbs import emit_pad_if_delta as _emit_pad_if_delta  # noqa: F401
from augmentum.companion_runtime.verbs import (
    enqueue_proposed_action as _enqueue_proposed_action,  # noqa: F401
)
from augmentum.companion_runtime.verbs import (
    narrate_state_to_user as _narrate_state_to_user,  # noqa: F401
)
from augmentum.companion_runtime.verbs import propose_action as _propose_action  # noqa: F401
from augmentum.companion_runtime.verbs import (
    settle_today_reflection as _settle_today_reflection,  # noqa: F401
)
from augmentum.companion_runtime.verbs import spend_energy as _spend_energy  # noqa: F401
from augmentum.companion_runtime.verbs import (
    tick_affect_baseline as _tick_affect_baseline,  # noqa: F401
)
from augmentum.companion_runtime.verbs import tick_drive as _tick_drive  # noqa: F401
from augmentum.companion_runtime.verbs import tick_energy as _tick_energy  # noqa: F401
from augmentum.companion_runtime.verbs import (
    tick_journal_compactor as _tick_journal_compactor,  # noqa: F401
)
from augmentum.companion_runtime.verbs import (
    tick_observation_consolidator as _tick_observation_consolidator,  # noqa: F401
)
from augmentum.companion_runtime.verbs import tick_scheduler as _tick_scheduler  # noqa: F401

__all__ = ["VerbRegistry"]
