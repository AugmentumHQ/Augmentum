"""Integration hook registry — the plug-and-play seam for service manifests.

Each entry in ``KNOWN_INTEGRATION_HOOKS`` maps a hook name (the key
manifest authors use in the ``integration`` block) to a tuple of
``(install_fn, uninstall_fn, hook_meta)``. Adding a new hook
means writing a handler module and registering it here — no edits
to the install dispatcher or manifest validator.

Signatures::

    async def install_hook(request, manifest, service_definition, user_id) -> None
    async def uninstall_hook(request, manifest, service_definition, user_id) -> None

Both receive the full FastAPI ``Request`` (for ``app.state`` access),
the parsed :class:`ServiceManifest`, the running
:class:`ServiceDefinition`, and the ``user_id``. Hooks are isolated:
one failure logs and continues — a failed hook never kills the install.

Forward compatibility: manifests may name hooks this server doesn't
know yet. Unknown hooks warn and no-op so a newer catalog entry
installs fine on an older server, minus the hook it doesn't have.

Each hook also carries a :class:`HookMeta` with UI-facing metadata
(label, icon, companion hint, status provider key). The Discover
home view reads this metadata to render capability cards — no
hardcoded lookup tables. Community hooks that register without
a HookMeta get a derived label + ⚙️ icon automatically.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

# ── Hook callable types ──────────────────────────────────────────────────

InstallHookFn = Callable[
    [Any, Any, Any, str],  # (request, manifest, service_definition, user_id)
    Awaitable[None],
]
UninstallHookFn = Callable[
    [Any, Any, Any, str],
    Awaitable[None],
]

# ── Hook UI metadata ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class HookMeta:
    """UI-facing metadata a hook module carries for the Discover home view.

    Each built-in hook provides one instance. Community hooks that
    register a plain ``(install, uninstall)`` 2-tuple get a derived
    fallback at read time — no hardcoded lookup tables.
    """

    label: str = ""            # "Calendar", "Music & Media"
    icon: str = "⚙️"           # emoji icon for capability cards
    companion_hint: str = ""   # "Companion knows your schedule"
    status_provider: str = ""  # enrichment key: "calendar", "subsonic", "webhook"
    # Whether the Discover card shows a runtime connect/disconnect TOGGLE.
    # True for hooks with a real per-user connection to flip (media_connect,
    # provider_bridge, …). False for install-time wiring that has no runtime
    # on/off (augmentum_backend injects env at provision) — those render as an
    # informational capability row, not a dead switch.
    toggleable: bool = True


# Composite registry entry.
HookEntry = tuple[InstallHookFn, UninstallHookFn, HookMeta]

# Legacy alias kept for consumers that only unpack the first two elements.
HookPair = HookEntry

# ── Registry — populated below ───────────────────────────────────────────

KNOWN_INTEGRATION_HOOKS: dict[str, HookEntry] = {}

# ── Import hooks so they self-register ────────────────────────────────────

from augmentum.marketplace.hooks import (
    augmentum_backend,  # noqa: E402, F401
    calendar,  # noqa: E402, F401
    media_connect,  # noqa: E402, F401
    notifications,  # noqa: E402, F401
    provider_bridge,  # noqa: E402, F401
)

__all__ = [
    "KNOWN_INTEGRATION_HOOKS",
    "HookEntry",
    "HookMeta",
    "HookPair",
    "InstallHookFn",
    "UninstallHookFn",
]
