"""External-coder driver registry — discovery + selection.

Holds the known drivers and picks one that's actually usable here. Selection is
preference-first then availability: the user's chosen engine if it's available,
else the first available, else None (caller falls back to the native coder).

Codex slots in identically — add ``CodexDriver`` to ``_candidates`` when its
driver lands; nothing else changes.
"""

from __future__ import annotations

from augmentum.coder.external.base import ExternalCoderDriver
from augmentum.coder.external.claude_code import ClaudeCodeDriver
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _candidates(
    *, cwd: str, claude_oauth_token: str, claude_api_key: str,
) -> list[ExternalCoderDriver]:
    """Instantiate all known drivers (not yet availability-checked)."""
    return [
        ClaudeCodeDriver(cwd=cwd, oauth_token=claude_oauth_token, api_key=claude_api_key),
        # CodexDriver(cwd=cwd, api_key=codex_api_key),  # next slice
    ]


async def available_drivers(
    *, cwd: str = "/workspace", claude_oauth_token: str = "", claude_api_key: str = "",
) -> list[ExternalCoderDriver]:
    """All drivers whose engine is actually runnable here (credential + SDK/
    binary present). Order is stable (preference order in _candidates)."""
    out: list[ExternalCoderDriver] = []
    cands = _candidates(
        cwd=cwd, claude_oauth_token=claude_oauth_token, claude_api_key=claude_api_key,
    )
    for d in cands:
        try:
            if await d.is_available():
                out.append(d)
        except Exception:  # noqa: BLE001 — a probe error means "not available"
            log.debug("driver_availability_probe_failed", driver=d.id, exc_info=True)
    return out


async def select_driver(
    prefer: str = "",
    *,
    cwd: str = "/workspace",
    claude_oauth_token: str = "",
    claude_api_key: str = "",
) -> ExternalCoderDriver | None:
    """Pick a usable driver: the preferred one if available, else the first
    available, else None. ``prefer`` is a driver id ("claude_code" | "codex")."""
    avail = await available_drivers(
        cwd=cwd, claude_oauth_token=claude_oauth_token, claude_api_key=claude_api_key,
    )
    if not avail:
        return None
    if prefer:
        for d in avail:
            if d.id == prefer:
                return d
    return avail[0]
