"""Per-user resolution of companion feature flags.

Multi-tenant fix (2026-06): the companion's user-facing flags (the
intensity-dial bundle, persona mode, presence mode) used to be written
to — and read from — the *install-wide* settings store / global
``settings`` singleton. That leaked one tenant's companion menu into
every other tenant's: user B opened the companion panel and saw / could
overwrite user A's toggles. See the ``Multi-Tenant Data Isolation``
section in CLAUDE.md.

This module is the single seam for resolving those flags *per user*,
mirroring the established ``SettingsStore.get_user_or_global`` /
``set_user`` spine already used by grove favorites and the avatar
default. Resolution order is **user override → install-wide store →
``settings`` singleton default**, so existing single-tenant installs are
byte-identical (no per-user row → falls straight back to global).

Honest architectural boundary: there is exactly ONE ``companion_runtime``
instance on ``app.state`` with one ``owner_user_id``. The background
autonomy loops (tick / dreams / journal / curator / …) read these flags
off the global singleton and run only for that owner — they are NOT a
cross-tenant leak surface (a non-owner has no background loop). So
``write_user_flag`` writes the acting user's per-user row AND mirrors to
the install-wide store *when the actor owns the single runtime* (or when
there's no auth / no known owner). That keeps the owner's intensity dial
driving their own background loop exactly as before, while a non-owner's
choice touches only their own row.
"""

from __future__ import annotations

from typing import Any

from augmentum.config import settings

# The intensity-dial bundle (mirrors ``augmentum/companion/intensity.py``
# preset flag keys) — every one is a per-user preference. The runtime
# MASTER switch ``companion_runtime_enabled`` is intentionally absent:
# one runtime instance per install means the master stays install-wide.
_INTENSITY_BUNDLE_KEYS: frozenset[str] = frozenset({
    "companion_dispatch_enabled",
    "companion_dispatch_routes_chat",
    "companion_becca_direct_enabled",
    "companion_salience_enabled",
    "companion_voice_journal_enabled",
    "companion_tick_enabled",
    "companion_journal_enabled",
    "companion_dreams_enabled",
    "companion_drift_audit_enabled",
    "companion_today_enabled",
    "companion_creations_enabled",
    "companion_consolidation_enabled",
    "companion_skills_enabled",
    "companion_initiative_enabled",
    "companion_pad_emit_enabled",
    "companion_cultural_intake_enabled",
})

# Boolean companion flags resolved per-user.
COMPANION_USER_BOOL_KEYS: frozenset[str] = _INTENSITY_BUNDLE_KEYS | frozenset({
    "companion_persona_mode",
})

# String companion flags resolved per-user.
COMPANION_USER_STR_KEYS: frozenset[str] = frozenset({
    "companion_intensity",
    "companion_presence_mode",
})

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})


def _coerce_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in _TRUE_TOKENS


def _encode(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


async def resolve_bool(
    store: Any, user_id: str, key: str, default: bool | None = None,
) -> bool:
    """Resolve a boolean companion flag for ``user_id``.

    Order: user override → install-wide store → ``settings`` default.
    Falls back to the ``settings`` singleton value when ``default`` is
    None so call sites don't have to restate every flag's default.
    """
    if default is None:
        default = bool(getattr(settings, key, False))
    if store is None:
        return default
    raw = (
        await store.get_user_or_global(user_id, key)
        if user_id else await store.get(key)
    )
    return _coerce_bool(raw, default)


async def resolve_str(
    store: Any, user_id: str, key: str, default: str | None = None,
) -> str:
    """Resolve a string companion flag for ``user_id`` (same order as
    :func:`resolve_bool`)."""
    if default is None:
        default = str(getattr(settings, key, "") or "")
    if store is None:
        return default
    raw = (
        await store.get_user_or_global(user_id, key)
        if user_id else await store.get(key)
    )
    return raw if raw is not None else default


async def write_user_flag(
    store: Any,
    *,
    user_id: str,
    owner_user_id: str,
    key: str,
    value: Any,
) -> None:
    """Persist a companion flag for the acting user.

    Always writes the per-user row when ``user_id`` is known (drives that
    user's menu + per-request reads). Mirrors to the install-wide store
    when the actor owns the single runtime — or when there's no auth / no
    known owner — so the one background autonomy loop keeps honoring the
    owner's dial exactly as before.
    """
    if store is None:
        return
    enc = _encode(value)
    if user_id:
        await store.set_user(user_id, key, enc)
    mirror_global = (
        not user_id or not owner_user_id or user_id == owner_user_id
    )
    if mirror_global:
        await store.set(key, enc)


__all__ = [
    "COMPANION_USER_BOOL_KEYS",
    "COMPANION_USER_STR_KEYS",
    "resolve_bool",
    "resolve_str",
    "write_user_flag",
]
