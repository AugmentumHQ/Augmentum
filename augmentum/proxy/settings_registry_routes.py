"""Declarative-action-substrate HTTP surface (Phase 1B).

Read-only endpoints exposing the ``SettingsRegistry`` to the UI and
external agents. Write operations continue through the existing
``PUT /api/config/tools`` / ``PUT /api/config/strings`` endpoints —
the registry's overlay in ``config_routes.py`` makes those endpoints
already see the registered settings, so no new PUT is needed in
Phase 1B.

Spec: docs/superpowers/specs/2026-06-04-declarative-action-substrate-design.md
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from augmentum.config import settings as _settings
from augmentum.registry.builtin import load_into_default_registry
from augmentum.registry.registry import RegistryError, get_registry
from augmentum.registry.settings import Setting
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/settings/registry", tags=["settings-registry"])

# Ensure built-in registrations are loaded at module import. This is
# idempotent — see ``load_into_default_registry``.
load_into_default_registry()


def _current_value(key: str) -> Any:
    """Return the live value of a setting, falling back to its
    registered default if the Settings dataclass doesn't carry it
    (which is the case for keys seeded via
    ``config_routes._TOOL_SETTING_DEFAULTS``).
    """
    val = getattr(_settings, key, None)
    if val is not None:
        return val
    try:
        return get_registry().get(key).default
    except RegistryError:
        return None


def _enrich(setting: Setting) -> dict[str, Any]:
    """Append the live current value to a wire-format entry."""
    return {
        "key": setting.key,
        "kind": setting.kind,
        "default": setting.default,
        "current": _current_value(setting.key),
        "label": setting.label,
        "description": setting.description,
        "section": setting.section,
        "tags": list(setting.tags),
        "min_value": setting.min_value,
        "max_value": setting.max_value,
        "max_length": setting.max_length,
        "enum_values": (
            list(setting.enum_values) if setting.enum_values else None
        ),
        "scope": setting.scope,
        "restart_required": setting.restart_required,
        "deprecated": setting.deprecated,
        "since_version": setting.since_version,
        "advanced": setting.advanced,
        "trust_tier": setting.trust_tier,
        "voice_aliases": list(setting.voice_aliases),
        "companion_surfaceable": setting.companion_surfaceable,
        "modified": _current_value(setting.key) != setting.default,
    }


@router.get("/")
async def list_settings(
    request: Request,
    section: str | None = None,
    tag: str | None = None,
    q: str | None = None,
    show_advanced: bool = False,
    include_deprecated: bool = False,
) -> dict[str, Any]:
    """List registered settings. Filters compose: if multiple of
    ``section`` / ``tag`` / ``q`` are passed, results must satisfy
    all of them.

    Query parameters:
      - section: dotted section prefix (returns ``section`` and
        ``section.*``)
      - tag: filter to settings carrying this tag in their ``tags``
      - q: case-insensitive substring match against key / label /
        description / section / tags / voice_aliases
      - show_advanced: include settings marked ``advanced=True``
        (default False)
      - include_deprecated: include deprecated settings (default False)
    """
    registry = get_registry()
    results = registry.list_all()

    if not include_deprecated:
        results = [s for s in results if not s.is_deprecated()]
    if not show_advanced:
        results = [s for s in results if not s.is_advanced()]
    if section:
        sec_set = {s.key for s in registry.list_by_section(section)}
        results = [s for s in results if s.key in sec_set]
    if tag:
        results = [s for s in results if s.has_tag(tag)]
    if q:
        results = [s for s in results if s.matches_search(q)]

    return {
        "settings": [_enrich(s) for s in results],
        "total": len(results),
    }


@router.get("/sections")
async def list_sections(request: Request) -> dict[str, Any]:
    """Return every distinct section in the registry, with a count
    of settings per section. Used by the UI to build the nav tree."""
    sections: dict[str, int] = {}
    for s in get_registry().list_all():
        sections[s.section] = sections.get(s.section, 0) + 1
    return {
        "sections": [
            {"section": k, "count": v}
            for k, v in sorted(sections.items())
        ],
    }


@router.get("/voice-aliases")
async def list_voice_aliases(request: Request) -> dict[str, Any]:
    """Return every registered voice alias and the setting it
    maps to. Phase 4 consumer (Becca's ``setting.set`` Tool) reads
    this to resolve natural-language phrases to settings keys."""
    out: list[dict[str, str]] = []
    for s in get_registry().list_all():
        for alias in s.voice_aliases:
            out.append({"alias": alias, "key": s.key})
    return {"aliases": out}


@router.get("/{key}")
async def get_setting(request: Request, key: str) -> dict[str, Any]:
    """Return one setting's metadata + current value."""
    try:
        s = get_registry().get(key)
    except RegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _enrich(s)
