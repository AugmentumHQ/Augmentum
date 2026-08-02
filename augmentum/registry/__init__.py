"""Declarative action substrate — the single source of truth for every
user-tunable knob and (eventually) every user-derived skill in Augmentum.

Sister substrate to ``augmentum/tools/registry.py`` from the Unified
Primitive Layer ([[project_unified_primitive_layer]]). Tools are verbs;
Settings are nouns. Both consumed by chat/voice/coder/companion surfaces
through their respective ``get_for_surface()`` lookups.

Phase 1A (this module on first import) is purely additive: the registry
exists, holds zero entries, and is callable. Existing settings continue
to live in ``config.py`` / ``config_routes.py`` / ``server.py`` /
``settings.js`` — they only migrate into the registry as Phase 1B/1C
proceed.

See ``docs/superpowers/specs/2026-06-04-declarative-action-substrate-design.md``.
"""

from __future__ import annotations

from augmentum.registry.registry import SettingsRegistry, get_registry
from augmentum.registry.settings import (
    Setting,
    SettingKind,
    SettingScope,
    TrustTier,
)

__all__ = [
    "Setting",
    "SettingKind",
    "SettingScope",
    "SettingsRegistry",
    "TrustTier",
    "get_registry",
]
