"""``SettingsRegistry`` — the runtime singleton holding every registered
``Setting``. Process-wide, populated at module-import time as each
subsystem's settings module is imported.

In Phase 1A the registry holds zero entries by default. Phase 1B begins
populating it with hand-picked settings; Phase 1C completes the bulk
migration. Until a setting is registered here, it continues to live
in the historical 4 declaration sites unchanged.
"""

from __future__ import annotations

from typing import Any, Callable

from augmentum.registry.settings import Setting


class RegistryError(Exception):
    """Raised when a registry operation violates an invariant —
    duplicate key, unknown key on get, etc. These are programmer
    errors at module-import time (Phase 1B onward), not runtime
    failures. Audit.py turns them into build breaks."""


class SettingsRegistry:
    """Single source of truth for declarative settings.

    Sister to ``augmentum/tools/registry.py``'s ``ToolRegistry``:
    Tools are verbs (callable actions), Settings are nouns (mutable
    state). Both are consumed by chat/voice/coder/companion surfaces
    via surface-specific projections.

    Not thread-safe for ``register`` — registrations happen at import
    time in a single-threaded startup phase. Reads (``get``,
    ``list_*``, ``search``) are thread-safe.
    """

    def __init__(self) -> None:
        self._settings: dict[str, Setting] = {}

    # ----- registration -----

    def register(self, setting: Setting) -> None:
        """Register a Setting. Raises if ``setting.key`` already
        registered (no silent overwrite — a duplicate declaration is
        always a bug)."""
        existing = self._settings.get(setting.key)
        if existing is not None:
            raise RegistryError(
                f"Setting {setting.key!r} already registered. "
                f"Existing label={existing.label!r}, new label={setting.label!r}"
            )
        self._settings[setting.key] = setting

    def unregister(self, key: str) -> None:
        """Remove a Setting from the registry. Intended for test
        cleanup only — production code never unregisters."""
        self._settings.pop(key, None)

    def clear(self) -> None:
        """Empty the registry. Intended for test cleanup only."""
        self._settings.clear()

    # ----- lookup -----

    def get(self, key: str) -> Setting:
        try:
            return self._settings[key]
        except KeyError as exc:
            raise RegistryError(f"Setting {key!r} not registered") from exc

    def try_get(self, key: str) -> Setting | None:
        return self._settings.get(key)

    def has(self, key: str) -> bool:
        return key in self._settings

    def list_all(self) -> list[Setting]:
        return list(self._settings.values())

    def keys(self) -> list[str]:
        return list(self._settings.keys())

    # ----- filtered queries -----

    def list_by_section(self, section: str) -> list[Setting]:
        """Return Settings whose section equals ``section`` OR starts
        with ``section + "."`` — supports tree-style nav (asking for
        ``companion`` returns all of ``companion.voice``,
        ``companion.warmth``, …)."""
        prefix = section + "."
        return [
            s
            for s in self._settings.values()
            if s.section == section or s.section.startswith(prefix)
        ]

    def list_by_tag(self, tag: str) -> list[Setting]:
        return [s for s in self._settings.values() if s.has_tag(tag)]

    def list_advanced(self) -> list[Setting]:
        return [s for s in self._settings.values() if s.is_advanced()]

    def list_user_facing(self, *, show_advanced: bool = False) -> list[Setting]:
        """Default UI surface: non-deprecated, non-admin-only, and
        not advanced unless ``show_advanced`` is True. The default
        cut for end users."""
        result = []
        for s in self._settings.values():
            if s.is_deprecated():
                continue
            if s.trust_tier == "admin_only":
                continue
            if s.is_advanced() and not show_advanced:
                continue
            result.append(s)
        return result

    def search(
        self, query: str, *, include_deprecated: bool = False
    ) -> list[Setting]:
        """Match ``query`` against key, label, description, section,
        tags, voice_aliases (case-insensitive substring). Skips
        deprecated unless asked."""
        return [
            s
            for s in self._settings.values()
            if (include_deprecated or not s.is_deprecated())
            and s.matches_search(query)
        ]

    def voice_alias_lookup(self, alias: str) -> Setting | None:
        """Reverse-lookup: given a natural-language alias, return the
        Setting that claims it. Case-insensitive exact match against
        every Setting's ``voice_aliases`` tuple. Returns None if no
        match. Used by Phase 4's ``setting.set`` Tool to resolve
        voice phrases like 'switch to the eva voice'."""
        target = alias.lower().strip()
        if not target:
            return None
        for s in self._settings.values():
            for a in s.voice_aliases:
                if a.lower() == target:
                    return s
        return None

    # ----- exports (Phase 1B consumers) -----

    def to_tool_settings(self) -> dict[str, tuple[type, float, float]]:
        """Emit a dict in the shape of ``config_routes._TOOL_SETTINGS``.
        Each registered numeric/bool setting contributes one entry.
        Settings of kind ``str``/``enum``/``tristate`` are skipped (those
        live in ``_STRING_SETTINGS`` / ``_TRI_STATE_BOOL_SETTINGS``).

        Phase 1A: this method exists for the Phase 1B wiring to consult
        when bridging the registry into the existing validator. Not
        called by anything yet.
        """
        out: dict[str, tuple[type, float, float]] = {}
        for s in self._settings.values():
            if s.kind == "bool":
                out[s.key] = (bool, 0, 1)
            elif s.kind == "int":
                lo = s.min_value if s.min_value is not None else 0
                hi = s.max_value if s.max_value is not None else 1_000_000
                out[s.key] = (int, lo, hi)
            elif s.kind == "float":
                lo_f = float(s.min_value) if s.min_value is not None else 0.0
                hi_f = float(s.max_value) if s.max_value is not None else 1.0
                out[s.key] = (float, lo_f, hi_f)
        return out

    def to_string_settings(self) -> dict[str, int]:
        """Emit a dict in the shape of ``config_routes._STRING_SETTINGS``.
        Each registered ``str``/``enum`` setting contributes one
        ``key -> max_length`` entry."""
        out: dict[str, int] = {}
        for s in self._settings.values():
            if s.kind == "str":
                out[s.key] = s.max_length if s.max_length is not None else 256
            elif s.kind == "enum":
                # Prefer the explicit max_length when declared. Otherwise
                # bound at the longest enum value + 8 chars of buffer
                # (room for one new enum value to land without redeclaring),
                # with a floor of 16 to avoid absurdly tight caps.
                if s.max_length is not None:
                    out[s.key] = s.max_length
                else:
                    longest = max(len(v) for v in (s.enum_values or ())) if s.enum_values else 0
                    out[s.key] = max(longest + 8, 16)
        return out

    def to_restore_map(self) -> dict[str, Callable[[Any], Any]]:
        """Emit a dict in the shape of ``server._SETTINGS_RESTORE_MAP``.
        Each registered Setting contributes one ``key -> parser`` entry
        used by the startup restore loop to coerce persisted strings
        back into typed values."""

        def _parse_bool(raw: Any) -> bool:
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, (int, float)):
                return bool(raw)
            s = str(raw).strip().lower()
            return s in {"1", "true", "yes", "on"}

        def _parse_tristate(raw: Any) -> bool | None:
            if raw is None:
                return None
            if isinstance(raw, str) and raw.lower() in {"", "auto", "none", "null"}:
                return None
            return _parse_bool(raw)

        out: dict[str, Callable[[Any], Any]] = {}
        for s in self._settings.values():
            if s.kind == "bool":
                out[s.key] = _parse_bool
            elif s.kind == "int":
                out[s.key] = int
            elif s.kind == "float":
                out[s.key] = float
            elif s.kind in ("str", "enum"):
                out[s.key] = str
            elif s.kind == "tristate":
                out[s.key] = _parse_tristate
        return out

    def to_wire_format(self) -> list[dict[str, Any]]:
        """Emit the JSON-serializable shape returned by
        ``GET /api/settings/registry`` (Phase 1B endpoint). Excludes
        non-serializable fields (``on_change``, ``tristate_resolver``)."""
        return [
            {
                "key": s.key,
                "kind": s.kind,
                "default": s.default,
                "label": s.label,
                "description": s.description,
                "section": s.section,
                "tags": list(s.tags),
                "min_value": s.min_value,
                "max_value": s.max_value,
                "max_length": s.max_length,
                "enum_values": list(s.enum_values) if s.enum_values else None,
                "scope": s.scope,
                "restart_required": s.restart_required,
                "deprecated": s.deprecated,
                "since_version": s.since_version,
                "advanced": s.advanced,
                "trust_tier": s.trust_tier,
                "voice_aliases": list(s.voice_aliases),
                "companion_surfaceable": s.companion_surfaceable,
            }
            for s in self._settings.values()
        ]


# ----- process singleton -----

_registry: SettingsRegistry | None = None


def get_registry() -> SettingsRegistry:
    """Return the process-wide ``SettingsRegistry`` singleton.

    Lazy-initialized so test code can replace it via ``_reset_for_tests``
    without bookkeeping at import time."""
    global _registry
    if _registry is None:
        _registry = SettingsRegistry()
    return _registry


def _reset_for_tests() -> None:
    """Test-only: clear the singleton. Production code never calls."""
    global _registry
    _registry = None
