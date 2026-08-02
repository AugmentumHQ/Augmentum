"""``setting.get`` / ``setting.set`` Tools — the Becca substrate for
the declarative action substrate.

These tools turn the SettingsRegistry into a uniformly-callable surface:
- Chat LLM can function-call to read or change settings
- Voice (the "swap to the Eva voice" path) can resolve natural aliases
  via ``SettingsRegistry.voice_alias_lookup`` and apply via setting.set
- Coder can read settings during planning (e.g. "what's the current
  compaction threshold?")

Trust tier enforcement happens at ``execute`` time so the same Tool
shape works for every caller:

  - local_reversible:    apply immediately
  - local_significant:   require ``confirm=True`` unless caller is admin
  - external:            require ``confirm=True`` always
  - admin_only:          require caller to be admin

Spec: docs/superpowers/specs/2026-06-04-declarative-action-substrate-design.md
"""

from __future__ import annotations

from typing import Any

from augmentum.registry.builtin import load_into_default_registry
from augmentum.registry.registry import RegistryError, get_registry
from augmentum.registry.settings import Setting, TrustTier
from augmentum.tools.base import (
    SurfaceExposure,
    Tool,
    ToolCategory,
    ToolResult,
)

# Ensure registrations exist when tools module is imported.
load_into_default_registry()


def _resolve_setting(key_or_alias: str) -> Setting:
    """Resolve a Setting by exact key OR voice alias.

    Voice-alias resolution is case-insensitive and matches the exact
    alias string. Falls back to exact-key lookup if no alias matches.

    Raises ``RegistryError`` if neither resolves.
    """
    if not key_or_alias:
        raise RegistryError("Setting key or alias must be non-empty")
    r = get_registry()
    s = r.voice_alias_lookup(key_or_alias)
    if s is not None:
        return s
    return r.get(key_or_alias)


def _coerce_value(setting: Setting, raw: Any) -> Any:
    """Type-coerce ``raw`` to the Setting's kind. Raises ``ValueError``
    on invalid input (out-of-range, unknown enum value, etc.)."""
    if setting.kind == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return bool(raw)
        s = str(raw).strip().lower()
        if s in {"true", "1", "yes", "on"}:
            return True
        if s in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Cannot coerce {raw!r} to bool for {setting.key}")
    if setting.kind == "int":
        v = int(raw)
        if setting.min_value is not None and v < setting.min_value:
            raise ValueError(
                f"{setting.key}: {v} below min {setting.min_value}"
            )
        if setting.max_value is not None and v > setting.max_value:
            raise ValueError(
                f"{setting.key}: {v} above max {setting.max_value}"
            )
        return v
    if setting.kind == "float":
        v = float(raw)
        if setting.min_value is not None and v < setting.min_value:
            raise ValueError(
                f"{setting.key}: {v} below min {setting.min_value}"
            )
        if setting.max_value is not None and v > setting.max_value:
            raise ValueError(
                f"{setting.key}: {v} above max {setting.max_value}"
            )
        return v
    if setting.kind == "str":
        v = str(raw)
        if setting.max_length is not None and len(v) > setting.max_length:
            raise ValueError(
                f"{setting.key}: length {len(v)} exceeds max "
                f"{setting.max_length}"
            )
        return v
    if setting.kind == "enum":
        v = str(raw)
        if setting.enum_values and v not in setting.enum_values:
            raise ValueError(
                f"{setting.key}: {v!r} not in enum_values "
                f"{setting.enum_values!r}"
            )
        return v
    if setting.kind == "tristate":
        if raw is None:
            return None
        if isinstance(raw, str) and raw.lower() in {"", "auto", "none", "null"}:
            return None
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        s = str(raw).strip().lower()
        if s in {"true", "1", "yes", "on"}:
            return True
        if s in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Cannot coerce {raw!r} to tristate for {setting.key}")
    raise ValueError(f"Unknown kind {setting.kind!r} for {setting.key}")


def _current_value(setting: Setting) -> Any:
    """Return the live value of a setting, falling back to its declared
    default if the Settings dataclass doesn't carry it."""
    from augmentum.config import settings as _settings  # noqa: PLC0415

    val = getattr(_settings, setting.key, None)
    return val if val is not None else setting.default


def _trust_check(
    setting: Setting,
    *,
    confirm: bool,
    is_admin: bool,
) -> tuple[bool, str]:
    """Return (allowed, reason)."""
    tier: TrustTier = setting.trust_tier
    if tier == "local_reversible":
        return True, ""
    if tier == "local_significant":
        if is_admin or confirm:
            return True, ""
        return (
            False,
            (
                f"Setting {setting.key!r} is tier 'local_significant' — "
                f"call again with confirm=True to apply."
            ),
        )
    if tier == "external":
        if confirm:
            return True, ""
        return (
            False,
            (
                f"Setting {setting.key!r} is tier 'external' — call again "
                f"with confirm=True after surfacing a preview to the user."
            ),
        )
    if tier == "admin_only":
        if is_admin:
            return True, ""
        return (
            False,
            f"Setting {setting.key!r} requires admin privileges.",
        )
    return True, ""


def _context_user_id(ctx: dict | None) -> str:
    if not ctx:
        return ""
    return str(ctx.get("user_id", ""))


def _context_is_admin(ctx: dict | None) -> bool:
    if not ctx:
        return False
    return bool(ctx.get("is_admin", False))


def _context_settings_store(ctx: dict | None):
    if not ctx:
        return None
    return ctx.get("settings_store")


# ===================================================================
# setting.get
# ===================================================================


class SettingGetTool(Tool):
    """Read a setting by key or natural-language alias.

    Inputs:
      - ``key``: snake_case setting key OR a registered voice_alias.
        Voice aliases match case-insensitively (e.g. ``"becca"`` →
        ``companion_runtime_enabled``).

    Output: A small JSON object with current value, default, kind,
    label, description, and whether the value is modified from default.
    """

    @property
    def name(self) -> str:
        return "setting.get"

    @property
    def description(self) -> str:
        return (
            "Read a setting by snake_case key or natural-language alias. "
            "Returns the current value, default, kind, and whether the "
            "value has been modified."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FETCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "snake_case setting key or a registered voice alias "
                        "(case-insensitive)"
                    ),
                },
            },
            "required": ["key"],
        }

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(
            chat=True,
            voice="core",
            coder=True,
            companion=True,
            voice_capability_line=(
                "Read any setting — say 'what's the X setting?' or 'show me "
                "the eva voice setting'."
            ),
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        key = kwargs.get("key", "")
        try:
            s = _resolve_setting(key)
        except RegistryError as exc:
            return ToolResult(success=False, error=str(exc), validation_error=True)

        current = _current_value(s)
        return ToolResult(
            success=True,
            output=(
                f"{s.label}: {current!r} "
                f"({'modified' if current != s.default else 'default'})"
            ),
            metadata={
                "key": s.key,
                "current": current,
                "default": s.default,
                "kind": s.kind,
                "label": s.label,
                "description": s.description,
                "section": s.section,
                "modified": current != s.default,
                "trust_tier": s.trust_tier,
                "advanced": s.advanced,
                "deprecated": bool(s.deprecated),
                "restart_required": s.restart_required,
            },
        )


# ===================================================================
# setting.set
# ===================================================================


class SettingSetTool(Tool):
    """Change a setting by key or natural-language alias.

    Trust tier enforcement:
      - local_reversible: applied immediately
      - local_significant: requires ``confirm=True`` unless caller is admin
      - external: requires ``confirm=True`` always
      - admin_only: requires caller to be admin

    Side-channel persistence: when ``_context.settings_store`` is provided,
    the new value is written to the SQLite kv-store so it survives
    restart. Without a store the change is in-process only — useful for
    tests but not for production callers.
    """

    @property
    def name(self) -> str:
        return "setting.set"

    @property
    def description(self) -> str:
        return (
            "Change a setting by snake_case key or natural-language alias. "
            "Some settings require confirm=True or admin privileges — "
            "consult the setting's trust_tier (read via setting.get)."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "snake_case setting key or registered voice alias"
                    ),
                },
                "value": {
                    "description": (
                        "new value, coerced to the setting's kind"
                    ),
                },
                "confirm": {
                    "type": "boolean",
                    "description": (
                        "explicit confirmation for tiered settings. "
                        "Required for local_significant (without admin) and "
                        "external."
                    ),
                    "default": False,
                },
            },
            "required": ["key", "value"],
        }

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(
            chat=True,
            voice="interactive",
            coder=True,
            companion=True,
            voice_capability_line=(
                "Change any setting by name or alias — 'switch to the eva "
                "voice', 'make her quieter', 'turn off web search'."
            ),
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        key = kwargs.get("key", "")
        if "value" not in kwargs:
            return ToolResult(
                success=False,
                error="setting.set requires a 'value' argument",
                validation_error=True,
            )
        raw_value = kwargs.get("value")
        confirm = bool(kwargs.get("confirm", False))
        ctx = kwargs.get("_context")
        is_admin = _context_is_admin(ctx)

        try:
            s = _resolve_setting(key)
        except RegistryError as exc:
            return ToolResult(
                success=False, error=str(exc), validation_error=True
            )

        if s.deprecated:
            return ToolResult(
                success=False,
                error=(
                    f"Setting {s.key!r} is deprecated: {s.deprecated}"
                ),
                validation_error=True,
            )

        allowed, reason = _trust_check(s, confirm=confirm, is_admin=is_admin)
        if not allowed:
            return ToolResult(
                success=False,
                error=reason,
                metadata={
                    "trust_tier": s.trust_tier,
                    "confirm_required": True,
                    "key": s.key,
                    "label": s.label,
                    "description": s.description,
                },
            )

        try:
            new_value = _coerce_value(s, raw_value)
        except ValueError as exc:
            return ToolResult(
                success=False, error=str(exc), validation_error=True
            )

        old_value = _current_value(s)
        if new_value == old_value:
            return ToolResult(
                success=True,
                output=f"{s.label} is already {new_value!r}",
                metadata={
                    "key": s.key,
                    "current": new_value,
                    "default": s.default,
                    "changed": False,
                    "restart_required": s.restart_required,
                },
            )

        # Apply to the in-process Settings singleton. This is the same
        # mechanism PUT /api/config/tools uses (object.__setattr__ on
        # a frozen pydantic dataclass).
        try:
            from augmentum.config import settings as _settings  # noqa: PLC0415

            object.__setattr__(_settings, s.key, new_value)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                error=f"Failed to apply to settings: {exc}",
            )

        # Persist to the kv-store if context provides one. Without a
        # store the change survives only this process.
        store = _context_settings_store(ctx)
        persisted = False
        if store is not None:
            try:
                if s.kind == "bool":
                    await store.set(s.key, "1" if new_value else "0")
                elif s.kind == "tristate":
                    if new_value is None:
                        await store.delete(s.key)
                    else:
                        await store.set(s.key, "1" if new_value else "0")
                else:
                    await store.set(s.key, str(new_value))
                persisted = True
            except Exception as exc:  # noqa: BLE001
                return ToolResult(
                    success=False,
                    error=f"Applied in-process but persistence failed: {exc}",
                    metadata={
                        "key": s.key,
                        "current": new_value,
                        "persisted": False,
                    },
                )

        # Fire the optional on_change callback.
        if s.on_change is not None:
            try:
                s.on_change(old_value, new_value)
            except Exception:  # noqa: BLE001
                # on_change failures don't roll back the write.
                pass

        return ToolResult(
            success=True,
            output=(
                f"{s.label}: {old_value!r} → {new_value!r}"
                + (" (restart required)" if s.restart_required else "")
            ),
            metadata={
                "key": s.key,
                "current": new_value,
                "previous": old_value,
                "default": s.default,
                "persisted": persisted,
                "changed": True,
                "restart_required": s.restart_required,
                "trust_tier": s.trust_tier,
            },
        )
