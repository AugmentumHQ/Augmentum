"""Phase 4 — setting.get / setting.set Tools.

Tests the Becca substrate end-to-end: voice-alias resolution, trust-tier
gating, type coercion, range validation, persistence, and the modified
flag round-trip.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.registry.registry import _reset_for_tests, get_registry
from augmentum.tools.setting_tools import (
    SettingGetTool,
    SettingSetTool,
    _coerce_value,
    _resolve_setting,
    _trust_check,
)

_RESTORE_KEYS = (
    "tts_voice_style",
    "tts_kokoro_hbe",
    "voice_smart_turn_threshold",
    "narrative_memory_mode",
    "narrative_smart_retrieval_count",
    "companion_runtime_enabled",
    "engine_use_jinja_template",
    "engine_multislot_enabled",
)


@pytest.fixture(autouse=True)
def _isolate_registry_and_settings():
    """Each test starts with a fresh singleton populated with the
    Phase 1B+1C built-in registrations AND a snapshot of the live
    Settings dataclass fields the tests are known to mutate. The
    snapshot is restored on teardown to prevent test bleed."""
    _reset_for_tests()
    from augmentum.config import settings as _settings
    from augmentum.registry.builtin import load_into_default_registry

    load_into_default_registry()
    snapshot = {k: getattr(_settings, k, None) for k in _RESTORE_KEYS}
    try:
        yield
    finally:
        for k, v in snapshot.items():
            object.__setattr__(_settings, k, v)
        _reset_for_tests()


# ===================== resolve =====================


def test_resolve_by_key() -> None:
    s = _resolve_setting("tts_voice_style")
    assert s.key == "tts_voice_style"


def test_resolve_by_voice_alias_case_insensitive() -> None:
    s = _resolve_setting("becca")
    assert s.key == "companion_runtime_enabled"
    s2 = _resolve_setting("BECCA")
    assert s2.key == "companion_runtime_enabled"


def test_resolve_empty_raises() -> None:
    from augmentum.registry.registry import RegistryError

    with pytest.raises(RegistryError, match="non-empty"):
        _resolve_setting("")


def test_resolve_unknown_raises() -> None:
    from augmentum.registry.registry import RegistryError

    with pytest.raises(RegistryError, match="not registered"):
        _resolve_setting("totally_nonexistent_setting_zzzz")


# ===================== coercion =====================


def test_coerce_bool_accepts_truthy_strings() -> None:
    s = get_registry().get("tts_kokoro_hbe")
    assert _coerce_value(s, "true") is True
    assert _coerce_value(s, "1") is True
    assert _coerce_value(s, "yes") is True
    assert _coerce_value(s, "false") is False
    assert _coerce_value(s, "0") is False


def test_coerce_int_clamps_via_range() -> None:
    s = get_registry().get("narrative_smart_retrieval_count")
    assert _coerce_value(s, 5) == 5
    with pytest.raises(ValueError, match="below min"):
        _coerce_value(s, 0)
    with pytest.raises(ValueError, match="above max"):
        _coerce_value(s, 21)


def test_coerce_float_range() -> None:
    s = get_registry().get("voice_smart_turn_threshold")
    assert _coerce_value(s, 0.5) == 0.5
    with pytest.raises(ValueError, match="below min"):
        _coerce_value(s, 0.0)
    with pytest.raises(ValueError, match="above max"):
        _coerce_value(s, 1.0)


def test_coerce_enum_rejects_invalid() -> None:
    s = get_registry().get("narrative_memory_mode")
    assert _coerce_value(s, "lite") == "lite"
    with pytest.raises(ValueError, match="not in enum_values"):
        _coerce_value(s, "fancy")


def test_coerce_str_enforces_max_length() -> None:
    s = get_registry().get("tts_voice_style")
    long = "x" * 999
    with pytest.raises(ValueError, match="exceeds max"):
        _coerce_value(s, long)


def test_coerce_tristate_handles_auto_and_none() -> None:
    s = get_registry().get("engine_multislot_enabled")
    assert _coerce_value(s, None) is None
    assert _coerce_value(s, "auto") is None
    assert _coerce_value(s, "true") is True
    assert _coerce_value(s, "false") is False


def test_coerce_bool_strict_against_int_one_as_bool() -> None:
    """For settings whose kind=int (not bool), coercion should still
    work — the strictness is only that bool kinds reject 1.0 truthiness
    via type. int kinds happily coerce."""
    s = get_registry().get("narrative_smart_retrieval_count")
    assert _coerce_value(s, "5") == 5
    assert _coerce_value(s, 5.0) == 5  # int(5.0) is 5


# ===================== trust check =====================


def test_trust_local_reversible_always_allowed() -> None:
    s = get_registry().get("tts_voice_style")
    assert s.trust_tier == "local_reversible"
    allowed, _ = _trust_check(s, confirm=False, is_admin=False)
    assert allowed is True


def test_trust_local_significant_requires_confirm_or_admin() -> None:
    s = get_registry().get("companion_runtime_enabled")
    assert s.trust_tier == "local_significant"
    allowed, reason = _trust_check(s, confirm=False, is_admin=False)
    assert not allowed
    assert "confirm" in reason.lower()
    allowed2, _ = _trust_check(s, confirm=True, is_admin=False)
    assert allowed2 is True
    allowed3, _ = _trust_check(s, confirm=False, is_admin=True)
    assert allowed3 is True


def test_trust_admin_only_blocks_non_admin() -> None:
    s = get_registry().get("engine_use_jinja_template")
    assert s.trust_tier == "admin_only"
    allowed, reason = _trust_check(s, confirm=True, is_admin=False)
    assert not allowed
    assert "admin" in reason.lower()
    allowed2, _ = _trust_check(s, confirm=False, is_admin=True)
    assert allowed2 is True


# ===================== SettingGetTool =====================


@pytest.mark.asyncio
async def test_get_tool_returns_metadata() -> None:
    tool = SettingGetTool()
    result = await tool.execute(key="tts_voice_style")
    assert result.success
    assert result.metadata["key"] == "tts_voice_style"
    assert result.metadata["kind"] == "str"
    assert result.metadata["default"] == ""
    assert result.metadata["trust_tier"] == "local_reversible"


@pytest.mark.asyncio
async def test_get_tool_resolves_by_voice_alias() -> None:
    tool = SettingGetTool()
    result = await tool.execute(key="becca")
    assert result.success
    assert result.metadata["key"] == "companion_runtime_enabled"


@pytest.mark.asyncio
async def test_get_tool_unknown_key_returns_failure() -> None:
    tool = SettingGetTool()
    result = await tool.execute(key="not_a_real_setting")
    assert not result.success
    assert result.validation_error


@pytest.mark.asyncio
async def test_get_tool_surface_exposure() -> None:
    """The get tool must be reachable from chat / voice (core) / coder /
    companion — the Becca substrate contract."""
    tool = SettingGetTool()
    s = tool.surfaces
    assert s.chat is True
    assert s.voice == "core"
    assert s.coder is True
    assert s.companion is True


# ===================== SettingSetTool =====================


@pytest.mark.asyncio
async def test_set_tool_local_reversible_applies_without_confirm() -> None:
    tool = SettingSetTool()
    result = await tool.execute(
        key="tts_voice_style",
        value="speak warmly",
        _context={"settings_store": None, "is_admin": False},
    )
    assert result.success
    assert result.metadata["current"] == "speak warmly"
    assert result.metadata["changed"] is True


@pytest.mark.asyncio
async def test_set_tool_local_significant_blocks_without_confirm() -> None:
    tool = SettingSetTool()
    result = await tool.execute(
        key="companion_runtime_enabled",
        value=True,
        _context={"is_admin": False},
    )
    assert not result.success
    assert "confirm" in result.error.lower()
    assert result.metadata.get("confirm_required") is True


@pytest.mark.asyncio
async def test_set_tool_local_significant_succeeds_with_confirm() -> None:
    tool = SettingSetTool()
    result = await tool.execute(
        key="companion_runtime_enabled",
        value=True,
        confirm=True,
        _context={"is_admin": False},
    )
    assert result.success


@pytest.mark.asyncio
async def test_set_tool_admin_only_blocks_non_admin_even_with_confirm() -> None:
    tool = SettingSetTool()
    result = await tool.execute(
        key="engine_use_jinja_template",
        value=False,
        confirm=True,
        _context={"is_admin": False},
    )
    assert not result.success
    assert "admin" in result.error.lower()


@pytest.mark.asyncio
async def test_set_tool_admin_only_succeeds_for_admin() -> None:
    tool = SettingSetTool()
    result = await tool.execute(
        key="engine_use_jinja_template",
        value=False,
        _context={"is_admin": True},
    )
    assert result.success
    assert result.metadata["restart_required"] is True


@pytest.mark.asyncio
async def test_set_tool_resolves_voice_alias() -> None:
    """The Eva-vignette: voice says 'becca off', alias resolves to
    companion_runtime_enabled, value coerces, change applies."""
    tool = SettingSetTool()
    result = await tool.execute(
        key="becca",
        value=True,
        confirm=True,
        _context={"is_admin": False},
    )
    assert result.success
    assert result.metadata["key"] == "companion_runtime_enabled"


@pytest.mark.asyncio
async def test_set_tool_rejects_out_of_range() -> None:
    tool = SettingSetTool()
    result = await tool.execute(
        key="voice_smart_turn_threshold",
        value=1.5,  # above max 0.95
        _context={"is_admin": True},
    )
    assert not result.success
    assert result.validation_error
    assert "above max" in result.error.lower()


@pytest.mark.asyncio
async def test_set_tool_rejects_invalid_enum() -> None:
    tool = SettingSetTool()
    result = await tool.execute(
        key="narrative_memory_mode",
        value="fancy",
        _context={"is_admin": True},
    )
    assert not result.success
    assert "not in enum_values" in result.error.lower()


@pytest.mark.asyncio
async def test_set_tool_persists_to_store_when_provided() -> None:
    tool = SettingSetTool()
    fake_store = MagicMock()
    fake_store.set = AsyncMock()
    fake_store.delete = AsyncMock()
    result = await tool.execute(
        key="tts_voice_style",
        value="warm",
        _context={"settings_store": fake_store, "is_admin": True},
    )
    assert result.success
    assert result.metadata["persisted"] is True
    fake_store.set.assert_awaited_once_with("tts_voice_style", "warm")


@pytest.mark.asyncio
async def test_set_tool_persists_tristate_as_delete_when_none() -> None:
    """A tristate with a persistent override gets the override removed
    when set to 'auto' / None. Verifies the contract that None on
    tristate is a clear-override, not a value write."""
    tool = SettingSetTool()
    fake_store = MagicMock()
    fake_store.set = AsyncMock()
    fake_store.delete = AsyncMock()
    # First put the tristate into the True state so the in-process
    # value differs from None on the next call.
    await tool.execute(
        key="engine_multislot_enabled",
        value=True,
        _context={"settings_store": fake_store, "is_admin": True},
    )
    fake_store.set.reset_mock()
    fake_store.delete.reset_mock()
    # Now reset to auto.
    result = await tool.execute(
        key="engine_multislot_enabled",
        value="auto",
        _context={"settings_store": fake_store, "is_admin": True},
    )
    assert result.success
    fake_store.delete.assert_awaited_once_with("engine_multislot_enabled")
    fake_store.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_tool_no_op_when_value_unchanged() -> None:
    """Setting a value to its current value should succeed but report
    ``changed=False`` so downstream callers can short-circuit."""
    tool = SettingSetTool()
    # Default for tts_voice_style is "" — set it to "" again.
    result = await tool.execute(
        key="tts_voice_style",
        value="",
        _context={"is_admin": True},
    )
    assert result.success
    assert result.metadata["changed"] is False


@pytest.mark.asyncio
async def test_set_tool_missing_value_is_validation_error() -> None:
    tool = SettingSetTool()
    result = await tool.execute(
        key="tts_voice_style",
        _context={"is_admin": True},
    )
    assert not result.success
    assert result.validation_error


@pytest.mark.asyncio
async def test_set_tool_surface_exposure() -> None:
    """The set tool must reach chat / voice (interactive — not core,
    because mutations are higher-stakes) / coder / companion."""
    tool = SettingSetTool()
    s = tool.surfaces
    assert s.chat is True
    assert s.voice == "interactive"
    assert s.coder is True
    assert s.companion is True
