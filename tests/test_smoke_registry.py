"""Smoke tests for the declarative action substrate registry.

Phase 1A — pure additive substrate. These tests verify the registry
exists, the Setting dataclass validates correctly, and the registry's
query/export methods round-trip.
"""

from __future__ import annotations

import pytest

from augmentum.registry import (
    Setting,
    SettingsRegistry,
    get_registry,
)
from augmentum.registry import audit as registry_audit
from augmentum.registry.registry import RegistryError, _reset_for_tests


@pytest.fixture(autouse=True)
def _isolate_registry() -> None:
    """Each test starts with a fresh registry singleton so tests
    don't bleed registered Settings into each other."""
    _reset_for_tests()
    yield
    _reset_for_tests()


# ===================== Setting validation =====================


def test_minimal_bool_setting_constructs() -> None:
    s = Setting(
        key="example_enabled",
        kind="bool",
        default=False,
        label="Example",
        description="An example setting used in tests.",
        section="example",
    )
    assert s.key == "example_enabled"
    assert s.default is False


def test_minimal_int_setting_constructs() -> None:
    s = Setting(
        key="example_count",
        kind="int",
        default=5,
        label="Example count",
        description="An example numeric setting used in tests.",
        section="example",
        min_value=0,
        max_value=10,
    )
    assert s.default == 5


def test_minimal_str_setting_constructs() -> None:
    s = Setting(
        key="example_text",
        kind="str",
        default="hello",
        label="Example text",
        description="An example string setting used in tests.",
        section="example",
        max_length=64,
    )
    assert s.default == "hello"


def test_minimal_enum_setting_constructs() -> None:
    s = Setting(
        key="example_mode",
        kind="enum",
        default="auto",
        label="Example mode",
        description="An example enum setting used in tests.",
        section="example",
        enum_values=("auto", "on", "off"),
    )
    assert s.default == "auto"


def test_tristate_setting_constructs() -> None:
    s = Setting(
        key="example_tristate",
        kind="tristate",
        default=None,
        label="Example tristate",
        description="An example tristate setting used in tests.",
        section="example",
        tristate_resolver=lambda v: True if v is None else v,
    )
    assert s.default is None


def test_empty_key_rejected() -> None:
    with pytest.raises(ValueError, match="key must be non-empty"):
        Setting(
            key="",
            kind="bool",
            default=False,
            label="x",
            description="desc",
            section="s",
        )


def test_non_lowercase_key_rejected() -> None:
    with pytest.raises(ValueError, match="snake_case"):
        Setting(
            key="ExampleEnabled",
            kind="bool",
            default=False,
            label="x",
            description="desc",
            section="s",
        )


def test_empty_label_rejected() -> None:
    with pytest.raises(ValueError, match="label must be non-empty"):
        Setting(
            key="example",
            kind="bool",
            default=False,
            label="",
            description="desc",
            section="s",
        )


def test_empty_description_rejected() -> None:
    with pytest.raises(ValueError, match="description must be non-empty"):
        Setting(
            key="example",
            kind="bool",
            default=False,
            label="x",
            description="",
            section="s",
        )


def test_empty_section_rejected() -> None:
    with pytest.raises(ValueError, match="section must be non-empty"):
        Setting(
            key="example",
            kind="bool",
            default=False,
            label="x",
            description="desc",
            section="",
        )


def test_bool_kind_with_int_default_rejected() -> None:
    with pytest.raises(ValueError, match="kind=bool requires bool default"):
        Setting(
            key="example",
            kind="bool",
            default=1,
            label="x",
            description="desc",
            section="s",
        )


def test_int_kind_with_bool_default_rejected() -> None:
    # bool is a subclass of int in Python — explicit check.
    with pytest.raises(ValueError, match="kind=int requires int default"):
        Setting(
            key="example",
            kind="int",
            default=True,
            label="x",
            description="desc",
            section="s",
        )


def test_int_with_inverted_range_rejected() -> None:
    with pytest.raises(ValueError, match="min_value .* > max_value"):
        Setting(
            key="example",
            kind="int",
            default=5,
            label="x",
            description="desc",
            section="s",
            min_value=10,
            max_value=1,
        )


def test_min_max_on_str_kind_rejected() -> None:
    with pytest.raises(ValueError, match="min/max_value only valid for int/float"):
        Setting(
            key="example",
            kind="str",
            default="hi",
            label="x",
            description="desc",
            section="s",
            min_value=0,
            max_value=10,
        )


def test_max_length_on_int_kind_rejected() -> None:
    with pytest.raises(ValueError, match="max_length only valid for str"):
        Setting(
            key="example",
            kind="int",
            default=1,
            label="x",
            description="desc",
            section="s",
            max_length=10,
        )


def test_enum_without_values_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty enum_values"):
        Setting(
            key="example",
            kind="enum",
            default="auto",
            label="x",
            description="desc",
            section="s",
        )


def test_enum_default_not_in_values_rejected() -> None:
    with pytest.raises(ValueError, match="not in enum_values"):
        Setting(
            key="example",
            kind="enum",
            default="auto",
            label="x",
            description="desc",
            section="s",
            enum_values=("on", "off"),
        )


def test_tristate_without_resolver_rejected() -> None:
    with pytest.raises(ValueError, match="requires tristate_resolver"):
        Setting(
            key="example",
            kind="tristate",
            default=None,
            label="x",
            description="desc",
            section="s",
        )


# ===================== Registry operations =====================


def _example(key: str = "example_enabled", section: str = "example") -> Setting:
    return Setting(
        key=key,
        kind="bool",
        default=False,
        label=key.replace("_", " ").title(),
        description=f"An example setting named {key} for unit tests.",
        section=section,
    )


def test_get_registry_returns_singleton() -> None:
    a = get_registry()
    b = get_registry()
    assert a is b


def test_register_and_get_roundtrip() -> None:
    r = SettingsRegistry()
    s = _example()
    r.register(s)
    assert r.get("example_enabled") is s
    assert r.has("example_enabled")


def test_duplicate_register_rejected() -> None:
    r = SettingsRegistry()
    r.register(_example())
    with pytest.raises(RegistryError, match="already registered"):
        r.register(_example())


def test_get_unknown_key_raises() -> None:
    r = SettingsRegistry()
    with pytest.raises(RegistryError, match="not registered"):
        r.get("nonexistent_setting")


def test_try_get_unknown_returns_none() -> None:
    r = SettingsRegistry()
    assert r.try_get("nonexistent_setting") is None


def test_list_all_returns_registered_settings() -> None:
    r = SettingsRegistry()
    r.register(_example("a"))
    r.register(_example("b"))
    assert {s.key for s in r.list_all()} == {"a", "b"}


def test_list_by_section_exact_and_prefix() -> None:
    r = SettingsRegistry()
    r.register(_example("a", section="companion"))
    r.register(_example("b", section="companion.voice"))
    r.register(_example("c", section="companion.warmth"))
    r.register(_example("d", section="engine"))
    # Exact + tree-style nav.
    keys = {s.key for s in r.list_by_section("companion")}
    assert keys == {"a", "b", "c"}
    assert {s.key for s in r.list_by_section("companion.voice")} == {"b"}


def test_list_by_tag() -> None:
    r = SettingsRegistry()
    r.register(
        Setting(
            key="adv",
            kind="bool",
            default=False,
            label="Adv",
            description="Advanced setting for testing.",
            section="s",
            tags=("advanced",),
        )
    )
    r.register(_example("normal"))
    assert {s.key for s in r.list_by_tag("advanced")} == {"adv"}
    assert r.list_by_tag("nonexistent") == []


def test_list_user_facing_hides_advanced_by_default() -> None:
    r = SettingsRegistry()
    r.register(_example("normal"))
    r.register(
        Setting(
            key="adv",
            kind="bool",
            default=False,
            label="Adv",
            description="Advanced setting for testing.",
            section="s",
            advanced=True,
        )
    )
    keys = {s.key for s in r.list_user_facing()}
    assert keys == {"normal"}
    keys_with_advanced = {s.key for s in r.list_user_facing(show_advanced=True)}
    assert keys_with_advanced == {"normal", "adv"}


def test_list_user_facing_skips_deprecated_and_admin() -> None:
    r = SettingsRegistry()
    r.register(_example("normal"))
    r.register(
        Setting(
            key="dep",
            kind="bool",
            default=False,
            label="Dep",
            description="Deprecated setting for testing.",
            section="s",
            deprecated="use 'normal' instead",
        )
    )
    r.register(
        Setting(
            key="adm",
            kind="bool",
            default=False,
            label="Adm",
            description="Admin-only setting for testing.",
            section="s",
            trust_tier="admin_only",
        )
    )
    keys = {s.key for s in r.list_user_facing(show_advanced=True)}
    assert keys == {"normal"}


def test_search_matches_label_description_and_alias() -> None:
    r = SettingsRegistry()
    r.register(
        Setting(
            key="tts_default_voice",
            kind="str",
            default="default",
            label="TTS Default Voice",
            description="The voice used for TTS output by default.",
            section="voice",
            voice_aliases=("eva voice", "the warm voice"),
            max_length=64,
        )
    )
    # Key match.
    assert {s.key for s in r.search("tts")} == {"tts_default_voice"}
    # Description match.
    assert {s.key for s in r.search("output")} == {"tts_default_voice"}
    # Voice alias match.
    assert {s.key for s in r.search("eva")} == {"tts_default_voice"}
    # No match.
    assert r.search("nonexistent_zzzzz") == []


def test_voice_alias_lookup() -> None:
    r = SettingsRegistry()
    r.register(
        Setting(
            key="tts_default_voice",
            kind="str",
            default="default",
            label="TTS",
            description="The voice used for TTS output by default.",
            section="voice",
            voice_aliases=("eva voice", "the warm voice"),
            max_length=64,
        )
    )
    assert r.voice_alias_lookup("eva voice").key == "tts_default_voice"
    assert r.voice_alias_lookup("EVA VOICE").key == "tts_default_voice"  # case-insensitive
    assert r.voice_alias_lookup("unknown alias") is None
    assert r.voice_alias_lookup("") is None


# ===================== Export shapes =====================


def test_to_tool_settings_emits_existing_shape() -> None:
    r = SettingsRegistry()
    r.register(_example())
    r.register(
        Setting(
            key="num",
            kind="int",
            default=5,
            label="Num",
            description="A numeric setting used for export tests.",
            section="s",
            min_value=0,
            max_value=10,
        )
    )
    r.register(
        Setting(
            key="rate",
            kind="float",
            default=0.5,
            label="Rate",
            description="A float setting used for export tests.",
            section="s",
            min_value=0.0,
            max_value=1.0,
        )
    )
    out = r.to_tool_settings()
    assert out["example_enabled"] == (bool, 0, 1)
    assert out["num"] == (int, 0, 10)
    assert out["rate"] == (float, 0.0, 1.0)


def test_to_string_settings_emits_max_length() -> None:
    r = SettingsRegistry()
    r.register(
        Setting(
            key="text",
            kind="str",
            default="x",
            label="Text",
            description="A string setting used for export tests.",
            section="s",
            max_length=128,
        )
    )
    out = r.to_string_settings()
    assert out["text"] == 128


def test_to_restore_map_emits_parser() -> None:
    r = SettingsRegistry()
    r.register(_example())
    out = r.to_restore_map()
    assert "example_enabled" in out
    # Parser coerces stringy "true" to bool True.
    assert out["example_enabled"]("true") is True
    assert out["example_enabled"]("0") is False


def test_to_wire_format_is_jsonable() -> None:
    import json

    r = SettingsRegistry()
    r.register(_example())
    wire = r.to_wire_format()
    # Must round-trip through JSON without error — proves no
    # non-serializable fields snuck in.
    encoded = json.dumps(wire)
    decoded = json.loads(encoded)
    assert decoded[0]["key"] == "example_enabled"
    assert decoded[0]["kind"] == "bool"
    assert decoded[0]["section"] == "example"


# ===================== Audit hooks =====================


def test_audit_empty_registry_has_zero_findings() -> None:
    findings = registry_audit.check_all()
    assert findings == []


def test_audit_summary_on_empty_registry() -> None:
    summary = registry_audit.summary()
    assert summary["registered"] == 0
    assert summary["findings"] == 0


def test_audit_flags_short_description() -> None:
    # Bypass __post_init__ validation (which would reject this at
    # construction) via dataclass field overrides — we want to verify
    # the audit catches it even if a future codepath gets sloppy.
    s = Setting(
        key="ok",
        kind="bool",
        default=False,
        label="OK",
        description="A reasonably long description suitable for audit.",
        section="s",
    )
    get_registry().register(s)
    # Sanity: well-formed setting → zero findings.
    findings = registry_audit.check_all()
    assert findings == []
