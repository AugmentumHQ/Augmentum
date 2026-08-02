"""Drift-detector tests — the safety net that gates Phase 1C bulk
migration. For every registered Setting, verify it agrees with the
historical 4 declaration sites; mismatches are surfaced as findings.
"""

from __future__ import annotations

import pytest

from augmentum.registry import Setting, SettingsRegistry
from augmentum.registry.registry import _reset_for_tests, get_registry


@pytest.fixture(autouse=True)
def _isolate_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


def test_drift_clean_when_phase_1b_loaded() -> None:
    """The 10 Phase 1B settings should report zero drift findings —
    this is the floor we hold throughout Phase 1C."""
    from augmentum.registry.builtin import load_into_default_registry
    from augmentum.registry.verify import check_all

    load_into_default_registry()
    findings = check_all()
    assert findings == [], f"Phase 1B settings have drift: {findings}"


def test_drift_detects_default_mismatch() -> None:
    """Mock a registered Setting whose default doesn't match the
    Settings dataclass — drift detector must surface it."""
    from augmentum.registry.verify import check_all

    r = get_registry()
    # tts_emotion_aware in config.py is False; register True.
    r.register(
        Setting(
            key="tts_emotion_aware",
            kind="bool",
            default=True,  # DELIBERATE MISMATCH
            label="Emotion aware",
            description="Test mismatch — should fire drift detector.",
            section="voice.tts",
        )
    )
    findings = check_all()
    keys_with_drift = {f["key"] for f in findings}
    assert "tts_emotion_aware" in keys_with_drift
    rules = {f["rule"] for f in findings if f["key"] == "tts_emotion_aware"}
    assert "registry_drift_default_mismatch" in rules


def test_drift_detects_no_persistence_target() -> None:
    """A Setting whose key doesn't exist as a config.py field AND
    isn't in _TOOL_SETTING_DEFAULTS must be flagged."""
    from augmentum.registry.verify import check_all

    r = get_registry()
    r.register(
        Setting(
            key="totally_fake_setting_zzz",
            kind="bool",
            default=False,
            label="Fake",
            description="No persistence target — should fire drift detector.",
            section="test",
        )
    )
    findings = check_all()
    rules = {f["rule"] for f in findings if f["key"] == "totally_fake_setting_zzz"}
    assert "registry_drift_no_persistence_target" in rules


def test_drift_skips_seeded_defaults_keys() -> None:
    """Settings in _TOOL_SETTING_DEFAULTS (the seeded-defaults table
    in config_routes) must NOT trigger no_persistence_target."""
    from augmentum.proxy.config_routes import _TOOL_SETTING_DEFAULTS
    from augmentum.registry.verify import check_all

    # Pick a known seeded key.
    if "tv_auto_update" in _TOOL_SETTING_DEFAULTS:
        r = get_registry()
        r.register(
            Setting(
                key="tv_auto_update",
                kind="bool",
                default=_TOOL_SETTING_DEFAULTS["tv_auto_update"],
                label="TV auto-update",
                description="Auto-pull TV catalog updates on startup.",
                section="tv",
            )
        )
        findings = check_all()
        rules = {f["rule"] for f in findings if f["key"] == "tv_auto_update"}
        assert "registry_drift_no_persistence_target" not in rules


def test_values_equal_handles_bool_vs_int_strictness() -> None:
    """bool is a subclass of int — drift detector must NOT treat
    True == 1 as a match."""
    from augmentum.registry.verify import _values_equal

    assert _values_equal(True, True)
    assert _values_equal(False, False)
    assert not _values_equal(True, 1)  # strict type check
    assert not _values_equal(False, 0)
    assert _values_equal(1, 1)
    assert _values_equal(1.0, 1.0)


def test_values_equal_handles_float_tolerance() -> None:
    """Tiny float differences from pydantic's typing should compare
    equal (1e-9 tolerance)."""
    from augmentum.registry.verify import _values_equal

    assert _values_equal(0.5, 0.5)
    assert _values_equal(0.5, 0.5 + 1e-15)
    assert not _values_equal(0.5, 0.6)


def test_summary_reports_finding_count_and_rules() -> None:
    """The summary helper rolls up findings for the audit's per-
    subsystem breakdown."""
    from augmentum.registry.verify import summary

    r = get_registry()
    r.register(
        Setting(
            key="fake_drift_key_for_summary",
            kind="bool",
            default=False,
            label="Fake",
            description="Will trigger no_persistence_target finding.",
            section="test",
        )
    )
    s = summary()
    assert s["registered"] == 1
    assert s["drift_findings"] >= 1
    assert s["by_rule"].get("registry_drift_no_persistence_target", 0) >= 1


def test_diagnostic_helpers_return_lists() -> None:
    """find_settings_with_no_label and find_settings_with_thin_descriptions
    are migration diagnostics — must return lists, not error."""
    from augmentum.registry.verify import (
        find_settings_with_no_label,
        find_settings_with_thin_descriptions,
    )

    r = get_registry()
    r.register(
        Setting(
            key="example",
            kind="bool",
            default=False,
            label="Example",
            description="Short.",  # 6 chars — counts as "thin"
            section="test",
        )
    )
    assert isinstance(find_settings_with_no_label(), list)
    thin = find_settings_with_thin_descriptions()
    assert any(s.key == "example" for s in thin)


def test_isolated_registry_does_not_leak_into_singleton() -> None:
    """A SettingsRegistry instance must NOT mutate the global
    singleton. This is the fixture's contract — guard against
    accidental imports that load builtin into the wrong registry."""
    isolated = SettingsRegistry()
    isolated.register(
        Setting(
            key="isolated_test_setting",
            kind="bool",
            default=False,
            label="Isolated",
            description="This must not appear in the global singleton.",
            section="test",
        )
    )
    assert isolated.has("isolated_test_setting")
    assert not get_registry().has("isolated_test_setting")
