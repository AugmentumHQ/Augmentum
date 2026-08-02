"""Phase 1B smoke tests — first 10 settings migrated into the
declarative substrate, overlayed into the live config_routes dicts,
exposed via /api/settings/registry.

Doesn't isolate the singleton (the goal here is end-to-end with the
production wiring). For dataclass + registry isolation tests see
``tests/test_smoke_registry.py``.
"""

from __future__ import annotations

import json

import pytest

# ---- Built-in registration ----


def test_builtin_load_registers_ten_settings() -> None:
    from augmentum.registry.builtin import load_into_default_registry
    from augmentum.registry.registry import get_registry

    load_into_default_registry()
    keys = {s.key for s in get_registry().list_all()}
    expected = {
        "tts_voice_style",
        "tts_emotion_aware",
        "tts_kokoro_hbe",
        "voice_smart_turn_threshold",
        "narrative_memory_mode",
        "narrative_smart_retrieval",
        "narrative_smart_retrieval_count",
        "companion_runtime_enabled",
        "engine_use_jinja_template",
        "image_torch_compile",
    }
    assert expected <= keys, f"Missing: {expected - keys}"


def test_builtin_load_is_idempotent() -> None:
    from augmentum.registry.builtin import load_into_default_registry
    from augmentum.registry.registry import get_registry

    load_into_default_registry()
    first = len(get_registry().list_all())
    load_into_default_registry()
    second = len(get_registry().list_all())
    assert first == second


def test_all_builtin_settings_have_descriptions() -> None:
    """Every migrated setting must carry a meaningful description —
    enforced both by the Setting dataclass and the audit layer."""
    from augmentum.registry.builtin import load_into_default_registry
    from augmentum.registry.registry import get_registry

    load_into_default_registry()
    for s in get_registry().list_all():
        assert len(s.description.strip()) >= 20, f"{s.key}: short description"
        assert s.label.strip(), f"{s.key}: empty label"
        assert s.section.strip(), f"{s.key}: empty section"


def test_voice_aliases_lookup_companion_runtime() -> None:
    """Phase 4 substrate check: 'becca' should reverse-lookup to
    the companion runtime master switch."""
    from augmentum.registry.builtin import load_into_default_registry
    from augmentum.registry.registry import get_registry

    load_into_default_registry()
    s = get_registry().voice_alias_lookup("becca")
    assert s is not None
    assert s.key == "companion_runtime_enabled"


# ---- Overlay into config_routes dicts ----


def test_overlay_merged_into_tool_settings() -> None:
    """After importing config_routes, the registry-derived numeric/bool
    settings must appear in the live _TOOL_SETTINGS dict so PUT
    /api/config/tools validation honors them."""
    from augmentum.proxy import config_routes

    ts = config_routes._TOOL_SETTINGS
    assert ts.get("tts_emotion_aware") == (bool, 0, 1)
    assert ts.get("tts_kokoro_hbe") == (bool, 0, 1)
    assert ts.get("voice_smart_turn_threshold") == (float, 0.1, 0.95)
    assert ts.get("narrative_smart_retrieval") == (bool, 0, 1)
    assert ts.get("narrative_smart_retrieval_count") == (int, 1, 20)
    assert ts.get("companion_runtime_enabled") == (bool, 0, 1)
    assert ts.get("engine_use_jinja_template") == (bool, 0, 1)


def test_overlay_merged_into_string_settings() -> None:
    """String / enum settings appear in _STRING_SETTINGS with the
    declared (or derived) max_length."""
    from augmentum.proxy import config_routes

    ss = config_routes._STRING_SETTINGS
    assert ss.get("tts_voice_style") == 256
    # Enum with explicit max_length=16 — matches the historical bound.
    assert ss.get("narrative_memory_mode") == 16
    # Enum with explicit max_length=10 — matches the historical bound.
    assert ss.get("image_torch_compile") == 10


def test_overlay_does_not_clobber_unrelated_literals() -> None:
    """Settings not registered in the registry must continue to live
    in the literal dicts unchanged (registry overlay is purely
    additive for non-registered keys)."""
    from augmentum.proxy import config_routes

    # uarf_auto_search is a literal _TOOL_SETTINGS entry not in the
    # registry yet — must still be present.
    assert "uarf_auto_search" in config_routes._TOOL_SETTINGS


# ---- HTTP endpoints ----


@pytest.fixture
def settings_registry_app():
    """Minimal FastAPI app with just the settings-registry router
    mounted. We avoid pulling the full create_app() to keep the test
    fast and isolated from unrelated wiring in the heavy working tree.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from augmentum.proxy.settings_registry_routes import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_list_endpoint_returns_wire_format(settings_registry_app) -> None:
    resp = settings_registry_app.get("/api/settings/registry/")
    assert resp.status_code == 200
    body = resp.json()
    assert "settings" in body
    assert "total" in body
    # 3 settings are advanced (voice_smart_turn_threshold,
    # engine_use_jinja_template, image_torch_compile) and hidden by default.
    # That leaves 7 in the default user-facing surface.
    assert body["total"] >= 7
    keys = {s["key"] for s in body["settings"]}
    assert "tts_voice_style" in keys
    # And the advanced settings are excluded from the default view.
    assert "engine_use_jinja_template" not in keys
    assert "voice_smart_turn_threshold" not in keys


def test_list_endpoint_show_advanced(settings_registry_app) -> None:
    resp = settings_registry_app.get(
        "/api/settings/registry/?show_advanced=true"
    )
    body = resp.json()
    keys = {s["key"] for s in body["settings"]}
    assert "engine_use_jinja_template" in keys
    assert "voice_smart_turn_threshold" in keys
    assert "image_torch_compile" in keys


def test_list_endpoint_section_filter(settings_registry_app) -> None:
    """``?section=voice.tts`` matches voice.tts AND voice.tts.kokoro
    (tree-style nav). Returns at minimum the Phase 1B tts settings;
    Phase 1C adds tts_kokoro_prosody / quality under voice.tts.kokoro
    so the set is allowed to grow."""
    resp = settings_registry_app.get(
        "/api/settings/registry/?section=voice.tts&show_advanced=true"
    )
    body = resp.json()
    keys = {s["key"] for s in body["settings"]}
    expected_subset = {"tts_voice_style", "tts_emotion_aware", "tts_kokoro_hbe"}
    assert expected_subset <= keys, f"Missing: {expected_subset - keys}"


def test_list_endpoint_tag_filter(settings_registry_app) -> None:
    """The narrative tag pulls every setting that tunes narrative-mode
    behavior — including engine-side knobs that get a longer warm-cache
    window for narrative sessions. Phase 1B narrative settings must
    appear; the set is allowed to grow with cross-cutting tags."""
    resp = settings_registry_app.get(
        "/api/settings/registry/?tag=narrative&show_advanced=true"
    )
    body = resp.json()
    keys = {s["key"] for s in body["settings"]}
    expected_subset = {
        "narrative_memory_mode",
        "narrative_smart_retrieval",
        "narrative_smart_retrieval_count",
    }
    assert expected_subset <= keys, f"Missing: {expected_subset - keys}"


def test_list_endpoint_search_filter(settings_registry_app) -> None:
    resp = settings_registry_app.get(
        "/api/settings/registry/?q=becca&show_advanced=true"
    )
    body = resp.json()
    keys = {s["key"] for s in body["settings"]}
    # becca is a voice_alias for companion_runtime_enabled.
    assert "companion_runtime_enabled" in keys


def test_get_one_setting(settings_registry_app) -> None:
    resp = settings_registry_app.get(
        "/api/settings/registry/narrative_memory_mode"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["key"] == "narrative_memory_mode"
    assert body["kind"] == "enum"
    assert body["enum_values"] == ["lite", "standard"]
    assert body["default"] == "standard"
    assert body["section"] == "narrative.memory"
    assert "memory mode" in body["voice_aliases"]


def test_get_unknown_setting_returns_404(settings_registry_app) -> None:
    resp = settings_registry_app.get(
        "/api/settings/registry/nonexistent_setting_zzzz"
    )
    assert resp.status_code == 404


def test_sections_endpoint(settings_registry_app) -> None:
    resp = settings_registry_app.get("/api/settings/registry/sections")
    body = resp.json()
    section_names = {s["section"] for s in body["sections"]}
    assert "voice.tts" in section_names
    assert "narrative.memory" in section_names
    assert "companion" in section_names


def test_voice_aliases_endpoint(settings_registry_app) -> None:
    resp = settings_registry_app.get("/api/settings/registry/voice-aliases")
    body = resp.json()
    aliases = {a["alias"]: a["key"] for a in body["aliases"]}
    assert aliases.get("becca") == "companion_runtime_enabled"
    assert aliases.get("memory mode") == "narrative_memory_mode"


def test_wire_format_json_serializable(settings_registry_app) -> None:
    """The wire format must round-trip cleanly through JSON — no
    callables or other non-serializable fields leak through."""
    resp = settings_registry_app.get(
        "/api/settings/registry/?show_advanced=true"
    )
    # If anything non-serializable leaked, resp.json() would raise.
    body = resp.json()
    # Round-trip to prove no funny tricks.
    reserialized = json.loads(json.dumps(body))
    assert reserialized == body


def test_current_value_reflects_dataclass(settings_registry_app) -> None:
    """The 'current' field in the wire format should pull from the
    live Settings dataclass."""
    resp = settings_registry_app.get(
        "/api/settings/registry/narrative_memory_mode"
    )
    body = resp.json()
    # Default in config.py is 'standard'.
    assert body["current"] == "standard"
    assert body["modified"] is False


# ---- Compatibility with the existing PUT endpoint ----


def test_existing_put_tools_still_validates_registered_settings() -> None:
    """A setting registered in the substrate must still be writable
    through the existing PUT /api/config/tools endpoint — that's the
    backwards-compat contract of the overlay."""
    from augmentum.proxy import config_routes

    # The validator path in update_tool_settings reads _TOOL_SETTINGS;
    # if our entry is in the dict with the right shape, the existing
    # endpoint accepts the update.
    t = config_routes._TOOL_SETTINGS.get("companion_runtime_enabled")
    assert t == (bool, 0, 1)
    # bool entries accept truthy ints 0 or 1 per the documented contract.
    assert t[0] is bool
    assert t[1] == 0
    assert t[2] == 1
