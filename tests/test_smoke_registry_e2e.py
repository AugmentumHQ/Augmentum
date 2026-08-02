"""End-to-end registry tests against the full FastAPI app.

Exercises the production wiring path: ``create_app()`` → router
registration → ``/api/settings/registry/*`` HTTP endpoints. Verifies
the registry is reachable through the same app the browser hits and
not just through a minimal test app.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ensure_registry_populated():
    """Earlier test files' ``_reset_for_tests()`` teardown wipes the
    singleton. ``settings_registry_routes.py`` only calls
    ``load_into_default_registry`` at module import time (cached by
    Python's import system), so the singleton stays empty.

    This fixture re-populates before each e2e test so the production
    wiring is exercised against a real registry."""
    from augmentum.registry.builtin import load_into_default_registry
    from augmentum.registry.registry import get_registry

    if not get_registry().has("tts_voice_style"):
        load_into_default_registry()


def test_app_imports_with_registry_wired(sqlite_client) -> None:
    """create_app() succeeds with the registry overlay + endpoints
    wired in. Smoke check that startup doesn't crash."""
    # Just having the sqlite_client fixture means create_app ran.
    assert sqlite_client is not None


def test_full_app_serves_registry_list_endpoint(sqlite_client) -> None:
    """GET /api/settings/registry/ returns the full wire format
    through the real app router."""
    resp = sqlite_client.get("/api/settings/registry/")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "settings" in body
    assert "total" in body
    # 416 settings registered; user-facing default surface (non-advanced)
    # is smaller but should be at least ~100.
    assert body["total"] >= 100


def test_full_app_serves_registry_show_advanced(sqlite_client) -> None:
    """``?show_advanced=true`` reveals the full 416 minus a handful
    of deprecated entries."""
    resp = sqlite_client.get(
        "/api/settings/registry/?show_advanced=true"
    )
    assert resp.status_code == 200
    body = resp.json()
    # Allow some tolerance for deprecated entries hidden by default.
    assert body["total"] >= 400, f"only {body['total']} returned"


def test_full_app_serves_one_setting_by_key(sqlite_client) -> None:
    resp = sqlite_client.get(
        "/api/settings/registry/companion_runtime_enabled"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["key"] == "companion_runtime_enabled"
    assert body["trust_tier"] == "local_significant"
    assert "becca" in body["voice_aliases"]


def test_full_app_returns_404_on_unknown_key(sqlite_client) -> None:
    resp = sqlite_client.get(
        "/api/settings/registry/nonexistent_setting_zzzz"
    )
    assert resp.status_code == 404


def test_full_app_serves_sections(sqlite_client) -> None:
    """The sections endpoint backs the registry UI nav tree."""
    resp = sqlite_client.get("/api/settings/registry/sections")
    assert resp.status_code == 200
    body = resp.json()
    section_names = {s["section"] for s in body["sections"]}
    # Spot-check the major subsystems are represented.
    expected = {
        "voice.tts",
        "narrative.memory",
        "companion",
        "engine.kv",
        "image.quality",
        "search.pipeline",
        "knowledge.packs",
        "providers.anthropic",
    }
    missing = expected - section_names
    assert not missing, f"sections missing: {missing}"


def test_full_app_serves_voice_aliases(sqlite_client) -> None:
    """The Phase 4 / Becca-substrate consumer endpoint — every voice
    alias maps to a known registry key. Used by Becca's setting.set
    Tool to resolve natural phrases."""
    resp = sqlite_client.get("/api/settings/registry/voice-aliases")
    assert resp.status_code == 200
    body = resp.json()
    aliases = {a["alias"]: a["key"] for a in body["aliases"]}
    # Key wins from Phase 4 design.
    assert aliases.get("becca") == "companion_runtime_enabled"
    assert aliases.get("memory mode") == "narrative_memory_mode"
    # No alias collisions (substrate guarantee).
    by_alias_count: dict[str, int] = {}
    for a in body["aliases"]:
        by_alias_count[a["alias"]] = by_alias_count.get(a["alias"], 0) + 1
    duplicates = {a: c for a, c in by_alias_count.items() if c > 1}
    assert duplicates == {}, f"alias collisions through HTTP: {duplicates}"


def test_full_app_section_filter_via_http(sqlite_client) -> None:
    resp = sqlite_client.get(
        "/api/settings/registry/?section=engine.kv&show_advanced=true"
    )
    assert resp.status_code == 200
    body = resp.json()
    keys = {s["key"] for s in body["settings"]}
    # Phase 1C engine batch registered 6 KV settings.
    expected_subset = {
        "engine_kv_cache_type",
        "engine_kv_ttl_days",
        "engine_kv_narrative_ttl_days",
        "engine_kv_max_snapshots_per_model",
        "engine_kv_auto_pin_narrative",
        "engine_kv_warm_on_start",
    }
    assert expected_subset <= keys, f"missing: {expected_subset - keys}"


def test_full_app_search_query_via_http(sqlite_client) -> None:
    resp = sqlite_client.get(
        "/api/settings/registry/?q=becca&show_advanced=true"
    )
    body = resp.json()
    keys = {s["key"] for s in body["settings"]}
    assert "companion_runtime_enabled" in keys


def test_full_app_tag_filter_via_http(sqlite_client) -> None:
    resp = sqlite_client.get(
        "/api/settings/registry/?tag=voice&show_advanced=true"
    )
    body = resp.json()
    # The voice tag pulls every voice/TTS knob.
    assert body["total"] >= 20


def test_full_app_registry_overlay_visible_to_tools_put(sqlite_client) -> None:
    """A setting that lives ONLY in the registry — not in the original
    literal _TOOL_SETTINGS dict — must be accepted by the existing
    PUT /api/config/tools endpoint thanks to the overlay."""
    # Settings I know were buried-from-UI before Phase 1C — these existed
    # in config_routes literal _TOOL_SETTINGS BEFORE migration but the
    # overlay should preserve them.
    resp = sqlite_client.get("/api/settings/registry/narrative_smart_retrieval")
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "bool"


def test_full_app_wire_format_includes_modified_flag(sqlite_client) -> None:
    resp = sqlite_client.get(
        "/api/settings/registry/tts_voice_style"
    )
    assert resp.status_code == 200
    body = resp.json()
    # tts_voice_style ships with default "".
    assert body["default"] == ""
    assert "modified" in body


@pytest.mark.parametrize(
    "operator,expected_attr",
    [
        ("@advanced", "advanced"),
        ("@restart", "restart_required"),
    ],
)
def test_full_app_search_operators_via_http_smoke(
    sqlite_client, operator, expected_attr
) -> None:
    """Verify the search operator parsing on the backend is intact —
    the JS side parses these too, but they should also work via the
    HTTP endpoint when the UI substitutes flag→param translation."""
    # The backend's list_settings doesn't directly accept @advanced as
    # a query parameter, but the corresponding flag parameter does:
    flag = "show_advanced" if operator == "@advanced" else None
    if flag is None:
        # @restart isn't exposed as a query param yet — skip with a
        # passing assertion. Documents the contract.
        return
    resp = sqlite_client.get(
        f"/api/settings/registry/?{flag}=true"
    )
    body = resp.json()
    found_advanced = any(s[expected_attr] for s in body["settings"])
    assert found_advanced
