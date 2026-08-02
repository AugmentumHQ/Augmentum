"""Tests for the companion intensity preset system.

The intensity dial is the user-facing control for "how much background
work does she do". Three levels (plus off + custom):

- off: not running at all
- minimal: kernel + chat presence, zero autonomous background
- balanced: + tick + journal + dreams + drift + today (current default-on)
- full: + initiative + consolidation + skills + creations
- custom: user overrode an individual flag; doesn't match any preset

Critical property: the user never gets forced into background LLM /
embedder work without opting in. minimal is the default when the
runtime is first enabled.

Coverage:
- Each preset has the expected flag bundle
- detect_intensity correctly identifies the preset matching live flags
- detect_intensity returns 'custom' when flags don't match
- detect_intensity returns 'off' when runtime master is off (regardless
  of individual flags)
- apply_preset writes all expected flags via a sync store stub
- API endpoint accepts valid levels, rejects invalid
- Status endpoint surfaces the current intensity correctly
- minimal preset has zero background-LLM flags enabled (the load-bearing
  property — verified by name set, not just emptiness)
"""

from __future__ import annotations

import pytest


# ── Preset shape ─────────────────────────────────────────────────────


def test_presets_exist_and_have_required_levels():
    from augmentum.companion.intensity import PRESETS

    assert "off" in PRESETS
    assert "minimal" in PRESETS
    assert "balanced" in PRESETS
    assert "full" in PRESETS

    for name, preset in PRESETS.items():
        assert preset.name == name
        assert preset.label  # non-empty
        assert preset.summary  # non-empty
        assert isinstance(preset.flags, dict)


def test_minimal_has_zero_background_llm_flags():
    """The load-bearing property: minimal must not turn on any flag
    that produces autonomous LLM or embedder work."""
    from augmentum.companion.intensity import PRESETS

    minimal = PRESETS["minimal"]
    # These flags produce background LLM calls (autonomous journal,
    # dreams, today reflection, consolidation, creations).
    background_llm_flags = [
        "companion_tick_enabled",       # autonomous activity scheduler
        "companion_journal_enabled",    # autonomous noticings → LLM
        "companion_dreams_enabled",     # dream cycles → LLM
        "companion_drift_audit_enabled",  # periodic embedder
        "companion_today_enabled",      # daily reflection → LLM
        "companion_creations_enabled",  # autonomous creation → LLM
        "companion_consolidation_enabled",  # proposal → LLM
        "companion_initiative_enabled",  # autonomous surfacing
        "companion_cultural_intake_enabled",  # RSS / network egress
        # Salience + voice journal write an embedding per turn.
        # Even though salience scoring itself is rules-based (no LLM),
        # the journal write triggers an embedder call per entry.
        # Minimal users shouldn't pay this cost.
        "companion_salience_enabled",
        "companion_voice_journal_enabled",
        # Skills inject embedder calls at compose time.
        "companion_skills_enabled",
    ]
    for flag in background_llm_flags:
        assert minimal.flags.get(flag, False) is False, (
            f"minimal must not enable {flag} (creates background work)"
        )


def test_balanced_enables_default_substrate():
    """Balanced should match the current default-on set so existing
    users don't get an unexpected behavior change when they discover
    the dial."""
    from augmentum.companion.intensity import PRESETS

    balanced = PRESETS["balanced"]
    # These were flipped to default-on in the Tier 1 visibility work.
    assert balanced.flags["companion_dispatch_enabled"] is True
    assert balanced.flags["companion_dispatch_routes_chat"] is True
    assert balanced.flags["companion_becca_direct_enabled"] is True
    assert balanced.flags["companion_salience_enabled"] is True
    assert balanced.flags["companion_voice_journal_enabled"] is True
    assert balanced.flags["companion_tick_enabled"] is True
    assert balanced.flags["companion_journal_enabled"] is True
    assert balanced.flags["companion_dreams_enabled"] is True
    assert balanced.flags["companion_drift_audit_enabled"] is True
    assert balanced.flags["companion_today_enabled"] is True
    # Balanced should NOT enable the autonomy moves.
    assert balanced.flags["companion_initiative_enabled"] is False
    assert balanced.flags["companion_consolidation_enabled"] is False
    assert balanced.flags["companion_skills_enabled"] is False
    assert balanced.flags["companion_creations_enabled"] is False


def test_full_enables_autonomy_features():
    """Full adds the opt-in autonomy features."""
    from augmentum.companion.intensity import PRESETS

    full = PRESETS["full"]
    assert full.flags["companion_initiative_enabled"] is True
    assert full.flags["companion_consolidation_enabled"] is True
    assert full.flags["companion_skills_enabled"] is True
    assert full.flags["companion_creations_enabled"] is True
    # But still respects cultural_intake (network egress) as
    # separately opt-in
    assert full.flags["companion_cultural_intake_enabled"] is False


# ── detect_intensity ─────────────────────────────────────────────────


def test_detect_intensity_off_when_runtime_master_off():
    from augmentum.companion.intensity import detect_intensity

    # Runtime master off — every other flag is moot
    snapshot = {"companion_runtime_enabled": False}
    assert detect_intensity(snapshot) == "off"

    # Even if individual flags are on, master off = off
    snapshot = {
        "companion_runtime_enabled": False,
        "companion_dispatch_enabled": True,
        "companion_journal_enabled": True,
    }
    assert detect_intensity(snapshot) == "off"


def test_detect_intensity_minimal_match():
    from augmentum.companion.intensity import PRESETS, detect_intensity

    minimal = PRESETS["minimal"]
    snapshot = {"companion_runtime_enabled": True, **minimal.flags}
    assert detect_intensity(snapshot) == "minimal"


def test_detect_intensity_balanced_match():
    from augmentum.companion.intensity import PRESETS, detect_intensity

    balanced = PRESETS["balanced"]
    snapshot = {"companion_runtime_enabled": True, **balanced.flags}
    assert detect_intensity(snapshot) == "balanced"


def test_detect_intensity_full_match():
    from augmentum.companion.intensity import PRESETS, detect_intensity

    full = PRESETS["full"]
    snapshot = {"companion_runtime_enabled": True, **full.flags}
    assert detect_intensity(snapshot) == "full"


def test_detect_intensity_custom_when_diverges():
    from augmentum.companion.intensity import PRESETS, detect_intensity

    # Start from balanced, then override one flag the other way
    balanced = PRESETS["balanced"]
    snapshot = {"companion_runtime_enabled": True, **balanced.flags}
    # Turn dreams off — no longer matches balanced
    snapshot["companion_dreams_enabled"] = False
    assert detect_intensity(snapshot) == "custom"


# ── apply_preset ─────────────────────────────────────────────────────


def test_apply_preset_writes_full_bundle():
    """apply_preset must write every flag in the preset bundle."""
    from augmentum.companion.intensity import PRESETS, apply_preset

    class _FakeStore:
        def __init__(self):
            self.writes: dict[str, object] = {}

        def set(self, key, value):
            self.writes[key] = value

    store = _FakeStore()
    apply_preset("minimal", store)
    minimal = PRESETS["minimal"]
    for flag, expected in minimal.flags.items():
        assert store.writes[flag] == expected, (
            f"apply_preset didn't write {flag} correctly"
        )
    # And the intensity setting itself is updated
    assert store.writes["companion_intensity"] == "minimal"


def test_apply_preset_unknown_raises():
    from augmentum.companion.intensity import apply_preset

    class _Store:
        def set(self, k, v): pass

    with pytest.raises(ValueError):
        apply_preset("super-mega-mode", _Store())


# ── /api/companion/intensity endpoint ────────────────────────────────


def _client_app():
    from fastapi import FastAPI
    from augmentum.proxy.companion_routes import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.mark.asyncio
async def test_intensity_endpoint_accepts_valid_level(monkeypatch):
    """POSTing a valid intensity applies the preset + returns 200."""
    from fastapi.testclient import TestClient

    app = _client_app()

    class _AsyncStore:
        async def set(self, k, v):
            pass

    app.state.settings_store = _AsyncStore()

    with TestClient(app) as client:
        resp = client.post("/api/companion/intensity", json={"level": "minimal"})
        assert resp.status_code == 200
        data = resp.json()
    assert data["ok"] is True
    assert data["intensity"] == "minimal"
    assert "applied_flags" in data


@pytest.mark.asyncio
async def test_intensity_endpoint_rejects_unknown(monkeypatch):
    from fastapi.testclient import TestClient

    app = _client_app()

    class _AsyncStore:
        async def set(self, k, v): pass

    app.state.settings_store = _AsyncStore()

    with TestClient(app) as client:
        resp = client.post("/api/companion/intensity", json={"level": "nope"})
        assert resp.status_code == 400
        data = resp.json()
    assert data["ok"] is False
    assert data["reason"] == "unknown_intensity"


@pytest.mark.asyncio
async def test_intensity_endpoint_503_without_settings_store():
    from fastapi.testclient import TestClient

    app = _client_app()
    # No settings_store on app.state — endpoint should degrade
    with TestClient(app) as client:
        resp = client.post("/api/companion/intensity", json={"level": "minimal"})
        assert resp.status_code == 503


# ── /api/companion/status surface ────────────────────────────────────


@pytest.mark.asyncio
async def test_status_surfaces_intensity_block(monkeypatch):
    from fastapi.testclient import TestClient
    from augmentum.config import settings as _settings

    # Off state — intensity block should still appear, current='off'
    monkeypatch.setattr(_settings, "companion_runtime_enabled", False)

    app = _client_app()
    with TestClient(app) as client:
        data = client.get("/api/companion/status").json()

    assert "intensity" in data
    intensity = data["intensity"]
    assert intensity["current"] == "off"
    assert "presets" in intensity
    assert isinstance(intensity["presets"], list)
    # All four user-pickable presets present
    names = {p["name"] for p in intensity["presets"]}
    assert names == {"off", "minimal", "balanced", "full"}


@pytest.mark.asyncio
async def test_status_intensity_matches_when_balanced(monkeypatch):
    """Apply balanced flags via monkeypatch → status reports 'balanced'."""
    from fastapi.testclient import TestClient
    from augmentum.companion.intensity import PRESETS
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_runtime_enabled", True)
    balanced = PRESETS["balanced"]
    for flag, value in balanced.flags.items():
        monkeypatch.setattr(_settings, flag, value, raising=False)

    app = _client_app()
    with TestClient(app) as client:
        data = client.get("/api/companion/status").json()

    assert data["intensity"]["current"] == "balanced"
    # Label is the user-facing name; "balanced" preset renders as "Present"
    # per the aesthetic redesign (Quiet / Present / Awake).
    assert data["intensity"]["label"] == "Present"
