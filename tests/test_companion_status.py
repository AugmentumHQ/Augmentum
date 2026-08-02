"""Tests for GET /api/companion/status — Tier 1 visibility surface.

The endpoint translates flag state into plain-language feature
descriptions for the settings panel. Covers:

- Shape: {enabled, persona_mode, presence_mode, features, advanced}
- Plain-language descriptions (no flag-name jargon in user-visible text)
- Runtime off → enabled:false, features list empty
- Runtime on → features populated with active states reflecting flags
- Advanced section lists possible-but-off features so user knows what's available
- The endpoint never returns 503 — UI always gets a renderable response
"""

from __future__ import annotations

import pytest


def _client_app():
    """Build a FastAPI app with just the companion router mounted.
    Avoids the full lifespan; tests can hit the endpoint directly."""
    from fastapi import FastAPI

    from augmentum.proxy.companion_routes import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.mark.asyncio
async def test_status_shape_when_runtime_off(monkeypatch):
    from fastapi.testclient import TestClient

    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_runtime_enabled", False)

    app = _client_app()
    with TestClient(app) as client:
        resp = client.get("/api/companion/status")
        assert resp.status_code == 200
        data = resp.json()

    # Shape: all expected keys present, types correct
    assert "enabled" in data
    assert "persona_mode" in data
    assert "presence_mode" in data
    assert "features" in data
    assert "advanced" in data
    assert data["enabled"] is False
    assert isinstance(data["features"], list)
    assert isinstance(data["advanced"], list)
    # Runtime off → features list empty (nothing's running)
    assert data["features"] == []
    assert data["advanced"] == []


@pytest.mark.asyncio
async def test_status_lists_active_features_when_runtime_on(monkeypatch):
    from fastapi.testclient import TestClient

    from augmentum.config import settings as _settings

    # Runtime on; downstream defaults are True per Tier 1 flip
    monkeypatch.setattr(_settings, "companion_runtime_enabled", True)
    monkeypatch.setattr(_settings, "companion_persona_mode", True)
    monkeypatch.setattr(_settings, "companion_salience_enabled", True)
    monkeypatch.setattr(_settings, "companion_voice_journal_enabled", True)
    monkeypatch.setattr(_settings, "companion_becca_direct_enabled", True)
    monkeypatch.setattr(_settings, "companion_dispatch_enabled", True)
    monkeypatch.setattr(_settings, "companion_dispatch_routes_chat", True)

    app = _client_app()
    with TestClient(app) as client:
        resp = client.get("/api/companion/status")
        assert resp.status_code == 200
        data = resp.json()

    assert data["enabled"] is True
    assert data["persona_mode"] is True
    feature_keys = {f["key"] for f in data["features"]}
    # All five Tier 1 features represented
    assert "becca_direct" in feature_keys
    assert "salience" in feature_keys
    assert "voice_journal" in feature_keys
    assert "user_affect" in feature_keys
    assert "dispatch_routing" in feature_keys
    # All active
    for f in data["features"]:
        assert f["active"] is True, f"feature {f['key']} should be active"


@pytest.mark.asyncio
async def test_status_features_have_plain_language(monkeypatch):
    """User-visible text must not contain flag names. This is the
    tasteful-surface discipline made testable."""
    from fastapi.testclient import TestClient

    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_runtime_enabled", True)

    app = _client_app()
    with TestClient(app) as client:
        data = client.get("/api/companion/status").json()

    flag_jargon = {
        "companion_salience_enabled",
        "companion_becca_direct_enabled",
        "companion_dispatch_enabled",
        "companion_dispatch_routes_chat",
        "companion_voice_journal_enabled",
        "synapse layer",
    }
    for entry in data["features"] + data["advanced"]:
        for field in ("title", "summary", "note"):
            text = (entry.get(field) or "").lower()
            for jargon in flag_jargon:
                assert jargon not in text, (
                    f"flag jargon {jargon!r} leaked into user-visible "
                    f"{field} of {entry['key']!r}: {text!r}"
                )


@pytest.mark.asyncio
async def test_status_advanced_lists_known_off_features(monkeypatch):
    """The advanced list surfaces things-she-could-do-but-doesn't so
    the user has a sense of the full system, not just the active part."""
    from fastapi.testclient import TestClient

    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_runtime_enabled", True)
    monkeypatch.setattr(_settings, "companion_consolidation_enabled", False)
    monkeypatch.setattr(_settings, "companion_skills_enabled", False)
    monkeypatch.setattr(_settings, "companion_initiative_enabled", False)
    monkeypatch.setattr(_settings, "companion_drives_enabled", False)

    app = _client_app()
    with TestClient(app) as client:
        data = client.get("/api/companion/status").json()

    advanced_keys = {a["key"] for a in data["advanced"]}
    assert "consolidation" in advanced_keys
    assert "skills" in advanced_keys
    assert "initiative" in advanced_keys
    assert "drives" in advanced_keys
    # All off
    for a in data["advanced"]:
        assert a["active"] is False


@pytest.mark.asyncio
async def test_status_reflects_user_choice_persona_off(monkeypatch):
    """Runtime on but persona off: features list still populates (the
    runtime IS running) but persona_mode reports false so the UI can
    show 'turn on persona to see her'."""
    from fastapi.testclient import TestClient

    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_runtime_enabled", True)
    monkeypatch.setattr(_settings, "companion_persona_mode", False)

    app = _client_app()
    with TestClient(app) as client:
        data = client.get("/api/companion/status").json()

    assert data["enabled"] is True
    assert data["persona_mode"] is False
    # Features still list — runtime IS running things internally
    assert len(data["features"]) > 0


@pytest.mark.asyncio
async def test_status_never_returns_5xx(monkeypatch):
    """The status endpoint should never break the settings UI.
    Returning a degraded response is fine; returning 500/503 is not."""
    from fastapi.testclient import TestClient

    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_runtime_enabled", False)

    app = _client_app()
    with TestClient(app) as client:
        resp = client.get("/api/companion/status")
        assert resp.status_code == 200
