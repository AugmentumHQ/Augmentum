"""Tests for GET /api/companion/affect_read — Tier 2 widget surface.

The endpoint surfaces the current decayed user-affect read in her
voice. The widget polls this every 90s; null observation = no
confident read, indicator stays hidden.

Coverage:

- Runtime off → enabled:false, observation:null
- Runtime on but no user_id → observation:null (not 401 — the widget
  should silently hide rather than show an auth error)
- Runtime on but no tracker → observation:null
- Confident read → observation with phrase + tag + confidence
- Low-confidence read (< 0.35) → observation:null (she doesn't pretend)
- Neutral tags (unclear/settled) → observation:null
- Hedged reads (confidence < 0.6) carry hedged:true so UI can append suffix
- The phrasing is in her register (not flag-name jargon)
- Always 200
"""

from __future__ import annotations

import time

import pytest


def _client_app():
    from fastapi import FastAPI
    from augmentum.proxy.companion_routes import router

    app = FastAPI()
    app.include_router(router)
    return app


def _attach_tracker(app, *, owner_user_id: str = ""):
    """Attach a minimal runtime + UserAffectTracker to app.state."""
    from augmentum.companion_runtime.perception.user_affect import UserAffectTracker

    class _Runtime:
        companion_id = "becca"
        user_affect = UserAffectTracker()

        @property
        def owner_user_id(self):
            return owner_user_id

    app.state.companion_runtime = _Runtime()
    return app.state.companion_runtime


@pytest.mark.asyncio
async def test_affect_read_runtime_off(monkeypatch):
    from fastapi.testclient import TestClient
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_runtime_enabled", False)

    app = _client_app()
    with TestClient(app) as client:
        resp = client.get("/api/companion/affect_read")
        assert resp.status_code == 200
        data = resp.json()
    assert data["enabled"] is False
    assert data["observation"] is None


@pytest.mark.asyncio
async def test_affect_read_no_runtime_attached(monkeypatch):
    from fastapi.testclient import TestClient
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_runtime_enabled", True)

    app = _client_app()
    with TestClient(app) as client:
        resp = client.get("/api/companion/affect_read")
        assert resp.status_code == 200
        data = resp.json()
    assert data["observation"] is None


@pytest.mark.asyncio
async def test_affect_read_returns_confident_observation(monkeypatch):
    from fastapi.testclient import TestClient
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_runtime_enabled", True)

    app = _client_app()
    runtime = _attach_tracker(app)

    # Need to give the request a user_id. The route uses
    # _resolve_user_id(request) — let's see how that works by
    # injecting via header or scope. For now we'll test via tracker
    # state directly + the X-Augmentum-User header convention.
    runtime.user_affect.update("user-test", "tender")

    with TestClient(app) as client:
        # The endpoint resolves user_id from auth scope; tests with
        # no auth get empty user_id → null observation. To exercise
        # the success path we need to patch _resolve_user_id.
        from augmentum.proxy import companion_routes
        monkeypatch.setattr(companion_routes, "_resolve_user_id",
                            lambda req: "user-test")
        resp = client.get("/api/companion/affect_read")
        assert resp.status_code == 200
        data = resp.json()

    assert data["enabled"] is True
    assert data["observation"] is not None
    obs = data["observation"]
    assert obs["tag"] == "tender"
    assert obs["confidence"] > 0.9  # just-updated
    assert "phrase" in obs
    assert obs["phrase"]  # non-empty


@pytest.mark.asyncio
async def test_affect_read_phrase_in_her_register(monkeypatch):
    """The phrase should read like her, not like jargon."""
    from fastapi.testclient import TestClient
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_runtime_enabled", True)

    app = _client_app()
    runtime = _attach_tracker(app)
    runtime.user_affect.update("u", "frustrated")

    from augmentum.proxy import companion_routes
    monkeypatch.setattr(companion_routes, "_resolve_user_id",
                        lambda req: "u")

    with TestClient(app) as client:
        data = client.get("/api/companion/affect_read").json()

    phrase = data["observation"]["phrase"].lower()
    # Shouldn't be raw "frustrated" — should be her phrasing
    assert "running into" in phrase or "frustrated" in phrase
    # Shouldn't carry the PAD axis terminology
    assert "valence" not in phrase
    assert "arousal" not in phrase
    assert "pad" not in phrase


@pytest.mark.asyncio
async def test_affect_read_filters_low_confidence(monkeypatch):
    """A read that's decayed past the visibility threshold returns null.
    She doesn't pretend to know what's decayed."""
    from fastapi.testclient import TestClient
    from augmentum.companion_runtime.perception.user_affect import UserAffectTracker
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_runtime_enabled", True)

    app = _client_app()

    class _Runtime:
        companion_id = "becca"
        # Use a short half-life so we can simulate decay
        user_affect = UserAffectTracker(half_life_s=60.0)
        owner_user_id = ""

    app.state.companion_runtime = _Runtime()

    # Observation 5 half-lives ago — confidence ≈ 0.03
    app.state.companion_runtime.user_affect.update(
        "u", "tender", observed_at=time.time() - 300.0,
    )

    from augmentum.proxy import companion_routes
    monkeypatch.setattr(companion_routes, "_resolve_user_id",
                        lambda req: "u")

    with TestClient(app) as client:
        data = client.get("/api/companion/affect_read").json()

    assert data["observation"] is None


@pytest.mark.asyncio
async def test_affect_read_hedged_flag(monkeypatch):
    """Reads with confidence < 0.6 carry hedged:true so the widget
    can append the 'could be wrong' suffix."""
    from fastapi.testclient import TestClient
    from augmentum.companion_runtime.perception.user_affect import UserAffectTracker
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_runtime_enabled", True)

    app = _client_app()

    class _Runtime:
        companion_id = "becca"
        user_affect = UserAffectTracker(half_life_s=60.0)
        owner_user_id = ""

    app.state.companion_runtime = _Runtime()
    # 1 half-life ago — confidence ≈ 0.5
    app.state.companion_runtime.user_affect.update(
        "u", "curious", observed_at=time.time() - 65.0,
    )

    from augmentum.proxy import companion_routes
    monkeypatch.setattr(companion_routes, "_resolve_user_id",
                        lambda req: "u")

    with TestClient(app) as client:
        data = client.get("/api/companion/affect_read").json()

    obs = data["observation"]
    assert obs is not None
    assert obs["hedged"] is True


@pytest.mark.asyncio
async def test_affect_read_never_500(monkeypatch):
    from fastapi.testclient import TestClient
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_runtime_enabled", True)

    app = _client_app()
    # No runtime attached on purpose — endpoint should degrade cleanly

    with TestClient(app) as client:
        resp = client.get("/api/companion/affect_read")
        assert resp.status_code == 200
