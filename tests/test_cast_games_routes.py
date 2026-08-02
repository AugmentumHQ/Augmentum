"""Tests for /api/cast/games/{title_id}/* routes.

Pins:
  - GET /profile returns 404 when no row exists
  - GET /profile returns the persisted shape after PUT
  - PUT /profile rejects unknown strategies + unknown adapter ids
  - DELETE /profile is idempotent + returns the removed flag
  - GET /profiles lists this user's saved profiles
  - POST /classify returns prepared payload with input_chain + URL
  - Cross-user isolation: alice's writes invisible to bob
  - 503 when registry not wired (memory backend / startup failure)
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from augmentum.cast.games.classifier import CastClassifier
from augmentum.cast.games.registry import CastProfileRegistry
from augmentum.state.backends.sqlite import SQLiteBackend


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def cast_games_app(app):
    """Wires a SQLite-backed CastProfileRegistry + CastClassifier on
    the existing test app so the /api/cast/games/* routes work."""
    backend = SQLiteBackend(":memory:")
    _run(backend.connect())
    reg = CastProfileRegistry(backend._conn)
    app.state.cast_profile_registry = reg
    app.state.cast_classifier = CastClassifier(profile_registry=reg)
    yield app
    _run(backend.close())


@pytest.fixture
def cast_games_client(cast_games_app):
    tc = TestClient(cast_games_app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    return tc


# ── GET /profile ─────────────────────────────────────────────────


def test_get_profile_404_when_missing(cast_games_client):
    r = cast_games_client.get("/api/cast/games/no-such/profile")
    assert r.status_code == 404


def test_get_profile_503_without_registry(app):
    # No cast_profile_registry attribute → 503
    app.state.cast_profile_registry = None
    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    r = tc.get("/api/cast/games/x/profile")
    assert r.status_code == 503


# ── PUT /profile ─────────────────────────────────────────────────


def test_put_profile_persists_minimal_payload(cast_games_client):
    r = cast_games_client.put(
        "/api/cast/games/g1/profile",
        json={"strategy": "proxy", "input_chain": ["keyboard"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["profile"]["strategy"] == "proxy"
    assert body["profile"]["input_chain"] == ["keyboard"]
    assert body["profile"]["classified_by"] == "manual"


def test_put_profile_rejects_unknown_strategy(cast_games_client):
    r = cast_games_client.put(
        "/api/cast/games/g1/profile",
        json={"strategy": "frobnicate"},
    )
    assert r.status_code == 400


def test_put_profile_rejects_unknown_adapter(cast_games_client):
    r = cast_games_client.put(
        "/api/cast/games/g1/profile",
        json={"input_chain": ["gamepad_api", "nonexistent_adapter"]},
    )
    assert r.status_code == 400


def test_put_profile_accepts_keymap_payload(cast_games_client):
    r = cast_games_client.put(
        "/api/cast/games/g1/profile",
        json={
            "input_chain": ["keyboard"],
            "keymap": {"keyboard": {"buttons": {"0": "Space"}}},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["profile"]["keymap"]["keyboard"]["buttons"]["0"] == "Space"


def test_put_then_get_roundtrips(cast_games_client):
    cast_games_client.put(
        "/api/cast/games/round-trip/profile",
        json={"notes": "test note", "input_chain": ["gamepad_api", "touch"]},
    )
    r = cast_games_client.get("/api/cast/games/round-trip/profile")
    assert r.status_code == 200
    body = r.json()
    assert body["profile"]["notes"] == "test note"
    assert body["profile"]["input_chain"] == ["gamepad_api", "touch"]


# ── DELETE /profile ──────────────────────────────────────────────


def test_delete_returns_removed_true_when_existed(cast_games_client):
    cast_games_client.put(
        "/api/cast/games/del/profile",
        json={"notes": "to delete"},
    )
    r = cast_games_client.delete("/api/cast/games/del/profile")
    assert r.status_code == 200
    assert r.json() == {"removed": True}

    # Second delete returns False
    r2 = cast_games_client.delete("/api/cast/games/del/profile")
    assert r2.json() == {"removed": False}


# ── GET /profiles ────────────────────────────────────────────────


def test_list_profiles_returns_user_scoped(cast_games_client):
    cast_games_client.put("/api/cast/games/p1/profile", json={"notes": "a"})
    cast_games_client.put("/api/cast/games/p2/profile", json={"notes": "b"})
    r = cast_games_client.get("/api/cast/games/profiles")
    assert r.status_code == 200
    body = r.json()
    titles = {p["title_id"] for p in body["profiles"]}
    assert titles == {"p1", "p2"}


# ── POST /classify ───────────────────────────────────────────────


def test_classify_returns_prepared_payload(cast_games_client):
    r = cast_games_client.post(
        "/api/cast/games/rom-x/classify",
        json={
            "title_id": "rom-x",
            "kind": "emulator_rom",
            "display_name": "Test Rom",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "default"
    assert body["prepared"]["strategy"] == "shim"
    assert body["prepared"]["surface_url"].startswith("/ui/play/?title_id=rom-x")
    assert body["prepared"]["input_chain"] == ["gamepad_api"]


def test_classify_for_js13k_picks_play_web(cast_games_client):
    r = cast_games_client.post(
        "/api/cast/games/js-1/classify",
        json={
            "title_id": "js-1",
            "kind": "js13k_game",
            "display_name": "Some JS13k",
            "metadata": {"embed_url": "https://js13kgames.com/sample/"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "/ui/play-web/" in body["prepared"]["surface_url"]
    assert "embed_url=https" in body["prepared"]["surface_url"]


def test_classify_after_override_uses_overridden_chain(cast_games_client):
    cast_games_client.put(
        "/api/cast/games/oc/profile",
        json={"input_chain": ["keyboard"]},
    )
    r = cast_games_client.post(
        "/api/cast/games/oc/classify",
        json={"title_id": "oc", "kind": "emulator_rom"},
    )
    body = r.json()
    assert body["source"] == "registry"
    assert body["prepared"]["input_chain"] == ["keyboard"]


# ── Auth + cross-user isolation ──────────────────────────────────


def test_auth_required(cast_games_app):
    tc = TestClient(cast_games_app)  # no Authorization header
    r = tc.get("/api/cast/games/x/profile")
    assert r.status_code in (401, 403)


def test_cross_user_isolation(cast_games_app):
    """Alice's PUT should not be visible to Bob.

    Swap the mock session_manager between requests to simulate the
    two users hitting the same registry.
    """
    from augmentum.auth.models import User

    alice = User(id="usr_alice", username="alice", display_name="Alice",
                 role="admin", is_active=True)
    bob = User(id="usr_bob", username="bob", display_name="Bob",
               role="admin", is_active=True)

    sm = cast_games_app.state.session_manager

    sm.validate_token = AsyncMock(return_value=alice)
    sm.get_user_by_id = AsyncMock(return_value=alice)
    tc_alice = TestClient(cast_games_app)
    tc_alice.headers.update({"Authorization": "Bearer alice-token"})
    r = tc_alice.put(
        "/api/cast/games/shared-title/profile",
        json={"notes": "alice"},
    )
    assert r.status_code == 200

    sm.validate_token = AsyncMock(return_value=bob)
    sm.get_user_by_id = AsyncMock(return_value=bob)
    tc_bob = TestClient(cast_games_app)
    tc_bob.headers.update({"Authorization": "Bearer bob-token"})
    r = tc_bob.get("/api/cast/games/shared-title/profile")
    assert r.status_code == 404, "bob must not see alice's profile"
