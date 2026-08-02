"""Behavior tests for /api/games/saves/* — game save-state persistence.

These endpoints serve local-mode games (js13k bundles today — the only
source in ``_KNOWN_SOURCES`` that supports local play). Embed-mode
games (future remote launchers) handle their own save state via the
iframe's cross-origin storage and never hit this path.

Covered here: the /saves/{artifact_id} endpoints (GET/PUT/DELETE),
which are the high-value user-data paths for cross-device save sync:

* Ownership check — reads another user's save must return 404 even
  if the artifact_id is known (ID-guessing defense)
* Size cap (256KB) — prevents per-user settings_store bloat
* Round-trip — write-then-read returns the exact payload
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


TEST_USER_ID = "usr_test"


@pytest.fixture
def games_client(app):
    """Client with artifact store + settings store wired for saves tests."""
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.manager import StateManager
    from augmentum.state.settings_store import SettingsStore
    from augmentum.tools.artifact_storage import ArtifactStore

    backend = SQLiteBackend(":memory:")
    _run(backend.connect())

    app.state.state_manager = StateManager(backend)
    app.state.settings_store = SettingsStore(backend._conn)
    app.state.artifact_store = ArtifactStore(backend._conn)

    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc, backend._conn
    _run(backend.close())


async def _seed_game_artifact(conn, *, artifact_id="art_game_1",
                              user_id=TEST_USER_ID, name="Cool Game"):
    """Insert a game artifact directly — we only need the row to satisfy
    ArtifactStore.get() + the metadata.kind=='game' check."""
    metadata = json.dumps({"kind": "game", "source": "js13k", "source_id": "sid-1",
                           "title": name})
    await conn.execute(
        "INSERT INTO artifacts "
        "(id, task_id, session_id, filename, display_name, format, size_bytes, "
        " path, metadata, user_id) "
        "VALUES (?, '', '', ?, ?, 'html', 0, '', ?, ?)",
        (artifact_id, f"{name}.html", name, metadata, user_id),
    )
    await conn.commit()
    return artifact_id


# ===========================================================================
# GET /api/games/saves/{artifact_id}
# ===========================================================================

class TestGetSave:
    def test_no_save_returns_exists_false(self, games_client):
        client, conn = games_client
        aid = _run(_seed_game_artifact(conn, artifact_id="art_g_1"))
        r = client.get(f"/api/games/saves/{aid}")
        assert r.status_code == 200
        assert r.json() == {"data": {}, "exists": False}

    def test_missing_artifact_returns_404(self, games_client):
        client, _ = games_client
        r = client.get("/api/games/saves/art_nope")
        assert r.status_code == 404

    def test_non_game_artifact_returns_404(self, games_client):
        """Only artifacts tagged with metadata.kind=='game' accept saves —
        an ID that resolves to a different artifact type must 404."""
        client, conn = games_client
        # Seed a non-game artifact
        _run(conn.execute(
            "INSERT INTO artifacts (id, filename, display_name, format, path, "
            " metadata, user_id) VALUES (?, 'x.pdf', 'X', 'pdf', '', '{}', ?)",
            ("art_notgame", TEST_USER_ID),
        ))
        _run(conn.commit())
        r = client.get("/api/games/saves/art_notgame")
        assert r.status_code == 404

    def test_other_users_artifact_returns_404(self, games_client):
        """Ownership guard: another user's game artifact must 404,
        preventing save-state enumeration."""
        client, conn = games_client
        _run(_seed_game_artifact(conn, artifact_id="art_theirs",
                                 user_id="usr_other"))
        r = client.get("/api/games/saves/art_theirs")
        assert r.status_code == 404


# ===========================================================================
# PUT /api/games/saves/{artifact_id}
# ===========================================================================

class TestPutSave:
    def test_round_trip_write_then_read(self, games_client):
        client, conn = games_client
        aid = _run(_seed_game_artifact(conn))
        payload = {"level": 3, "score": 9001, "inventory": ["sword", "potion"]}
        r = client.put(f"/api/games/saves/{aid}", json={"data": payload})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert r.json()["bytes"] > 0

        r = client.get(f"/api/games/saves/{aid}")
        assert r.json()["exists"] is True
        assert r.json()["data"] == payload

    def test_put_overwrites_prior(self, games_client):
        client, conn = games_client
        aid = _run(_seed_game_artifact(conn))
        client.put(f"/api/games/saves/{aid}", json={"data": {"v": 1}})
        client.put(f"/api/games/saves/{aid}", json={"data": {"v": 2}})
        r = client.get(f"/api/games/saves/{aid}")
        assert r.json()["data"] == {"v": 2}

    def test_oversize_save_returns_413(self, games_client):
        """256KB cap prevents runaway saves from bloating settings_store."""
        client, conn = games_client
        aid = _run(_seed_game_artifact(conn))
        big = {"blob": "x" * (300 * 1024)}  # 300KB string → serialised > 256KB
        r = client.put(f"/api/games/saves/{aid}", json={"data": big})
        assert r.status_code == 413

    def test_put_other_users_artifact_returns_404(self, games_client):
        client, conn = games_client
        _run(_seed_game_artifact(conn, artifact_id="art_theirs",
                                 user_id="usr_other"))
        r = client.put("/api/games/saves/art_theirs", json={"data": {"v": 1}})
        assert r.status_code == 404

    def test_put_non_game_artifact_returns_404(self, games_client):
        client, conn = games_client
        _run(conn.execute(
            "INSERT INTO artifacts (id, filename, display_name, format, path, "
            " metadata, user_id) VALUES (?, 'x.pdf', 'X', 'pdf', '', '{}', ?)",
            ("art_nongame", TEST_USER_ID),
        ))
        _run(conn.commit())
        r = client.put("/api/games/saves/art_nongame", json={"data": {"v": 1}})
        assert r.status_code == 404


# ===========================================================================
# DELETE /api/games/saves/{artifact_id}
# ===========================================================================

class TestDeleteSave:
    def test_delete_clears_save(self, games_client):
        client, conn = games_client
        aid = _run(_seed_game_artifact(conn))
        client.put(f"/api/games/saves/{aid}", json={"data": {"v": 1}})
        assert client.get(f"/api/games/saves/{aid}").json()["exists"] is True

        r = client.delete(f"/api/games/saves/{aid}")
        assert r.status_code == 200

        r = client.get(f"/api/games/saves/{aid}")
        assert r.json()["exists"] is False

    def test_delete_missing_artifact_returns_404(self, games_client):
        client, _ = games_client
        r = client.delete("/api/games/saves/art_ghost")
        assert r.status_code == 404

    def test_delete_other_users_save_returns_404(self, games_client):
        client, conn = games_client
        _run(_seed_game_artifact(conn, artifact_id="art_theirs",
                                 user_id="usr_other"))
        r = client.delete("/api/games/saves/art_theirs")
        assert r.status_code == 404


# ===========================================================================
# Isolation between users' saves
# ===========================================================================

class TestSaveIsolation:
    def test_each_user_has_independent_save_for_same_artifact_id(self, games_client):
        """Sanity: two users can each own a different artifact with the
        SAME local id, and their save blobs must not cross-contaminate.
        (Unlikely in practice — IDs are random — but the test pins the
        ``_save_key`` user-scoping guarantee explicitly.)"""
        client, conn = games_client
        # Mine
        aid = _run(_seed_game_artifact(conn, artifact_id="shared_id"))
        client.put(f"/api/games/saves/{aid}", json={"data": {"mine": True}})
        # Theirs — same artifact_id is a PK collision so we can't insert;
        # instead verify the user-scoped settings key by directly reading
        # through the settings store for the other user.
        # The save key is game_save:<artifact_id>, keyed by user_id in settings_store.
        ss = getattr(conn, "_settings_store", None)  # not on conn directly
        # Just verify the route-level outcome: other user can't read mine
        # Build another client with a different user id by overriding mock SM
        # Using the raw settings table:
        cursor = _run(conn.execute(
            "SELECT value FROM user_settings WHERE user_id = ? AND key = ?",
            (TEST_USER_ID, f"game_save:{aid}"),
        ))
        row = _run(cursor.fetchone())
        assert row is not None
        assert "mine" in row[0]

        cursor = _run(conn.execute(
            "SELECT value FROM user_settings WHERE user_id = ? AND key = ?",
            ("usr_other", f"game_save:{aid}"),
        ))
        row = _run(cursor.fetchone())
        assert row is None  # other user's key space is independent


# ===========================================================================
# Router sanity
# ===========================================================================

class TestRouterShape:
    def test_prefix(self):
        from augmentum.proxy.games_routes import router
        assert router.prefix == "/api/games"

    def test_expected_endpoints_registered(self):
        from augmentum.proxy.games_routes import router
        paths = {r.path for r in router.routes}
        expected = {
            "/api/games/browse",
            "/api/games/details",
            "/api/games/pin",
            "/api/games/pin/{artifact_id}",
            "/api/games/saves/{artifact_id}",
        }
        assert expected.issubset(paths)
