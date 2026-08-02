"""Surface session substrate tests for augmentum.proxy.surface_routes."""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from augmentum.surfaces import (
    SurfaceAccessTokenStore,
    SurfaceConflictError,
    SurfaceRuntime,
    SurfaceStore,
)

_MIGRATIONS_DIR = Path(__file__).parent.parent / "augmentum" / "state" / "migrations"
_BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
INSERT OR IGNORE INTO users (id, username) VALUES ('usr_test', 'tester');
INSERT OR IGNORE INTO users (id, username) VALUES ('usr_other', 'other');
"""


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _apply_migration(conn: aiosqlite.Connection, version: int) -> None:
    for path in _MIGRATIONS_DIR.glob("*.sql"):
        try:
            v = int(path.stem.split("_")[0])
        except (ValueError, IndexError):
            continue
        if v == version:
            await conn.executescript(path.read_text(encoding="utf-8"))
            await conn.commit()
            return
    raise FileNotFoundError(f"Migration {version:03d} not found")


async def _make_conn() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_BOOTSTRAP_SQL)
    await _apply_migration(conn, 147)
    return conn


@pytest.fixture
def surface_client(app):
    conn = _run(_make_conn())
    app.state.surface_store = SurfaceStore(conn)
    app.state.surface_runtime = SurfaceRuntime()
    app.state.surface_access_token_store = SurfaceAccessTokenStore(default_ttl_s=600)
    client = TestClient(app)
    client.headers.update({"Authorization": "Bearer test-token"})
    yield client, app, conn
    _run(conn.close())


class TestSurfaceStore:
    @pytest.mark.asyncio
    async def test_create_join_patch_and_user_scope(self):
        conn = await _make_conn()
        try:
            store = SurfaceStore(conn)
            session = await store.create(
                user_id="usr_test",
                kind="comic.reader.webtoon",
                title="Chapter 1",
                content_ref={"kind": "comic", "file_id": "fi_1"},
                state={"reader": {"page": 1, "scroll_ratio": 0}},
            )

            assert session["revision"] == 0
            assert await store.get(session["id"], user_id="usr_other") is None

            joined = await store.join(
                session["id"],
                user_id="usr_test",
                participant={"participant_id": "phone", "role": "controller"},
            )
            assert joined is not None
            assert joined["revision"] == 1
            assert joined["participants"][0]["id"] == "phone"

            patched = await store.patch_state(
                session["id"],
                user_id="usr_test",
                patch={"reader": {"page": 4, "scroll_ratio": 0.25}},
                base_revision=1,
                source_participant_id="phone",
            )
            assert patched is not None
            assert patched["revision"] == 2
            assert patched["state"]["reader"]["page"] == 4
            assert patched["state"]["reader"]["scroll_ratio"] == 0.25

            with pytest.raises(SurfaceConflictError):
                await store.patch_state(
                    session["id"],
                    user_id="usr_test",
                    patch={"reader": {"page": 5}},
                    base_revision=0,
                )

            events = await store.events_after(session["id"], user_id="usr_test", after_revision=0)
            assert [event["type"] for event in events] == [
                "surface.participant.joined",
                "surface.state.patched",
            ]
        finally:
            await conn.close()


class TestSurfaceRoutes:
    def test_recipes_expose_comic_contract(self, surface_client):
        client, _, _ = surface_client
        response = client.get("/api/surfaces/recipes")
        assert response.status_code == 200
        kinds = {recipe["kind"] for recipe in response.json()["recipes"]}
        assert "comic.reader.webtoon" in kinds

    def test_phone_tv_surface_flow(self, surface_client):
        client, app, _ = surface_client
        response = client.post(
            "/api/surfaces/sessions",
            json={
                "kind": "comic.reader.webtoon",
                "title": "Webtoon Night",
                "content_ref": {"kind": "comic", "file_id": "fi_comic", "title": "Webtoon Night"},
                "state": {"reader": {"page": 1, "page_count": 12, "scroll_ratio": 0}},
            },
        )
        assert response.status_code == 200
        session = response.json()["session"]
        session_id = session["id"]

        token_response = client.post(f"/api/surfaces/sessions/{session_id}/access-token", json={})
        assert token_response.status_code == 200
        access = token_response.json()["access"]
        assert "surface-receiver.html" in access["receiver_url"]
        assert "comic:read" in access["scopes"]
        token = access["token"]

        handoff_response = client.post(
            f"/api/surfaces/sessions/{session_id}/handoff",
            json={"target_role": "display", "target_label": "onn TV", "bluetooth_mtu": 185},
        )
        assert handoff_response.status_code == 200
        handoff = handoff_response.json()["handoff"]
        assert handoff["version"] == "augmentum.surface.handoff@1"
        assert handoff["transport"] == "bluetooth_to_ip"
        assert handoff["ble_payload"]["mode"] == "bluetooth_to_ip"
        assert handoff["ble_payload"]["join"]["transport"] == "bluetooth_handoff_https"
        assert handoff["ble_payload"]["join_url"].startswith("http")
        assert handoff["bluetooth"]["write_format"] == "utf8-json-concat"
        assert handoff["bluetooth"]["chunking"] in {"single-write", "json-fragments"}

        public_client = TestClient(app)
        public_join = public_client.post(
            f"/api/surface-public/{token}/join",
            json={
                "participant_id": "onn-tv",
                "role": "display",
                "label": "onn TV",
                "capabilities": ["display.comic_read@1"],
            },
        )
        assert public_join.status_code == 200
        public_session = public_join.json()["session"]
        assert "user_id" not in public_session
        assert public_session["revision"] == 1
        assert public_session["participants"][0]["id"] == "onn-tv"

        patch = client.post(
            f"/api/surfaces/sessions/{session_id}/state",
            json={
                "base_revision": 1,
                "source_participant_id": "phone",
                "patch": {"reader": {"page": 3, "scroll_ratio": 0.5}},
            },
        )
        assert patch.status_code == 200
        assert patch.json()["session"]["revision"] == 2

        events = public_client.get(f"/api/surface-public/{token}/events?after_revision=1")
        assert events.status_code == 200
        data = events.json()
        assert data["events"][0]["type"] == "surface.state.patched"
        assert data["session"]["state"]["reader"]["page"] == 3
        assert data["session"]["state"]["reader"]["scroll_ratio"] == 0.5

        stale = client.post(
            f"/api/surfaces/sessions/{session_id}/state",
            json={"base_revision": 0, "patch": {"reader": {"page": 4}}},
        )
        assert stale.status_code == 409
        assert stale.json()["error"] == "revision_conflict"
