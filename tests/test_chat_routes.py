"""Tests for chat session CRUD API routes (/api/chats/)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sqlite_client(app):
    """TestClient with a real SQLite backend so chat routes can function."""
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.manager import StateManager

    backend = SQLiteBackend(":memory:")
    asyncio.get_event_loop().run_until_complete(backend.connect())
    app.state.state_manager = StateManager(backend)

    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc

    asyncio.get_event_loop().run_until_complete(backend.close())


# ── List (empty) ──────────────────────────────────────────────────────────

def test_list_chats_empty(sqlite_client):
    """GET /api/chats/ returns empty sessions map initially."""
    resp = sqlite_client.get("/api/chats/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["sessions"] == {}


def test_list_chats_meta_empty(sqlite_client):
    """GET /api/chats/?meta=1 returns empty sessions map initially."""
    resp = sqlite_client.get("/api/chats/?meta=1")
    assert resp.status_code == 200
    assert resp.json()["sessions"] == {}


# ── PUT (upsert) ─────────────────────────────────────────────────────────

def test_put_creates_session(sqlite_client):
    """PUT /api/chats/{id} creates a session (upsert)."""
    body = {"title": "Test Chat", "mode": "passthrough", "tree": {"msg1": {}}}
    resp = sqlite_client.put("/api/chats/sess-1", json=body)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert resp.json()["id"] == "sess-1"


def test_put_upserts_existing(sqlite_client):
    """PUT /api/chats/{id} updates an existing session."""
    body1 = {"title": "First Title", "mode": "passthrough"}
    sqlite_client.put("/api/chats/sess-1", json=body1)

    body2 = {"title": "Updated Title", "mode": "analytical"}
    resp = sqlite_client.put("/api/chats/sess-1", json=body2)
    assert resp.status_code == 200

    # Verify the update took effect
    get_resp = sqlite_client.get("/api/chats/sess-1")
    assert get_resp.status_code == 200
    assert get_resp.json()["title"] == "Updated Title"
    assert get_resp.json()["mode"] == "analytical"


# ── GET single ────────────────────────────────────────────────────────────

def test_get_chat_returns_session(sqlite_client):
    """GET /api/chats/{id} returns the full session."""
    body = {"title": "My Chat", "mode": "narrative", "tree": {"n1": {"text": "hi"}}}
    sqlite_client.put("/api/chats/sess-2", json=body)

    resp = sqlite_client.get("/api/chats/sess-2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "sess-2"
    assert data["title"] == "My Chat"
    assert data["mode"] == "narrative"
    assert "tree" in data


def test_get_chat_not_found(sqlite_client):
    """GET /api/chats/{id} returns 404 for missing session."""
    resp = sqlite_client.get("/api/chats/nonexistent")
    assert resp.status_code == 404
    assert "error" in resp.json()


# ── DELETE ────────────────────────────────────────────────────────────────

def test_delete_chat(sqlite_client):
    """DELETE /api/chats/{id} removes the session."""
    sqlite_client.put("/api/chats/sess-3", json={"title": "To Delete"})

    resp = sqlite_client.delete("/api/chats/sess-3")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # Confirm gone
    assert sqlite_client.get("/api/chats/sess-3").status_code == 404


def test_delete_chat_not_found(sqlite_client):
    """DELETE /api/chats/{id} returns 404 for missing session."""
    resp = sqlite_client.delete("/api/chats/nonexistent")
    assert resp.status_code == 404


# ── POST /sync (bulk) ────────────────────────────────────────────────────

def test_sync_creates_sessions(sqlite_client):
    """POST /api/chats/sync upserts multiple sessions at once."""
    body = {
        "sessions": {
            "s1": {"title": "Chat One", "mode": "passthrough"},
            "s2": {"title": "Chat Two", "mode": "analytical"},
        }
    }
    resp = sqlite_client.post("/api/chats/sync", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["imported"] == 2

    # Both sessions should be retrievable
    assert sqlite_client.get("/api/chats/s1").status_code == 200
    assert sqlite_client.get("/api/chats/s2").status_code == 200


def test_sync_with_deletions(sqlite_client):
    """POST /api/chats/sync can delete sessions in the same call."""
    # Create a session first
    sqlite_client.put("/api/chats/to-delete", json={"title": "Doomed"})

    body = {
        "sessions": {"s3": {"title": "New Chat"}},
        "deleted": ["to-delete"],
    }
    resp = sqlite_client.post("/api/chats/sync", json=body)
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 1

    assert sqlite_client.get("/api/chats/to-delete").status_code == 404
    assert sqlite_client.get("/api/chats/s3").status_code == 200


# ── List (populated) ─────────────────────────────────────────────────────

def test_list_chats_returns_sessions(sqlite_client):
    """GET /api/chats/ returns all sessions after creation."""
    sqlite_client.put("/api/chats/a1", json={"title": "A"})
    sqlite_client.put("/api/chats/a2", json={"title": "B"})

    resp = sqlite_client.get("/api/chats/")
    assert resp.status_code == 200
    sessions = resp.json()["sessions"]
    assert "a1" in sessions
    assert "a2" in sessions


def test_list_chats_meta_returns_stubs(sqlite_client):
    """GET /api/chats/?meta=1 returns metadata stubs without full tree data."""
    sqlite_client.put("/api/chats/m1", json={
        "title": "Meta Test",
        "mode": "passthrough",
        "tree": {"n1": {}, "n2": {}},
    })

    resp = sqlite_client.get("/api/chats/?meta=1")
    assert resp.status_code == 200
    sessions = resp.json()["sessions"]
    assert "m1" in sessions
    stub = sessions["m1"]
    assert stub["title"] == "Meta Test"
    assert stub["mode"] == "passthrough"
    assert "messageCount" in stub
    # Meta stubs should NOT contain the full tree
    assert "tree" not in stub


# ── MemoryBackend fallback ────────────────────────────────────────────────

def test_list_chats_no_sqlite_returns_empty(client):
    """With MemoryBackend (no SQLite), list returns empty sessions."""
    resp = client.get("/api/chats/")
    assert resp.status_code == 200
    assert resp.json()["sessions"] == {}


def test_get_chat_no_sqlite_returns_503(client):
    """With MemoryBackend, GET single chat returns 503."""
    resp = client.get("/api/chats/any-id")
    assert resp.status_code == 503


# ── Stale-write guard (multi-tab clobber protection) ─────────────────────
#
# The sync/save paths upsert the WHOLE session blob. Without a guard, a
# tab holding a stale copy of a session silently erased every turn another
# device had added since (last-writer-wins on the full tree). The guard
# compares the blob's client edit-stamp (`updatedAt`, ms) — NOT the
# updated_at column, which records sync time and would bless the clobber.

def _session_blob(updated_at, *node_texts):
    tree = {}
    prev = None
    for i, text in enumerate(node_texts):
        nid = f"n_{i}"
        tree[nid] = {
            "id": nid,
            "role": "user" if i % 2 == 0 else "assistant",
            "content": text,
            "parentId": prev,
            "children": [],
        }
        if prev:
            tree[prev]["children"].append(nid)
        prev = nid
    blob = {"version": 2, "title": f"t{updated_at}", "mode": "passthrough",
            "tree": tree, "rootId": "n_0", "activeLeafId": prev}
    if updated_at is not None:
        blob["updatedAt"] = updated_at
    return blob


def test_sync_stale_session_rejected(sqlite_client):
    """A sync carrying an older edit-stamp must not replace a newer tree."""
    fresh = _session_blob(2000, "hello", "hi there", "how are you")
    resp = sqlite_client.post("/api/chats/sync", json={"sessions": {"s1": fresh}})
    assert resp.status_code == 200
    assert resp.json()["imported"] == 1

    stale = _session_blob(1000, "hello")  # older stamp, fewer turns
    resp = sqlite_client.post("/api/chats/sync", json={"sessions": {"s1": stale}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["imported"] == 0
    assert body["stale"] == ["s1"]

    stored = sqlite_client.get("/api/chats/s1").json()
    assert stored["updatedAt"] == 2000
    assert len(stored["tree"]) == 3, "newer tree was clobbered by a stale sync"


def test_sync_newer_session_accepted(sqlite_client):
    resp = sqlite_client.post(
        "/api/chats/sync", json={"sessions": {"s1": _session_blob(1000, "a")}})
    assert resp.json()["imported"] == 1
    resp = sqlite_client.post(
        "/api/chats/sync", json={"sessions": {"s1": _session_blob(3000, "a", "b")}})
    body = resp.json()
    assert body["imported"] == 1
    assert body["stale"] == []
    assert sqlite_client.get("/api/chats/s1").json()["updatedAt"] == 3000


def test_sync_equal_stamp_accepted(sqlite_client):
    """Idempotent re-send (retry after network blip) must not be rejected."""
    blob = _session_blob(1500, "a", "b")
    sqlite_client.post("/api/chats/sync", json={"sessions": {"s1": blob}})
    resp = sqlite_client.post("/api/chats/sync", json={"sessions": {"s1": blob}})
    assert resp.json()["imported"] == 1
    assert resp.json()["stale"] == []


def test_sync_legacy_blob_without_stamp_accepted(sqlite_client):
    """Blobs with no updatedAt (legacy clients) keep the old behavior."""
    sqlite_client.post(
        "/api/chats/sync", json={"sessions": {"s1": _session_blob(2000, "a")}})
    resp = sqlite_client.post(
        "/api/chats/sync", json={"sessions": {"s1": _session_blob(None, "x", "y")}})
    assert resp.json()["imported"] == 1
    assert resp.json()["stale"] == []


def test_sync_stale_does_not_block_batch(sqlite_client):
    """One stale session must not stop fresh sessions in the same call."""
    sqlite_client.post(
        "/api/chats/sync", json={"sessions": {"s1": _session_blob(2000, "a", "b")}})
    resp = sqlite_client.post("/api/chats/sync", json={"sessions": {
        "s1": _session_blob(1000, "a"),          # stale — rejected
        "s2": _session_blob(500, "brand new"),   # new id — accepted
    }})
    body = resp.json()
    assert body["imported"] == 1
    assert body["stale"] == ["s1"]
    assert sqlite_client.get("/api/chats/s2").status_code == 200
    assert len(sqlite_client.get("/api/chats/s1").json()["tree"]) == 2


def test_put_stale_returns_409(sqlite_client):
    sqlite_client.put("/api/chats/s1", json=_session_blob(2000, "a", "b"))
    resp = sqlite_client.put("/api/chats/s1", json=_session_blob(1000, "a"))
    assert resp.status_code == 409
    assert resp.json()["stale"] is True
    assert len(sqlite_client.get("/api/chats/s1").json()["tree"]) == 2


def test_put_newer_accepted(sqlite_client):
    sqlite_client.put("/api/chats/s1", json=_session_blob(1000, "a"))
    resp = sqlite_client.put("/api/chats/s1", json=_session_blob(2000, "a", "b"))
    assert resp.status_code == 200
    assert len(sqlite_client.get("/api/chats/s1").json()["tree"]) == 2
