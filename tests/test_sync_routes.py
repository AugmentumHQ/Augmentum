"""Tests for sync_routes.py — reading positions + browse-history ingest.

The stores are mocked (their real logic is covered in test_sync_store.py and
test_discovery_store.py); these assert the route wiring: auth gating, request
validation, store dispatch, and response shaping.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def _mock_sync_store():
    store = MagicMock()
    store.upsert_positions = AsyncMock(return_value=(2, 1, ["book:stale"]))
    store.list_since = AsyncMock(return_value=[
        {"key": "book:1", "kind": "book", "position_fraction": 0.5,
         "position_detail": 10, "last_read_ms": 2000, "device_id": "tablet",
         "title": "A Book"},
    ])
    return store


def _mock_discovery_store(collision=False):
    store = MagicMock()
    store.upsert_history = AsyncMock(
        return_value={"collision": True} if collision else {"id": "h1"}
    )
    return store


class TestPushReadingPositions:
    def test_no_store_503(self, client):
        resp = client.post("/api/sync/reading-positions",
                           json={"positions": []})
        assert resp.status_code == 503

    def test_bad_positions_400(self, app, client):
        app.state.sync_store = _mock_sync_store()
        resp = client.post("/api/sync/reading-positions",
                           json={"positions": "nope"})
        assert resp.status_code == 400

    def test_success_shapes_response(self, app, client):
        app.state.sync_store = _mock_sync_store()
        resp = client.post(
            "/api/sync/reading-positions",
            json={"device_id": "phone", "positions": [
                {"key": "book:1", "kind": "book", "position_fraction": 0.5,
                 "position_detail": 10, "last_read_ms": 2000,
                 "device_id": "phone", "title": "A Book"},
            ]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted"] == 2
        assert body["rejected"] == 1
        assert body["conflicts"] == ["book:stale"]

    def test_upsert_called_with_user_id(self, app, client):
        store = _mock_sync_store()
        app.state.sync_store = store
        client.post("/api/sync/reading-positions",
                    json={"positions": [{"key": "a"}]})
        _, kwargs = store.upsert_positions.call_args
        assert kwargs.get("user_id")  # non-empty authenticated user


class TestPullReadingPositions:
    def test_no_store_503(self, client):
        resp = client.get("/api/sync/reading-positions")
        assert resp.status_code == 503

    def test_success_includes_now_ms(self, app, client):
        app.state.sync_store = _mock_sync_store()
        resp = client.get("/api/sync/reading-positions?since_ms=100&device_id=phone")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["positions"]) == 1
        assert body["now_ms"] > 0

    def test_excludes_caller_device(self, app, client):
        store = _mock_sync_store()
        app.state.sync_store = store
        client.get("/api/sync/reading-positions?since_ms=0&device_id=phone")
        _, kwargs = store.list_since.call_args
        assert kwargs.get("exclude_device_id") == "phone"
        assert kwargs.get("since_ms") == 0


class TestPushBrowseHistory:
    def test_no_store_503(self, client):
        resp = client.post("/api/sync/browse-history", json={"events": []})
        assert resp.status_code == 503

    def test_bad_events_400(self, app, client):
        app.state.discovery_store = _mock_discovery_store()
        resp = client.post("/api/sync/browse-history", json={"events": "nope"})
        assert resp.status_code == 400

    def test_success_counts_accepted(self, app, client):
        app.state.discovery_store = _mock_discovery_store()
        resp = client.post(
            "/api/sync/browse-history",
            json={"device_id": "phone", "events": [
                {"url": "https://example.com/a", "title": "A",
                 "opened_ms": 1000, "duration_ms": 5000,
                 "content_type": "article", "device_id": "phone"},
                {"url": "", "title": "blank"},  # skipped — no url
            ]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted"] == 1
        assert body["skipped"] == 1

    def test_collision_counts_skipped(self, app, client):
        app.state.discovery_store = _mock_discovery_store(collision=True)
        resp = client.post(
            "/api/sync/browse-history",
            json={"events": [
                {"url": "https://example.com/a", "title": "A",
                 "content_type": "article", "device_id": "phone"},
            ]},
        )
        assert resp.status_code == 200
        assert resp.json()["skipped"] == 1

    def test_upsert_history_scoped_to_user(self, app, client):
        store = _mock_discovery_store()
        app.state.discovery_store = store
        client.post(
            "/api/sync/browse-history",
            json={"events": [
                {"url": "https://example.com/a", "content_type": "article"},
            ]},
        )
        _, kwargs = store.upsert_history.call_args
        assert kwargs.get("user_id")
        assert kwargs.get("metadata", {}).get("source") == "phone"
