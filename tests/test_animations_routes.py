"""Smoke tests for /api/animations/* — upload, list, update, delete, serve."""
from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.manager import StateManager


@pytest.fixture
def sqlite_client(app, tmp_path, monkeypatch):
    monkeypatch.setenv("AUGMENTUM_ANIMATIONS_DIR", str(tmp_path / "anims"))
    backend = SQLiteBackend(":memory:")
    asyncio.get_event_loop().run_until_complete(backend.connect())
    app.state.state_manager = StateManager(backend)
    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc
    asyncio.get_event_loop().run_until_complete(backend.close())


def _upload(client, *, filename="dance.vrma", body=b"VRMA1\x00\x00fake",
            metadata=None):
    files = {"file": (filename, io.BytesIO(body), "application/octet-stream")}
    data = {}
    if metadata is not None:
        data["metadata"] = json.dumps(metadata)
    return client.post("/api/animations/upload", files=files, data=data)


class TestAnimationsList:
    def test_list_starts_empty(self, sqlite_client):
        resp = sqlite_client.get("/api/animations/list")
        assert resp.status_code == 200
        assert resp.json() == {"animations": []}


class TestAnimationsUpload:
    def test_upload_minimal_vrma(self, sqlite_client):
        resp = _upload(sqlite_client)
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["animation"]["id"].startswith("user:")
        assert body["animation"]["type"] == "vrma"
        assert body["animation"]["label"] == "dance"
        assert body["animation"]["userOwned"] is True

    def test_upload_with_metadata(self, sqlite_client):
        resp = _upload(sqlite_client, metadata={
            "label": "big spin",
            "roles": ["celebrate", "show-off"],
            "cost": 0.7,
            "loop_flag": True,
        })
        assert resp.status_code == 200
        a = resp.json()["animation"]
        assert a["label"] == "big spin"
        assert a["roles"] == ["celebrate", "show-off"]
        assert a["cost"] == 0.7
        assert a["loop"] is True

    def test_upload_bvh(self, sqlite_client):
        resp = _upload(sqlite_client, filename="capture.bvh",
                       body=b"HIERARCHY\n")
        assert resp.status_code == 200
        assert resp.json()["animation"]["type"] == "bvh"

    def test_upload_rejects_other_extension(self, sqlite_client):
        resp = _upload(sqlite_client, filename="dance.fbx")
        assert resp.status_code == 400

    def test_upload_rejects_empty(self, sqlite_client):
        resp = _upload(sqlite_client, body=b"")
        assert resp.status_code == 400

    def test_upload_then_list_includes_it(self, sqlite_client):
        _upload(sqlite_client)
        resp = sqlite_client.get("/api/animations/list")
        assert len(resp.json()["animations"]) == 1


class TestAnimationsUpdateDelete:
    def test_update_label(self, sqlite_client):
        anim_id = _upload(sqlite_client).json()["animation"]["id"]
        resp = sqlite_client.put(
            f"/api/animations/{anim_id}",
            json={"label": "renamed"},
        )
        assert resp.status_code == 200
        assert resp.json()["animation"]["label"] == "renamed"

    def test_update_rejects_source_path(self, sqlite_client):
        """source_path is intentionally not in the update whitelist."""
        anim_id = _upload(sqlite_client).json()["animation"]["id"]
        resp = sqlite_client.put(
            f"/api/animations/{anim_id}",
            json={"source_path": "/evil"},
        )
        # The PUT still succeeds (other fields could land); but
        # source isn't editable, so source URL stays the canonical
        # /api/animations/{id}/file form.
        assert resp.status_code == 200
        assert resp.json()["animation"]["source"] == (
            f"/api/animations/{anim_id}/file"
        )

    def test_update_404_for_missing(self, sqlite_client):
        resp = sqlite_client.put(
            "/api/animations/user:nope_aaa", json={"label": "x"},
        )
        assert resp.status_code == 404

    def test_delete_removes_row_and_file(self, sqlite_client, tmp_path):
        anim_id = _upload(sqlite_client).json()["animation"]["id"]
        # File should exist on disk now.
        anim_dir = tmp_path / "anims"
        files_before = sum(1 for _ in anim_dir.rglob("*.vrma"))
        assert files_before == 1
        resp = sqlite_client.delete(f"/api/animations/{anim_id}")
        assert resp.status_code == 200
        # And gone after.
        files_after = sum(1 for _ in anim_dir.rglob("*.vrma"))
        assert files_after == 0
        assert sqlite_client.get("/api/animations/list").json() == {
            "animations": [],
        }

    def test_delete_rejects_path_traversal(self, sqlite_client):
        resp = sqlite_client.delete("/api/animations/..%2Fevil")
        assert resp.status_code in (400, 404)
        # Most importantly, no 500 / crash.

    def test_delete_404_for_missing(self, sqlite_client):
        resp = sqlite_client.delete("/api/animations/user:nope_aaa")
        assert resp.status_code == 404


class TestAnimationsServe:
    def test_serve_returns_bytes(self, sqlite_client):
        body_bytes = b"VRMA1\x00stuff"
        anim_id = _upload(sqlite_client, body=body_bytes).json()["animation"]["id"]
        resp = sqlite_client.get(f"/api/animations/{anim_id}/file")
        assert resp.status_code == 200
        assert resp.content == body_bytes

    def test_serve_404_for_missing(self, sqlite_client):
        resp = sqlite_client.get("/api/animations/user:nope_aaa/file")
        assert resp.status_code == 404

    def test_serve_rejects_path_traversal(self, sqlite_client):
        # ../etc/passwd-style attempt
        resp = sqlite_client.get("/api/animations/..%2Fhostile/file")
        assert resp.status_code in (400, 404)
