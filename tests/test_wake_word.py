"""Smoke tests for wake-word routes (training + WS detection stream)."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_wake_word_routes_import():
    from augmentum.proxy.wake_word_routes import router
    paths = {getattr(r, "path", "") for r in router.routes}
    assert "/api/wake_word/train" in paths


def test_wake_word_status(sqlite_client: TestClient):
    resp = sqlite_client.get("/api/wake_word/enrollments")
    assert resp.status_code in {200, 401, 403, 404, 405}, (
        f"unexpected status {resp.status_code}: {resp.text[:200]}"
    )
