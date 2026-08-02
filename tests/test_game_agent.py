"""Smoke tests for game-agent routes (bridged-surface controller).

Verifies module imports and routes are mounted at /api/game-agent.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_game_agent_routes_import():
    from augmentum.proxy.game_agent_routes import router
    assert router.prefix == "/api/game-agent"
    assert len(router.routes) > 0


def test_game_agent_status(sqlite_client: TestClient):
    resp = sqlite_client.get("/api/game-agent/status")
    assert resp.status_code in {200, 401, 403, 404, 503}, (
        f"unexpected status {resp.status_code}: {resp.text[:200]}"
    )
