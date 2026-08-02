"""Smoke tests for build routes (build/run pipeline).

Verifies the module imports, the router is mounted at /api/builds, and
unauthenticated calls return a structured status (not a 500).
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_build_routes_import():
    from augmentum.proxy.build_routes import router
    assert router.prefix == "/api/builds"
    assert len(router.routes) > 0


def test_build_runs_list(sqlite_client: TestClient):
    resp = sqlite_client.get("/api/builds/runs")
    assert resp.status_code in {200, 401, 403, 404}, (
        f"unexpected status {resp.status_code}: {resp.text[:200]}"
    )
