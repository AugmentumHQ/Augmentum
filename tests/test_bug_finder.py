"""Smoke tests for bug_finder routes.

Verifies the module imports, the router exists with routes registered,
and unauthenticated callers don't crash the handler.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_bug_finder_routes_import():
    from augmentum.proxy.bug_finder_routes import router
    assert router is not None
    assert len(router.routes) > 0, "router has no endpoints registered"


def test_bug_finder_orchestrator_imports():
    from augmentum.bug_finder import orchestrator
    assert hasattr(orchestrator, "__name__")


def test_bug_finder_list_returns_clean_status(sqlite_client: TestClient):
    """Listing runs without auth should return a structured response, not 500."""
    resp = sqlite_client.get("/api/bug-finder/runs")
    assert resp.status_code in {200, 401, 403, 404}, (
        f"unexpected status {resp.status_code}: {resp.text[:200]}"
    )
