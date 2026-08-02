"""Tests for the /api/vision/* routes.

Status endpoint must work whether the router is wired or not — the
operator UI needs a stable response shape to render a meaningful
diagnostic even when vision_provider_enabled is False.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_vision_status_endpoint_registered():
    """The route file's APIRouter must expose /api/vision/status."""
    from augmentum.proxy.vision_routes import router
    paths = {getattr(r, "path", "") for r in router.routes}
    assert "/api/vision/status" in paths


def test_vision_restart_endpoint_registered():
    """The restart endpoint must exist for operator recovery."""
    from augmentum.proxy.vision_routes import router
    paths = {getattr(r, "path", "") for r in router.routes}
    assert "/api/vision/restart" in paths


def test_vision_status_no_router_returns_inert_shape(client: TestClient):
    """Without a vision_router on app.state, status returns the
    inert shape — all booleans False, no port, no sibling state."""
    # The default `client` fixture in conftest builds an app whose
    # state has no vision_router attached. The endpoint must handle
    # this without 500-ing.
    resp = client.get("/api/vision/status")
    if resp.status_code == 404:
        pytest.skip("vision routes not registered in test app")
    assert resp.status_code == 200
    data = resp.json()
    # Required shape — UI binds these keys; renaming any of them is a
    # breaking change.
    for key in (
        "enabled", "has_router", "primary_available",
        "smolvlm_available", "sibling_port", "sibling_state", "base_url",
    ):
        assert key in data
    assert data["has_router"] is False
    assert data["primary_available"] is False
    assert data["smolvlm_available"] is False
