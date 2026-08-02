"""Smoke tests for device routes (connected devices / DLNA / cast).

Verifies the routers mount at /api/devices and /api/cast, and that
unauthenticated discovery doesn't crash the handler.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_device_routes_import():
    from augmentum.proxy.device_routes import cast_blob_router, router
    assert router.prefix == "/api/devices"
    assert cast_blob_router.prefix == "/api/cast"


def test_device_discover_endpoint_registered():
    from augmentum.proxy.device_routes import router
    paths = {getattr(r, "path", "") for r in router.routes}
    assert "/api/devices/discover" in paths


def test_device_capabilities(sqlite_client: TestClient):
    resp = sqlite_client.get("/api/devices/capabilities")
    assert resp.status_code in {200, 401, 403, 404}, (
        f"unexpected status {resp.status_code}: {resp.text[:200]}"
    )
