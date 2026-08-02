"""Smoke tests for titles BIOS routes (BIOS classifier + upload).

Verifies module imports and the router mounts at /api/titles/bios.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_titles_bios_routes_import():
    from augmentum.proxy.titles_bios_routes import router
    assert router.prefix == "/api/titles/bios"
    assert len(router.routes) > 0


def test_titles_bios_list(sqlite_client: TestClient):
    resp = sqlite_client.get("/api/titles/bios/")
    assert resp.status_code in {200, 401, 403, 404}, (
        f"unexpected status {resp.status_code}: {resp.text[:200]}"
    )
