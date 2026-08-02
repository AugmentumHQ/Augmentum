"""Smoke tests for titles save-game routes.

Verifies module imports and routes are mounted under /api/titles.
"""
from __future__ import annotations


def test_titles_saves_routes_import():
    from augmentum.proxy.titles_saves_routes import router
    assert router.prefix == "/api/titles"
    paths = {getattr(r, "path", "") for r in router.routes}
    assert any("saves" in p for p in paths), "no /saves endpoints registered"
