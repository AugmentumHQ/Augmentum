"""Smoke tests for the Game Portal (routes + js13k provider).

Verifies that the modules import, the GameBrowseResult round-trips,
and that the browse route rejects unauthenticated callers. Hitting
the live js13k catalog is a Tier 3 concern -- those tests belong in
``tests/live/`` gated by ``--run-live``.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from augmentum.games.models import GameBrowseResult
from augmentum.games.providers import js13k as js13k_provider


def test_browse_result_roundtrip():
    r = GameBrowseResult(
        source="js13k", source_id="2024/foo", name="Foo", author="foo",
        tagline="a game", thumbnail_url="https://cdn/x.png",
        source_url="https://js13kgames.com/games/foo",
        embed_url="https://js13kgames.com/games/foo",
    )
    d = r.to_dict()
    assert d["source"] == "js13k"
    assert d["source_id"] == "2024/foo"
    assert d["name"] == "Foo"
    assert d["play_mode"] == "embed"


def test_js13k_provider_module_imports():
    """Provider module loads and exposes the expected browse/details api."""
    assert hasattr(js13k_provider, "browse")
    assert hasattr(js13k_provider, "fetch_details")


def test_games_browse_rejects_unknown_source(client: TestClient):
    """Unknown source returns a structured 400, not a silent 200."""
    resp = client.get("/api/games/browse?source=doesnotexist")
    assert resp.status_code == 400
    assert "Unknown source" in resp.json().get("error", "")


def test_games_browse_rejects_legacy_itch_source(client: TestClient):
    """itch.io was removed; ?source=itch must 400 (not 200), confirming the
    old source can't sneak back in via cached client code."""
    resp = client.get("/api/games/browse?source=itch")
    assert resp.status_code == 400


def test_games_router_prefix():
    from augmentum.proxy.games_routes import router
    assert router.prefix == "/api/games"
