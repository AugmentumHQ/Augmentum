"""Unit tests for the pure helpers in the foundry wiring + trigger route.

The play stage itself is the live-stack seam (needs sidecar + orchestrator);
these pin the deterministic pieces around it.
"""
from __future__ import annotations

import base64

import pytest

from augmentum.proxy.foundry_routes import _slugify


def test_slugify_basic():
    assert _slugify("Coin Dash!") == "coin-dash"   # '!' dropped, not replaced
    assert _slugify("A/B  Test") == "ab--test"      # '/' dropped; spaces → '-'
    assert _slugify("") == "game"


def test_play_host_base_from_ws_setting(monkeypatch):
    from augmentum.coder.foundry import wire
    monkeypatch.setattr(wire.settings, "agent_bridge_base_url",
                        "ws://host.docker.internal:8080", raising=False)
    assert wire._play_host_base() == "http://host.docker.internal:8080"


def test_play_host_base_wss(monkeypatch):
    from augmentum.coder.foundry import wire
    monkeypatch.setattr(wire.settings, "agent_bridge_base_url",
                        "wss://augmentum.example:443/", raising=False)
    assert wire._play_host_base() == "https://augmentum.example:443"


def test_play_host_base_unset_falls_back(monkeypatch):
    from augmentum.coder.foundry import wire
    monkeypatch.setattr(wire.settings, "agent_bridge_base_url", "", raising=False)
    assert wire._play_host_base() == "http://augmentum:6100"


def test_render_play_host_embeds_payload_and_composer():
    from augmentum.proxy.game_agent_routes import _render_play_host
    payload = {
        "html": "<canvas></canvas>", "entry": "index.html",
        "files": {"index.html": {"c": "<canvas></canvas>", "e": "text"},
                  "assets/x.glb": {"c": base64.b64encode(b"GLB").decode(), "e": "base64"}},
        "agentBridge": {"wsUrl": "/api/game-agent/surfaces/js13k/bridge/s_1",
                        "sessionId": "s_1", "token": "t", "semanticToKey": {"action": "Space"}},
    }
    html = _render_play_host(payload)
    # Reuses the ONE client composer (no shim duplication).
    assert "/ui/scripts/bundle-composer.js" in html
    # Payload embedded so the browser can mount in agent mode.
    assert "window.__PLAY__" in html
    assert "s_1" in html and "assets/x.glb" in html
    # Per-file encoding is honored in the mount.
    assert "f.e || 'text'" in html


@pytest.mark.asyncio
async def test_captioner_auto_uses_router():
    from augmentum.coder.foundry.wire import make_captioner

    class _Router:
        async def caption(self, image_bytes, *, prompt, max_tokens, workload):
            return "looks fine"

    class _AppState:
        vision_router = _Router()

    cap = make_captioner(_AppState(), verify_model="")
    assert await cap(b"png", "inspect") == "looks fine"


@pytest.mark.asyncio
async def test_captioner_no_router_degrades_empty():
    from augmentum.coder.foundry.wire import make_captioner

    class _AppState:
        vision_router = None

    cap = make_captioner(_AppState(), verify_model="")
    assert await cap(b"png", "inspect") == ""
