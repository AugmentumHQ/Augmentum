"""Tests for the server-backed WebXR session spine."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.manager import StateManager
from augmentum.xr.session import XRSessionStore


@pytest.fixture
def xr_client(app):
    backend = SQLiteBackend(":memory:")
    asyncio.get_event_loop().run_until_complete(backend.connect())
    app.state.state_manager = StateManager(backend)
    app.state.xr_store = XRSessionStore(backend.conn)
    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc
    asyncio.get_event_loop().run_until_complete(backend.close())


def test_xr_capabilities_surface_server_contract(xr_client):
    resp = xr_client.get("/api/xr/capabilities")

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["server"]["sessionApi"] == 1
    assert data["server"]["surfaceHub"] is True
    assert data["server"]["spatialPanels"] is True
    assert data["server"]["handPanelGestures"] is True
    assert data["server"]["webEmbeds"] is True
    assert data["webxr"]["sessionMode"] == "immersive-vr"
    assert "immersive-ar" in data["webxr"]["sessionModes"]
    assert "dom-overlay" in data["webxr"]["optionalFeatures"]
    assert "layers" in data["webxr"]["optionalFeatures"]
    assert data["webxr"]["mixedReality"]["sessionMode"] == "immersive-ar"
    assert "hit-test" in data["webxr"]["mixedReality"]["optionalFeatures"]
    assert data["defaultRoomId"] == "modern-room"
    assert {s["id"] for s in data["surfaces"]} >= {
        "chat",
        "analytical",
        "agentic",
        "browse",
        "files",
        "coder",
        "narrative",
        "notes",
        "studio",
        "media",
        "devices",
        "games",
    }
    assert all("placement" in s for s in data["surfaces"])
    assert all(s.get("embedUrl", "").startswith("/ui/?xrEmbed=1") for s in data["surfaces"])
    assert data["roomStateDefaults"]["activeSurface"] == "voice"
    assert data["roomStateDefaults"]["selectedPanelAction"] == ""
    assert data["roomStateDefaults"]["surfacePanels"] == {}
    assert data["inputDefaults"]["pinchSelect"] is True
    assert data["inputDefaults"]["twoHandResize"] is True


def test_xr_session_lifecycle_and_events(xr_client):
    created = xr_client.post(
        "/api/xr/sessions",
        json={
            "surface": "voice",
            "voice_session_id": "voice_123",
            "device_hint": {"platform": "Quest"},
            "pwa": True,
        },
    )

    assert created.status_code == 201
    session = created.json()["session"]
    assert session["status"] == "preflight"
    assert session["voice_session_id"] == "voice_123"
    assert session["device_hint"]["platform"] == "Quest"
    assert session["device_hint"]["pwa"] is True
    assert session["room_manifest"]["assetUrl"].endswith("/modern-room.glb")
    assert "hubPanel" in session["room_manifest"]["anchors"]
    assert session["seat_layout"]["id"] == "default"
    assert session["room_state"]["activeSurface"] == "voice"
    assert session["room_state"]["surfacePanels"] == {}
    assert {s["id"] for s in session["surface_catalog"]} >= {
        "analytical",
        "agentic",
        "notes",
        "studio",
        "media",
    }

    session_id = session["id"]
    patched = xr_client.patch(
        f"/api/xr/sessions/{session_id}",
        json={
            "status": "running",
            "last_snapshot": {"hud": {"transcriptVisible": True}},
        },
    )
    assert patched.status_code == 200
    assert patched.json()["session"]["status"] == "running"

    event = xr_client.post(
        f"/api/xr/sessions/{session_id}/events",
        json={"type": "session_started", "payload": {"reference_space": "local-floor"}},
    )
    assert event.status_code == 200
    event_id = event.json()["event_id"]
    assert event_id > 0

    events = xr_client.get(f"/api/xr/sessions/{session_id}/events")
    assert events.status_code == 200
    event_list = events.json()["events"]
    assert len(event_list) == 1
    assert event_list[0]["id"] == event_id
    assert event_list[0]["session_id"] == session_id
    assert event_list[0]["type"] == "session_started"
    assert event_list[0]["payload"] == {"reference_space": "local-floor"}

    resume = xr_client.get(f"/api/xr/sessions/{session_id}/resume")
    assert resume.status_code == 200
    assert resume.json()["snapshot"] == {"hud": {"transcriptVisible": True}}


def test_xr_browser_panel_rejects_public_urls(xr_client):
    resp = xr_client.post(
        "/api/xr/browser-panels",
        json={"url": "https://example.com/", "width": 1440, "height": 900},
    )

    assert resp.status_code == 400
    assert "local" in resp.json()["detail"].lower()


def test_xr_browser_panel_routes_local_url_to_manager(xr_client):
    client, app = xr_client, xr_client.app

    class FakeBrowserPanelManager:
        def __init__(self):
            self.created = None
            self.panel = SimpleNamespace(
                id="xrb_fake",
                url="",
                width=1440,
                height=900,
                revision=1,
                last_frame=b"frame",
                last_media_type="image/jpeg",
            )

        async def create(self, **kwargs):
            self.created = kwargs
            self.panel.url = kwargs["url"]
            return self.panel

        async def get(self, panel_id, *, user_id):
            return self.panel if panel_id == self.panel.id else None

        async def capture(self, panel_id, *, user_id):
            return self.panel if panel_id == self.panel.id else None

        async def input(self, panel_id, *, user_id, event):
            self.panel.revision += 1
            return self.panel if panel_id == self.panel.id else None

        async def close(self, panel_id, *, user_id):
            return panel_id == self.panel.id

    manager = FakeBrowserPanelManager()
    app.state.xr_browser_panel_manager = manager
    client.cookies.set("augmentum_session", "cookie123")

    created = client.post("/api/xr/browser-panels", json={"url": "/ui/?xrEmbed=1"})
    assert created.status_code == 201
    data = created.json()["panel"]
    assert data["id"] == "xrb_fake"
    assert manager.created["url"].endswith("/ui/?xrEmbed=1")
    assert manager.created["auth_headers"]["Authorization"] == "Bearer test-token"
    assert manager.created["cookies"]["augmentum_session"] == "cookie123"

    frame = client.get("/api/xr/browser-panels/xrb_fake/frame")
    assert frame.status_code == 200
    assert frame.content == b"frame"

    clicked = client.post(
        "/api/xr/browser-panels/xrb_fake/input",
        json={"type": "click", "x": 0.25, "y": 0.5},
    )
    assert clicked.status_code == 200
    assert clicked.json()["revision"] == 2


def test_xr_seat_calibration_flows_into_new_session(xr_client):
    seat = xr_client.put(
        "/api/xr/seats/couch-left",
        json={
            "label": "Couch Left",
            "x": -0.6,
            "y": 0,
            "z": 2.1,
            "rotY": 3.0,
            "envId": "modern-room",
            "metadata": {"calibrated": True},
        },
    )
    assert seat.status_code == 200
    assert seat.json()["seat"]["metadata"] == {"calibrated": True}

    created = xr_client.post(
        "/api/xr/sessions",
        json={"seat_id": "couch-left", "room_id": "modern-room"},
    )
    assert created.status_code == 201
    session = created.json()["session"]
    assert session["seat_id"] == "couch-left"
    assert session["seat_layout"]["x"] == -0.6
    assert session["seat_layout"]["label"] == "Couch Left"


def test_voice_xr_surface_instruction_names_active_surface():
    from augmentum.proxy.voice_routes import (
        _xr_panel_action_addendum,
        _xr_surface_addendum,
    )

    coder = _xr_surface_addendum("coder")
    browse = _xr_surface_addendum("browse")

    assert "Coder surface" in coder
    assert "software work" in coder
    assert "Analyze surface" in _xr_surface_addendum("analytical")
    assert "Build surface" in _xr_surface_addendum("agentic")
    assert "Browse surface" in browse
    assert "Studio surface" in _xr_surface_addendum("studio")
    assert "Games surface" in _xr_surface_addendum("games")
    assert _xr_surface_addendum("unknown") == ""
    assert "summarize page" in _xr_panel_action_addendum("summarize_page")
    assert _xr_panel_action_addendum("") == ""
