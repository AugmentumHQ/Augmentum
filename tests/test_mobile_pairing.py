"""Android/mobile pairing flow tests."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def mobile_pair_app():
    from augmentum.auth.mobile_pairing import MobilePairStore
    from augmentum.auth.session_manager import SessionManager
    from augmentum.proxy.server import create_app
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.manager import StateManager

    backend = SQLiteBackend(":memory:")
    _run(backend.connect())

    app = create_app()
    app.state.session_manager = SessionManager(backend._conn)
    app.state.state_manager = StateManager(backend)
    app.state.mobile_pair_store = MobilePairStore()

    yield app

    _run(backend.close())


def _auth_client(app, *, username: str = "alice"):
    user = _run(app.state.session_manager.create_user(username, "supersecret", role="user"))
    token = _run(app.state.session_manager.create_session(user.id, source="web"))
    client = TestClient(app)
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client, user, token


def _claim_payload(device_id: str = "android-test-device"):
    return {
        "device_id": device_id,
        "label": "Pixel Test",
        "platform": "android",
        "app_version": "0.1-test",
        "public_key": "test-public-key",
        "key_alg": "ed25519",
        "scopes": ["chat.sync", "tts.engine"],
        "capabilities": ["android.tts_engine@1", "android.chat_sync@1"],
    }


def _pair_to_grant(app):
    web, _, _ = _auth_client(app)
    start = web.post("/api/auth/pair/start")
    assert start.status_code == 200
    pair_code = start.json()["pair_code"]

    anon = TestClient(app)
    claim = anon.post(f"/api/auth/pair/claim/{pair_code}", json=_claim_payload())
    assert claim.status_code == 200
    claim_token = claim.json()["claim_token"]

    status = web.get(f"/api/auth/pair/status/{pair_code}")
    assert status.status_code == 200
    assert status.json()["state"] == "claimed"
    assert status.json()["claim"]["label"] == "Pixel Test"

    approve = web.post(f"/api/auth/pair/approve/{pair_code}")
    assert approve.status_code == 200

    poll = anon.get(f"/api/auth/pair/poll/{claim_token}")
    assert poll.status_code == 200
    grant_token = poll.json()["grant_token"]
    return web, anon, grant_token


def test_mobile_pair_happy_path_creates_device_and_android_session(mobile_pair_app):
    web, anon, grant_token = _pair_to_grant(mobile_pair_app)

    finish = anon.post("/api/auth/pair/finish", json={"grant_token": grant_token})
    assert finish.status_code == 200
    data = finish.json()
    assert data["source"] == "android"
    assert data["auth_type"] == "bearer"
    assert data["session_token"]
    assert data["device"]["label"] == "Pixel Test"
    assert data["device"]["scopes"] == ["chat.sync", "tts.engine"]

    devices = web.get("/api/auth/pair/devices")
    assert devices.status_code == 200
    assert len(devices.json()["devices"]) == 1
    assert devices.json()["devices"][0]["device_id"] == "android-test-device"


def test_mobile_pair_finish_returns_lan_primary_without_remote_when_unconfigured(mobile_pair_app, monkeypatch):
    # No operator public host configured and no Tailscale SAN → phone gets the
    # LAN address it reached us on as the primary, and no remote endpoint.
    monkeypatch.delenv("AUGMENTUM_TLS_EXTRA_SANS", raising=False)
    _, anon, grant_token = _pair_to_grant(mobile_pair_app)
    data = anon.post("/api/auth/pair/finish", json={"grant_token": grant_token}).json()
    assert data["server_url"]  # the host the phone reached us on
    assert "testserver" in data["server_url"]
    assert data["remote_url"] == ""


def test_mobile_pair_finish_learns_explicit_public_host(mobile_pair_app):
    # Operator set AUGMENTUM_PUBLIC_HOST (custom domain) → phone auto-learns it
    # as the remote endpoint. Explicit override wins.
    from augmentum.devices.host_resolver import PublicHostResolver

    mobile_pair_app.state.public_host_resolver = PublicHostResolver(
        configured="becca.example.com"
    )
    _, anon, grant_token = _pair_to_grant(mobile_pair_app)
    data = anon.post("/api/auth/pair/finish", json={"grant_token": grant_token}).json()
    # Primary stays the LAN/request host; remote is the configured public host.
    assert "testserver" in data["server_url"]
    assert data["remote_url"] == "http://becca.example.com"


def test_mobile_pair_finish_auto_derives_tailscale_remote_from_sans(mobile_pair_app, monkeypatch):
    # OSS-automatic path: no AUGMENTUM_PUBLIC_HOST, but start.sh auto-populated
    # the cert SANs with the host's Tailscale IP. /finish reuses it so the phone
    # gets an off-LAN endpoint with ZERO operator config — and without touching
    # the resolver (so cast URLs are unaffected).
    monkeypatch.setenv(
        "AUGMENTUM_TLS_EXTRA_SANS", "IP:192.168.1.42,IP:100.64.0.1"
    )
    _, anon, grant_token = _pair_to_grant(mobile_pair_app)
    data = anon.post("/api/auth/pair/finish", json={"grant_token": grant_token}).json()
    assert "testserver" in data["server_url"]
    # TestClient reaches us on default port 80 (no :port in host) → bare IP.
    assert data["remote_url"] == "https://100.64.0.1"


def test_mobile_pair_finish_no_remote_when_sans_lan_only(mobile_pair_app, monkeypatch):
    # SANs with only LAN IPs (no Tailscale) → no remote endpoint offered.
    monkeypatch.setenv("AUGMENTUM_TLS_EXTRA_SANS", "IP:192.168.1.42")
    _, anon, grant_token = _pair_to_grant(mobile_pair_app)
    data = anon.post("/api/auth/pair/finish", json={"grant_token": grant_token}).json()
    assert data["remote_url"] == ""


def test_mobile_pair_start_returns_qr_material_and_consumed_status(mobile_pair_app):
    web, _, _ = _auth_client(mobile_pair_app)
    start = web.post("/api/auth/pair/start")
    assert start.status_code == 200
    data = start.json()
    pair_code = data["pair_code"]
    assert data["pair_url"].startswith("augmentum://pair?")
    assert data["qr_url"] == f"/api/auth/pair/qr/{pair_code}.svg"
    assert data["status_path"] == f"/api/auth/pair/status/{pair_code}"

    anon = TestClient(mobile_pair_app)
    qr = anon.get(data["qr_url"])
    assert qr.status_code == 200
    assert qr.headers["content-type"].startswith("image/svg+xml")

    claim = anon.post(f"/api/auth/pair/claim/{pair_code}", json=_claim_payload("android-qr-contract"))
    assert claim.status_code == 200
    approve = web.post(f"/api/auth/pair/approve/{pair_code}")
    assert approve.status_code == 200
    grant_token = anon.get(f"/api/auth/pair/poll/{claim.json()['claim_token']}").json()["grant_token"]

    finish = anon.post("/api/auth/pair/finish", json={"grant_token": grant_token})
    assert finish.status_code == 200

    status = web.get(data["status_path"])
    assert status.status_code == 200
    assert status.json()["state"] == "consumed"


def test_mobile_pair_grant_is_single_use(mobile_pair_app):
    _, anon, grant_token = _pair_to_grant(mobile_pair_app)

    first = anon.post("/api/auth/pair/finish", json={"grant_token": grant_token})
    assert first.status_code == 200

    second = anon.post("/api/auth/pair/finish", json={"grant_token": grant_token})
    assert second.status_code == 401


def test_revoking_mobile_device_invalidates_android_session(mobile_pair_app):
    web, anon, grant_token = _pair_to_grant(mobile_pair_app)

    finish = anon.post("/api/auth/pair/finish", json={"grant_token": grant_token})
    assert finish.status_code == 200
    android_token = finish.json()["session_token"]

    android = TestClient(mobile_pair_app)
    android.headers.update({"Authorization": f"Bearer {android_token}"})
    assert android.get("/api/auth/pair/devices").status_code == 200

    devices = web.get("/api/auth/pair/devices").json()["devices"]
    mobile_id = devices[0]["id"]
    revoke = web.post(f"/api/auth/pair/devices/{mobile_id}/revoke")
    assert revoke.status_code == 200
    assert revoke.json()["revoked_sessions"] >= 1

    assert android.get("/api/auth/pair/devices").status_code == 401


def test_wrong_user_cannot_approve_mobile_pair(mobile_pair_app):
    alice, _, _ = _auth_client(mobile_pair_app, username="alice")
    start = alice.post("/api/auth/pair/start")
    pair_code = start.json()["pair_code"]

    anon = TestClient(mobile_pair_app)
    claim = anon.post(f"/api/auth/pair/claim/{pair_code}", json=_claim_payload("android-wrong-user"))
    assert claim.status_code == 200

    bob, _, _ = _auth_client(mobile_pair_app, username="bob")
    approve = bob.post(f"/api/auth/pair/approve/{pair_code}")
    assert approve.status_code == 409


def test_non_mobile_source_device_session_does_not_require_mobile_device(mobile_pair_app):
    _, user, _ = _auth_client(mobile_pair_app)
    token = _run(
        mobile_pair_app.state.session_manager.create_session(
            user.id,
            source="cast_receiver",
            source_device_id="receiver-123",
        )
    )

    validated = _run(mobile_pair_app.state.session_manager.validate_token(token))
    assert validated is not None
    assert validated.id == user.id


def test_mobile_source_session_without_device_binding_is_rejected(mobile_pair_app):
    _, user, _ = _auth_client(mobile_pair_app)
    token = _run(
        mobile_pair_app.state.session_manager.create_session(
            user.id,
            source="android",
        )
    )

    assert _run(mobile_pair_app.state.session_manager.validate_token(token)) is None
