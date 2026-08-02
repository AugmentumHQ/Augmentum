"""Guest push-wake endpoint (Connect Phase 3d).

Drives ``guest_ping`` directly with a fake request, asserting the scope checks
and that it wakes the right user via the existing notification primitive (which
is monkeypatched — no live push)."""

from __future__ import annotations

import types

import pytest
from fastapi import HTTPException

from augmentum.auth.session_manager import SessionManager
from augmentum.config import settings
from augmentum.connect.guest_grant_store import create_grant
from augmentum.proxy import connect_routes
from augmentum.state.backends.sqlite import SQLiteBackend


class _FakeReq:
    def __init__(self, user, app, body):
        self.scope = {"user": user}
        self.app = app
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setattr(settings, "connect_enabled", True, raising=False)
    monkeypatch.setattr(settings, "connect_instance_handle", "", raising=False)
    monkeypatch.setattr(settings, "augmentum_public_host", "", raising=False)


async def _env():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    sm = SessionManager(backend._conn)
    host = await sm.create_user("host", "supersecret")
    guest = await sm.create_user("visitor", "supersecret", role="guest")
    app = types.SimpleNamespace(state=types.SimpleNamespace(
        state_manager=types.SimpleNamespace(backend=backend),
        notification_hub=None,
    ))
    return backend, backend._conn, sm, host, guest, app


def _capture(monkeypatch):
    calls = []

    async def fake_publish(conn, **kwargs):
        calls.append(kwargs)
        return "ntf_test"

    monkeypatch.setattr("augmentum.notifications.hub.publish_and_dispatch", fake_publish)
    return calls


@pytest.mark.asyncio
async def test_guest_text_ping_wakes_host(monkeypatch):
    backend, conn, sm, host, guest, app = await _env()
    try:
        await create_grant(
            conn, host_user_id=host.id, host_did=f"{host.id}@this-instance",
            guest_user_id=guest.id, guest_did=f"{guest.id}@this-instance", scopes="text",
        )
        calls = _capture(monkeypatch)
        req = _FakeReq(guest, app, {"kind": "text", "peer_did": f"{host.id}@this-instance"})
        res = await connect_routes.guest_ping(req)
        assert res["woke"] is True
        assert len(calls) == 1
        assert calls[0]["user_id"] == host.id           # woke the HOST
        assert "message" in calls[0]["title"].lower()
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_guest_call_ping_blocked_without_call_scope(monkeypatch):
    backend, conn, sm, host, guest, app = await _env()
    try:
        await create_grant(
            conn, host_user_id=host.id, host_did=f"{host.id}@this-instance",
            guest_user_id=guest.id, guest_did=f"{guest.id}@this-instance", scopes="text",
        )
        calls = _capture(monkeypatch)
        req = _FakeReq(guest, app, {"kind": "call", "peer_did": f"{host.id}@this-instance"})
        with pytest.raises(HTTPException) as exc:
            await connect_routes.guest_ping(req)
        assert exc.value.status_code == 403
        assert not calls  # never woke anyone
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_guest_call_ping_allowed_when_call_scoped(monkeypatch):
    backend, conn, sm, host, guest, app = await _env()
    try:
        await create_grant(
            conn, host_user_id=host.id, host_did=f"{host.id}@this-instance",
            guest_user_id=guest.id, guest_did=f"{guest.id}@this-instance", scopes="text,call",
        )
        calls = _capture(monkeypatch)
        req = _FakeReq(guest, app, {"kind": "call", "peer_did": f"{host.id}@this-instance"})
        res = await connect_routes.guest_ping(req)
        assert res["woke"] is True and calls[0]["user_id"] == host.id
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_guest_cannot_ping_a_stranger(monkeypatch):
    backend, conn, sm, host, guest, app = await _env()
    try:
        stranger = await sm.create_user("stranger", "supersecret")
        await create_grant(
            conn, host_user_id=host.id, host_did=f"{host.id}@this-instance",
            guest_user_id=guest.id, guest_did=f"{guest.id}@this-instance", scopes="text,call",
        )
        calls = _capture(monkeypatch)
        req = _FakeReq(guest, app, {"kind": "text", "peer_did": f"{stranger.id}@this-instance"})
        with pytest.raises(HTTPException) as exc:
            await connect_routes.guest_ping(req)
        assert exc.value.status_code == 403
        assert not calls
    finally:
        await backend.close()
