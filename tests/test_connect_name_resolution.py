"""Connect contacts + call history resolve peer usernames, never raw DIDs.

Regression for the "saved contacts and recent calls showed the DID instead of
usernames" report. Contacts created implicitly on first inbound traffic store
an empty ``peer_display_name``, and call rows store only DIDs — both list
routes now resolve ``users.display_name``/``username`` from the DID server-side
so the UI never falls back to the ``usr_<hash>@instance`` form.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from augmentum.auth.session_manager import SessionManager
from augmentum.config import settings
from augmentum.connect.contact_store import add_contact
from augmentum.connect.contacts import local_did_for
from augmentum.connect.fabric_inbound import apply_inbound_fabric_envelope
from augmentum.connect.fabric_transport import dispatch_fabric_envelope
from augmentum.connect.protocol import (
    MSG_TEXT_SEND,
    ConnectEnvelope,
    serialise_envelope,
)
from augmentum.proxy.server import create_app
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.manager import StateManager
from augmentum.state.settings_store import SettingsStore


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def _reset_handle(monkeypatch):
    # Pin the instance handle so DIDs minted in the test resolve as local.
    monkeypatch.setattr(settings, "connect_instance_handle", "", raising=False)
    monkeypatch.setattr(settings, "augmentum_public_host", "", raising=False)


@pytest.fixture
def app_env():
    backend = SQLiteBackend(":memory:")
    _run(backend.connect())

    app = create_app()
    app.state.session_manager = SessionManager(backend._conn)
    app.state.state_manager = StateManager(backend)
    app.state.settings_store = SettingsStore(backend.conn)

    me = _run(app.state.session_manager.create_user("alice", "supersecret"))
    peer = _run(app.state.session_manager.create_user("bob", "supersecret"))
    yield app, backend, me, peer
    _run(backend.close())


def _client_for(app, user):
    token = _run(app.state.session_manager.create_session(user.id))
    tc = TestClient(app)
    tc.headers.update({"Authorization": f"Bearer {token}"})
    return tc


def test_contact_with_empty_display_name_resolves_username(app_env):
    app, backend, me, peer = app_env
    # Simulate an implicit contact: stored with NO display name (the
    # ensure_contact path on first inbound message/call).
    peer_did = local_did_for(peer.id)
    _run(add_contact(
        backend.conn, user_id=me.id, peer_did=peer_did,
        peer_display_name="", discovery_source="implicit",
    ))

    r = _client_for(app, me).get("/api/connect/contacts")
    assert r.status_code == 200, r.text
    contacts = r.json()["contacts"]
    assert len(contacts) == 1
    c = contacts[0]
    assert c["peer_did"] == peer_did
    # Resolved to the peer's username, NOT the raw usr_<hash> DID.
    assert c["peer_display_name"] == "bob"
    assert "@" not in c["peer_display_name"]


def test_contact_keeps_explicit_display_name(app_env):
    # An explicitly-named contact isn't overwritten by the resolver.
    app, backend, me, peer = app_env
    peer_did = local_did_for(peer.id)
    _run(add_contact(
        backend.conn, user_id=me.id, peer_did=peer_did,
        peer_display_name="Bobby (work)",
    ))
    r = _client_for(app, me).get("/api/connect/contacts")
    assert r.json()["contacts"][0]["peer_display_name"] == "Bobby (work)"


def test_call_history_resolves_peer_username(app_env):
    app, backend, me, peer = app_env
    my_did = local_did_for(me.id)
    peer_did = local_did_for(peer.id)
    # An outgoing call row (initiator = me) where only DIDs are persisted.
    _run(backend.conn.execute(
        """INSERT INTO call_sessions
               (call_id, user_id, initiator_did, receiver_did,
                modalities, state, end_reason, initiated_at,
                connected_at, ended_at, quality_rating, quality_notes,
                becca_present)
           VALUES (?, ?, ?, ?, 'audio', 'ended', '', '2026-06-22T10:00:00+00:00',
                   NULL, NULL, NULL, '', 0)""",
        ("call_xyz", me.id, my_did, peer_did),
    ))
    _run(backend.conn.commit())

    r = _client_for(app, me).get("/api/connect/calls")
    assert r.status_code == 200, r.text
    calls = r.json()["calls"]
    assert len(calls) == 1
    assert calls[0]["peer_did"] == peer_did
    assert calls[0]["peer_display_name"] == "bob"
    assert calls[0]["direction"] == "outgoing"


# --------------------------------------------------------------------------
# Cross-instance (fabric) peers carry + resolve their remote username
# --------------------------------------------------------------------------

class _CapturingCoordinator:
    """Stands in for the fabric coordinator; records the dispatched payload."""

    def __init__(self):
        self.payload = None

    async def _resolve(self, hostname):  # pragma: no cover - unused hook
        return "node-x"

    def peer_state(self, node_id):  # pragma: no cover - sentinel rewrite skip
        return None

    async def send_to_peer(self, node_id, *, msg_type, payload):
        self.payload = payload
        return True


def test_sender_attaches_its_display_name_to_fabric_payload(app_env):
    # When a local user dispatches over fabric, the envelope must carry the
    # sender's OWN username so the remote box can render it (the remote has no
    # local row to resolve our usr_<hash> against).
    app, backend, me, _peer = app_env
    coord = _CapturingCoordinator()
    env = ConnectEnvelope(
        kind="msg", verb=MSG_TEXT_SEND, peer="someone@remote-box",
        data={"thread_id": "t1", "message_id": "m1", "body": "hi"},
    )
    # Patch node resolution so dispatch reaches send_to_peer with our payload.
    import augmentum.connect.fabric_transport as ft

    orig = ft._resolve_node_id

    async def _fake_resolve(coordinator, *, hostname):
        return "node-x"

    ft._resolve_node_id = _fake_resolve
    try:
        _run(dispatch_fabric_envelope(
            backend.conn,
            coordinator=coord,
            target_hostname="remote-box",
            source_did=local_did_for(me.id),
            sender_user_id=me.id,
            sender_party_id="",
            envelope=env,
        ))
    finally:
        ft._resolve_node_id = orig

    assert coord.payload is not None
    # me was created as "alice" → that's the name that rides across.
    assert coord.payload["source_display_name"] == "alice"


def test_inbound_fabric_remembers_remote_name_and_lists_resolve(app_env):
    # A remote peer's name rides in on their first message; the receiver caches
    # it on a contact row, and BOTH the contacts list and the call history then
    # render the username instead of the fabric DID local-part.
    app, backend, me, _peer = app_env
    remote_did = "usr_remote_hash@instance-A"
    env = ConnectEnvelope(
        kind="msg", verb=MSG_TEXT_SEND, peer=local_did_for(me.id),
        data={"thread_id": "t-fab", "message_id": "m-fab", "body": "hello"},
    )
    payload = {
        "envelope": serialise_envelope(env),
        "source_did": remote_did,
        "target_user_id": me.id,
        "sender_party_id": "",
        "source_display_name": "Rita Remote",
    }
    res = _run(apply_inbound_fabric_envelope(
        backend.conn,
        connect_hub=None,
        notification_hub=None,
        fabric_payload=payload,
    ))
    assert res["applied"], res

    # Contact row now carries the remembered name.
    r = _client_for(app, me).get("/api/connect/contacts")
    contacts = r.json()["contacts"]
    fab = [c for c in contacts if c["peer_did"] == remote_did]
    assert fab and fab[0]["peer_display_name"] == "Rita Remote"

    # An incoming call from that same fabric DID resolves via the contact cache
    # (display_name_for_did can't — there's no local user row for a fabric peer).
    _run(backend.conn.execute(
        """INSERT INTO call_sessions
               (call_id, user_id, initiator_did, receiver_did,
                modalities, state, end_reason, initiated_at,
                connected_at, ended_at, quality_rating, quality_notes,
                becca_present)
           VALUES (?, ?, ?, ?, 'audio', 'missed', '', '2026-06-22T11:00:00+00:00',
                   NULL, NULL, NULL, '', 0)""",
        ("call_fab", me.id, remote_did, local_did_for(me.id)),
    ))
    _run(backend.conn.commit())

    r = _client_for(app, me).get("/api/connect/calls")
    calls = r.json()["calls"]
    fabcall = [c for c in calls if c["peer_did"] == remote_did]
    assert fabcall and fabcall[0]["peer_display_name"] == "Rita Remote"
    assert fabcall[0]["direction"] == "incoming"
