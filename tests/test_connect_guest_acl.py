"""Guest ACL gate + discovery exclusion (Connect Phase 3a).

A role='guest' user may reach ONLY the host it holds a live grant for, and never
appears in anyone's directory/search.
"""

from __future__ import annotations

import pytest

from augmentum.auth.session_manager import SessionManager
from augmentum.config import settings
from augmentum.connect.call_routing import handle_signaling_envelope
from augmentum.connect.guest_grant_store import create_grant, guest_scope_blocked
from augmentum.connect.hub import ConnectHub
from augmentum.connect.message_routing import handle_message_envelope
from augmentum.connect.protocol import (
    MSG_INVITE,
    MSG_TEXT_SEND,
    ConnectEnvelope,
)
from augmentum.notifications import NotificationHub
from augmentum.proxy.connect_routes import (
    _query_discoverable_same_instance_peers,
    _search_discoverable_peers,
)
from augmentum.state.backends.sqlite import SQLiteBackend


@pytest.fixture(autouse=True)
def _sentinel_handle(monkeypatch):
    monkeypatch.setattr(settings, "connect_instance_handle", "", raising=False)
    monkeypatch.setattr(settings, "augmentum_public_host", "", raising=False)


async def _env():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    sm = SessionManager(backend._conn)
    host = await sm.create_user("host", "supersecret")
    guest = await sm.create_user("visitor", "supersecret", role="guest")
    await create_grant(
        backend._conn, host_user_id=host.id, host_did=f"{host.id}@this-instance",
        guest_user_id=guest.id, guest_did=f"{guest.id}@this-instance", scopes="text,call",
    )
    return backend, backend._conn, sm, host, guest


# ── store-level gate ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_guest_scope_blocked_logic():
    backend, conn, sm, host, guest = await _env()
    try:
        other = await sm.create_user("stranger", "supersecret")
        # Guest → its host: allowed.
        assert await guest_scope_blocked(
            conn, sender_user_id=guest.id, sender_role="guest", target_user_id=host.id,
        ) is False
        # Guest → a stranger: blocked.
        assert await guest_scope_blocked(
            conn, sender_user_id=guest.id, sender_role="guest", target_user_id=other.id,
        ) is True
        # A normal user is NEVER blocked by this gate (fast-path).
        assert await guest_scope_blocked(
            conn, sender_user_id=host.id, sender_role="user", target_user_id=other.id,
        ) is False
    finally:
        await backend.close()


# ── routing integration: the gate is actually wired ───────────────

@pytest.mark.asyncio
async def test_message_envelope_blocks_guest_to_stranger():
    backend, conn, sm, host, guest = await _env()
    try:
        env = ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND, peer="usr_stranger@this-instance",
            data={"thread_id": "t1", "message_id": "m1", "body": "hi"},
        )
        res = await handle_message_envelope(
            conn=conn, connect_hub=ConnectHub(), notification_hub=NotificationHub(),
            env=env, sender_user_id=guest.id, sender_did=f"{guest.id}@this-instance",
            sender_role="guest",
        )
        assert res.error_code == "guest_scope_violation"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_message_envelope_allows_guest_to_host():
    backend, conn, sm, host, guest = await _env()
    try:
        env = ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND, peer=f"{host.id}@this-instance",
            data={"thread_id": "t1", "message_id": "m1", "body": "hi host"},
        )
        res = await handle_message_envelope(
            conn=conn, connect_hub=ConnectHub(), notification_hub=NotificationHub(),
            env=env, sender_user_id=guest.id, sender_did=f"{guest.id}@this-instance",
            sender_role="guest",
        )
        # The gate let it through — whatever else happens, it's NOT a scope block.
        assert res.error_code != "guest_scope_violation"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_call_envelope_blocks_guest_to_stranger():
    backend, conn, sm, host, guest = await _env()
    try:
        env = ConnectEnvelope(
            kind="msg", verb=MSG_INVITE, peer="usr_stranger@this-instance",
            data={"call_id": "c1", "modalities": "audio"},
        )
        res = await handle_signaling_envelope(
            conn=conn, connect_hub=ConnectHub(), notification_hub=NotificationHub(),
            env=env, sender_user_id=guest.id, sender_did=f"{guest.id}@this-instance",
            sender_party_id="p1", sender_role="guest",
        )
        assert res.error_code == "guest_scope_violation"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_regular_user_unaffected_by_gate():
    backend, conn, sm, host, guest = await _env()
    try:
        env = ConnectEnvelope(
            kind="msg", verb=MSG_INVITE, peer=f"{host.id}@this-instance",
            data={"call_id": "c1", "modalities": "audio"},
        )
        # A normal user (sender_role='user') calling anyone is never scope-blocked.
        res = await handle_signaling_envelope(
            conn=conn, connect_hub=ConnectHub(), notification_hub=NotificationHub(),
            env=env, sender_user_id="usr_normal", sender_did="usr_normal@this-instance",
            sender_party_id="p1", sender_role="user",
        )
        assert res.error_code != "guest_scope_violation"
    finally:
        await backend.close()


# ── discovery exclusion ───────────────────────────────────────────

async def _make_discoverable(conn, user_id):
    await conn.execute(
        "INSERT OR REPLACE INTO user_settings (user_id, key, value) VALUES (?, ?, ?)",
        (user_id, "ui.connectDiscoverableSameInstance", "true"),
    )
    await conn.commit()


@pytest.mark.asyncio
async def test_guest_excluded_from_directory_and_search():
    backend, conn, sm, host, guest = await _env()
    try:
        normal = await sm.create_user("alice", "supersecret", display_name="Alice")
        # Even if a guest somehow flips discoverability on, they stay hidden.
        for uid in (host.id, guest.id, normal.id):
            await _make_discoverable(conn, uid)

        dir_rows = await _query_discoverable_same_instance_peers(conn, host.id)
        ids = {r[0] for r in dir_rows}
        assert normal.id in ids
        assert guest.id not in ids  # guest never appears in the directory

        # Search by a fragment that would match the guest's username too.
        hits = await _search_discoverable_peers(conn, host.id, "i")  # matches visitor + alice
        hit_ids = {r[0] for r in hits}
        assert guest.id not in hit_ids
    finally:
        await backend.close()
