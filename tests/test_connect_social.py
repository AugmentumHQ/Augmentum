"""Connect social layer — profiles, presence, handle search, directory
enrichment (comms platform Phase 1, task 3)."""

from __future__ import annotations

import pytest

from augmentum.auth.session_manager import SessionManager
from augmentum.config import settings
from augmentum.connect.hub import ConnectHub
from augmentum.connect.presence_store import get_presence_for, mark_presence
from augmentum.connect.profile_store import (
    get_profile,
    get_profiles_for,
    upsert_profile,
)
from augmentum.proxy.connect_routes import (
    _enrich_people,
    _query_discoverable_same_instance_peers,
    _search_discoverable_peers,
)
from augmentum.state.backends.sqlite import SQLiteBackend


async def _env():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    return backend, backend._conn, SessionManager(backend._conn)


async def _make_discoverable(conn, user_id: str) -> None:
    await conn.execute(
        "INSERT OR REPLACE INTO user_settings (user_id, key, value) VALUES (?, ?, ?)",
        (user_id, "ui.connectDiscoverableSameInstance", "true"),
    )
    await conn.commit()


@pytest.fixture(autouse=True)
def _reset_handle(monkeypatch):
    monkeypatch.setattr(settings, "connect_instance_handle", "myhost.example.com", raising=False)
    monkeypatch.setattr(settings, "augmentum_public_host", "", raising=False)


# ── profiles ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_profile_defaults_then_partial_upsert():
    backend, conn, sm = await _env()
    try:
        u = await sm.create_user("alice", "supersecret")
        prof = await get_profile(conn, user_id=u.id)
        assert prof["bio"] == "" and prof["status_message"] == ""

        await upsert_profile(conn, user_id=u.id, status_message="building things")
        prof = await get_profile(conn, user_id=u.id)
        assert prof["status_message"] == "building things"
        assert prof["bio"] == ""  # untouched field stays empty

        # Second partial update doesn't clobber the first field.
        await upsert_profile(conn, user_id=u.id, bio="hi there")
        prof = await get_profile(conn, user_id=u.id)
        assert prof["status_message"] == "building things"
        assert prof["bio"] == "hi there"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_profile_truncates_to_cap_and_ignores_none():
    backend, conn, sm = await _env()
    try:
        u = await sm.create_user("bob", "supersecret")
        await upsert_profile(conn, user_id=u.id, status_message="x" * 500, bio=None)
        prof = await get_profile(conn, user_id=u.id)
        assert len(prof["status_message"]) == 140  # cap
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_get_profiles_for_bulk():
    backend, conn, sm = await _env()
    try:
        a = await sm.create_user("aa", "supersecret")
        b = await sm.create_user("bb", "supersecret")
        await upsert_profile(conn, user_id=a.id, status_emoji="🚀")
        got = await get_profiles_for(conn, [a.id, b.id])
        assert got[a.id]["status_emoji"] == "🚀"
        assert b.id not in got  # no profile row yet → absent
    finally:
        await backend.close()


# ── presence ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_presence_transitions_and_last_seen():
    backend, conn, sm = await _env()
    try:
        u = await sm.create_user("carol", "supersecret")
        await mark_presence(conn, user_id=u.id, online=True)
        got = await get_presence_for(conn, [u.id])
        assert got[u.id]["state"] == "online"
        first_seen = got[u.id]["last_seen_at"]
        assert first_seen

        await mark_presence(conn, user_id=u.id, online=False)
        got = await get_presence_for(conn, [u.id])
        assert got[u.id]["state"] == "offline"
        assert got[u.id]["last_seen_at"]  # stamped on going offline too
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_hub_presence_sink_invoked():
    backend, conn, sm = await _env()
    try:
        u = await sm.create_user("dave", "supersecret")
        hub = ConnectHub()

        async def sink(user_id, online):
            await mark_presence(conn, user_id=user_id, online=online)

        hub.set_presence_sink(sink)

        class _WS:
            async def send_text(self, _):
                return None

        att = await hub.attach(ws=_WS(), user_id=u.id, user_did="dave@myhost.example.com")
        got = await get_presence_for(conn, [u.id])
        assert got[u.id]["state"] == "online"

        await hub.detach(att.connection_id)
        got = await get_presence_for(conn, [u.id])
        assert got[u.id]["state"] == "offline"
    finally:
        await backend.close()


# ── directory enrichment + search ─────────────────────────────────

@pytest.mark.asyncio
async def test_directory_query_includes_username_and_enriches():
    backend, conn, sm = await _env()
    try:
        caller = await sm.create_user("me", "supersecret")
        peer = await sm.create_user("alice", "supersecret", display_name="Alice")
        await _make_discoverable(conn, caller.id)
        await _make_discoverable(conn, peer.id)
        await upsert_profile(conn, user_id=peer.id, status_message="around")
        await mark_presence(conn, user_id=peer.id, online=False)

        rows = await _query_discoverable_same_instance_peers(conn, caller.id)
        assert rows and rows[0][2] == "alice"  # username present

        people = await _enrich_people(conn, ConnectHub(), rows)
        entry = people[0]
        assert entry["handle"] == "@alice"
        assert entry["display_name"] == "Alice"
        assert entry["status_message"] == "around"
        assert entry["online"] is False
        assert entry["last_seen_at"]  # offline → carries persisted last-seen
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_search_matches_username_and_display_name():
    backend, conn, sm = await _env()
    try:
        caller = await sm.create_user("me", "supersecret")
        alice = await sm.create_user("alice", "supersecret", display_name="Alice Smith")
        bob = await sm.create_user("bobby", "supersecret", display_name="Bob")
        for uid in (caller.id, alice.id, bob.id):
            await _make_discoverable(conn, uid)

        by_username = await _search_discoverable_peers(conn, caller.id, "alic")
        assert {r[2] for r in by_username} == {"alice"}

        by_display = await _search_discoverable_peers(conn, caller.id, "Smith")
        assert {r[2] for r in by_display} == {"alice"}

        none = await _search_discoverable_peers(conn, caller.id, "zzz")
        assert none == []
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_search_escapes_sql_wildcards():
    backend, conn, sm = await _env()
    try:
        caller = await sm.create_user("me", "supersecret")
        alice = await sm.create_user("alice", "supersecret")
        await _make_discoverable(conn, caller.id)
        await _make_discoverable(conn, alice.id)
        # A literal "%" must not act as a wildcard that matches everyone.
        assert await _search_discoverable_peers(conn, caller.id, "%") == []
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_everyone_visible_by_default_but_optout_hides():
    # Internal-directory model: a user appears with NO setting at all (default
    # visible); only an explicit opt-OUT removes them.
    backend, conn, sm = await _env()
    try:
        caller = await sm.create_user("me", "supersecret")
        await sm.create_user("alice", "supersecret")  # no discoverability setting
        hidden = await sm.create_user("bob", "supersecret")
        await conn.execute(
            "INSERT OR REPLACE INTO user_settings (user_id, key, value) VALUES (?, ?, ?)",
            (hidden.id, "ui.connectDiscoverableSameInstance", "false"),
        )
        await conn.commit()
        # alice (no setting) is visible; bob (explicit opt-out) is hidden.
        hits = {r[2] for r in await _search_discoverable_peers(conn, caller.id, "")}
        assert "alice" in hits
        assert "bob" not in hits
    finally:
        await backend.close()
