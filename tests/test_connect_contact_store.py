"""Contact store CRUD + idempotency tests."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from augmentum.connect.contact_store import (
    add_contact,
    ensure_contact,
    get_contact,
    list_contacts,
    new_contact_id,
    remove_contact,
    set_blocked,
    set_tags,
    update_presence,
)


CONNECT_MIGRATION = Path(
    "augmentum/state/migrations/219_connect_substrate.sql"
).read_text()


ALICE = "alice"
BOB_DID = "bob@this-instance"
CHARLIE_DID = "charlie@this-instance"


@pytest.fixture
async def conn():
    async with aiosqlite.connect(":memory:") as c:
        await c.executescript(CONNECT_MIGRATION)
        await c.commit()
        yield c


# ── add + idempotency ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_contact_inserts_row(conn) -> None:
    c = await add_contact(
        conn, user_id=ALICE, peer_did=BOB_DID,
        peer_display_name="Bob",
    )
    assert c.peer_did == BOB_DID
    assert c.peer_display_name == "Bob"
    assert c.blocked is False
    assert c.tags == []


@pytest.mark.asyncio
async def test_add_contact_idempotent_on_peer_did(conn) -> None:
    """Re-add returns the existing row, doesn't overwrite display name."""

    a = await add_contact(
        conn, user_id=ALICE, peer_did=BOB_DID,
        peer_display_name="Bob",
    )
    b = await add_contact(
        conn, user_id=ALICE, peer_did=BOB_DID,
        peer_display_name="Robert",  # different display name
    )
    assert a.contact_id == b.contact_id
    assert b.peer_display_name == "Bob"  # original preserved


@pytest.mark.asyncio
async def test_ensure_contact_preserves_existing(conn) -> None:
    a = await add_contact(
        conn, user_id=ALICE, peer_did=BOB_DID,
        peer_display_name="Bob",
        discovery_source="handle_added",
    )
    b = await ensure_contact(
        conn, user_id=ALICE, peer_did=BOB_DID,
        discovery_source="implicit",  # routing-layer source
    )
    assert b.contact_id == a.contact_id
    assert b.discovery_source == "handle_added"  # not clobbered


@pytest.mark.asyncio
async def test_ensure_contact_creates_when_absent(conn) -> None:
    c = await ensure_contact(
        conn, user_id=ALICE, peer_did=BOB_DID,
        discovery_source="implicit",
    )
    assert c.peer_did == BOB_DID
    assert c.discovery_source == "implicit"


# ── list ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_excludes_blocked_by_default(conn) -> None:
    await add_contact(
        conn, user_id=ALICE, peer_did=BOB_DID,
    )
    blocked = await add_contact(
        conn, user_id=ALICE, peer_did=CHARLIE_DID,
    )
    await set_blocked(
        conn, user_id=ALICE, contact_id=blocked.contact_id, blocked=True,
    )

    visible = await list_contacts(conn, user_id=ALICE)
    assert [c.peer_did for c in visible] == [BOB_DID]

    all_ = await list_contacts(conn, user_id=ALICE, include_blocked=True)
    assert {c.peer_did for c in all_} == {BOB_DID, CHARLIE_DID}


@pytest.mark.asyncio
async def test_list_filters_by_tag(conn) -> None:
    a = await add_contact(
        conn, user_id=ALICE, peer_did=BOB_DID,
        tags=["family"],
    )
    await add_contact(
        conn, user_id=ALICE, peer_did=CHARLIE_DID,
        tags=["work"],
    )
    family = await list_contacts(conn, user_id=ALICE, tag="family")
    assert [c.contact_id for c in family] == [a.contact_id]


# ── get ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_by_peer_did(conn) -> None:
    a = await add_contact(
        conn, user_id=ALICE, peer_did=BOB_DID,
    )
    fetched = await get_contact(
        conn, user_id=ALICE, peer_did=BOB_DID,
    )
    assert fetched.contact_id == a.contact_id


@pytest.mark.asyncio
async def test_get_by_contact_id(conn) -> None:
    a = await add_contact(
        conn, user_id=ALICE, peer_did=BOB_DID,
    )
    fetched = await get_contact(
        conn, user_id=ALICE, contact_id=a.contact_id,
    )
    assert fetched.peer_did == BOB_DID


@pytest.mark.asyncio
async def test_get_returns_none_when_absent(conn) -> None:
    fetched = await get_contact(
        conn, user_id=ALICE, peer_did="ghost@here",
    )
    assert fetched is None


@pytest.mark.asyncio
async def test_get_requires_one_lookup_key(conn) -> None:
    with pytest.raises(ValueError):
        await get_contact(conn, user_id=ALICE)


# ── remove ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_remove_returns_false_when_absent(conn) -> None:
    ok = await remove_contact(
        conn, user_id=ALICE, contact_id=new_contact_id(),
    )
    assert ok is False


@pytest.mark.asyncio
async def test_remove_returns_true_when_present(conn) -> None:
    a = await add_contact(
        conn, user_id=ALICE, peer_did=BOB_DID,
    )
    ok = await remove_contact(
        conn, user_id=ALICE, contact_id=a.contact_id,
    )
    assert ok is True
    assert (
        await get_contact(conn, user_id=ALICE, peer_did=BOB_DID)
    ) is None


# ── tags ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_tags_round_trip(conn) -> None:
    a = await add_contact(
        conn, user_id=ALICE, peer_did=BOB_DID,
    )
    await set_tags(
        conn, user_id=ALICE, contact_id=a.contact_id,
        tags=["work", "team"],
    )
    fetched = await get_contact(
        conn, user_id=ALICE, contact_id=a.contact_id,
    )
    assert fetched.tags == ["work", "team"]


# ── presence ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_presence_round_trip(conn) -> None:
    await add_contact(
        conn, user_id=ALICE, peer_did=BOB_DID,
    )
    ok = await update_presence(
        conn, user_id=ALICE, peer_did=BOB_DID,
        status="online",
    )
    assert ok is True
    fetched = await get_contact(
        conn, user_id=ALICE, peer_did=BOB_DID,
    )
    assert fetched.last_seen_status == "online"
    assert fetched.last_seen_at is not None


@pytest.mark.asyncio
async def test_update_presence_validates_status(conn) -> None:
    await add_contact(conn, user_id=ALICE, peer_did=BOB_DID)
    with pytest.raises(ValueError):
        await update_presence(
            conn, user_id=ALICE, peer_did=BOB_DID, status="weirdo",
        )


# ── staleness reconciliation ────────────────────────────────────


USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  username TEXT,
  display_name TEXT,
  role TEXT
);
CREATE TABLE IF NOT EXISTS connect_guest_grants (
  grant_id TEXT PRIMARY KEY,
  user_id TEXT,
  guest_user_id TEXT,
  revoked_at TEXT
);
"""


@pytest.fixture
async def conn_with_users():
    """Contacts schema PLUS the users/grants tables staleness consults."""
    async with aiosqlite.connect(":memory:") as c:
        await c.executescript(CONNECT_MIGRATION)
        await c.executescript(USERS_TABLE)
        await c.commit()
        yield c


async def _add_user(c, uid, role="user"):
    await c.execute(
        "INSERT INTO users (id, username, display_name, role) VALUES (?,?,?,?)",
        (uid, uid, uid.title(), role),
    )
    await c.commit()


@pytest.mark.asyncio
async def test_deleted_peer_dropped_from_contacts(conn_with_users) -> None:
    """A contact whose account no longer exists is not a person you can call."""

    await _add_user(conn_with_users, "bob")
    await add_contact(conn_with_users, user_id=ALICE, peer_did=BOB_DID)
    # charlie is never inserted into users -- the orphan case seen live.
    await add_contact(conn_with_users, user_id=ALICE, peer_did=CHARLIE_DID)

    visible = await list_contacts(conn_with_users, user_id=ALICE)
    assert [c.peer_did for c in visible] == [BOB_DID]

    kept = await list_contacts(conn_with_users, user_id=ALICE, include_stale=True)
    assert {c.peer_did for c in kept} == {BOB_DID, CHARLIE_DID}


@pytest.mark.asyncio
async def test_revoked_guest_dropped_from_contacts(conn_with_users) -> None:
    """Guest account survives revocation; the contact row must not."""

    await _add_user(conn_with_users, "guest1", role="guest")
    await add_contact(conn_with_users, user_id=ALICE, peer_did="guest1@this-instance")

    await conn_with_users.execute(
        "INSERT INTO connect_guest_grants (grant_id, user_id, guest_user_id, revoked_at)"
        " VALUES (?,?,?,?)",
        ("g1", ALICE, "guest1", ""),
    )
    await conn_with_users.commit()
    assert len(await list_contacts(conn_with_users, user_id=ALICE)) == 1

    await conn_with_users.execute(
        "UPDATE connect_guest_grants SET revoked_at = ? WHERE grant_id = ?",
        ("2026-07-25 10:07:01", "g1"),
    )
    await conn_with_users.commit()
    assert await list_contacts(conn_with_users, user_id=ALICE) == []


@pytest.mark.asyncio
async def test_revocation_is_scoped_to_the_host(conn_with_users) -> None:
    """Alice revoking a guest must not remove them from Dave's contacts."""

    await _add_user(conn_with_users, "guest1", role="guest")
    await add_contact(conn_with_users, user_id=ALICE, peer_did="guest1@this-instance")
    await add_contact(conn_with_users, user_id="dave", peer_did="guest1@this-instance")
    await conn_with_users.executescript(
        "INSERT INTO connect_guest_grants VALUES ('g1','alice','guest1','2026-07-25');"
        "INSERT INTO connect_guest_grants VALUES ('g2','dave','guest1','');"
    )
    await conn_with_users.commit()

    assert await list_contacts(conn_with_users, user_id=ALICE) == []
    assert len(await list_contacts(conn_with_users, user_id="dave")) == 1


@pytest.mark.asyncio
async def test_fabric_contacts_never_judged_stale(conn_with_users) -> None:
    """Cross-instance peers have no local user row -- that is NORMAL."""

    await add_contact(
        conn_with_users, user_id=ALICE, peer_did="erin@some-other-box",
        peer_display_name="Erin",
    )
    visible = await list_contacts(conn_with_users, user_id=ALICE)
    assert [c.peer_did for c in visible] == ["erin@some-other-box"]


@pytest.mark.asyncio
async def test_missing_users_table_fails_open(conn) -> None:
    """Isolated schema (no users table) must not blank the contact list."""

    await add_contact(conn, user_id=ALICE, peer_did=BOB_DID)
    assert len(await list_contacts(conn, user_id=ALICE)) == 1
