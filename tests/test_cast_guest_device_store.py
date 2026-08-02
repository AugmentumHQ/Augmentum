"""Tests for the cast couch co-op GuestDeviceStore (Phase 3).

Pins the device-fingerprint substrate that enables the welcome-back
auto-reconnect flow on the cast-guest-join page.
"""

from __future__ import annotations

import pathlib

import aiosqlite
import pytest

from augmentum.state.guest_device_store import GuestDeviceStore
from augmentum.state.guest_store import GuestStore

_USERS_SQL = """
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT
);
"""


def _load_migration(name: str) -> str:
    p = pathlib.Path("augmentum/state/migrations") / name
    return p.read_text(encoding="utf-8")


async def _mkstores():
    """Both GuestStore and GuestDeviceStore on the same conn."""
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_USERS_SQL)
    await conn.execute("INSERT INTO users (id) VALUES ('host1')")
    await conn.execute("INSERT INTO users (id) VALUES ('host2')")
    await conn.commit()
    await conn.executescript(_load_migration("229_cast_guest_profiles.sql"))
    await conn.executescript(_load_migration("230_cast_guest_devices.sql"))
    return GuestStore(conn), GuestDeviceStore(conn), conn


# ── link_device ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_link_device_creates_row():
    guests, devices, _ = await _mkstores()
    alice = await guests.create_profile(
        host_user_id="host1", display_name="alice",
    )
    record = await devices.link_device(
        guest_profile_id=alice["id"], host_user_id="host1",
        device_uuid="dev-uuid-abc", ua_hash="ua-hash-1",
    )
    assert record["id"].startswith("gd_")
    assert record["guest_profile_id"] == alice["id"]
    assert record["device_uuid"] == "dev-uuid-abc"


@pytest.mark.asyncio
async def test_link_device_rejects_missing_inputs():
    _, devices, _ = await _mkstores()
    with pytest.raises(ValueError):
        await devices.link_device(
            guest_profile_id="", host_user_id="host1",
            device_uuid="dev-uuid",
        )
    with pytest.raises(ValueError):
        await devices.link_device(
            guest_profile_id="gp_x", host_user_id="",
            device_uuid="dev-uuid",
        )
    with pytest.raises(ValueError):
        await devices.link_device(
            guest_profile_id="gp_x", host_user_id="host1",
            device_uuid="",
        )


@pytest.mark.asyncio
async def test_link_device_rebinds_on_duplicate_uuid():
    """When the same device_uuid links a second time, the row is
    rebound to the new profile_id (the "not me?" → pick-different
    flow).
    """
    guests, devices, _ = await _mkstores()
    alice = await guests.create_profile(
        host_user_id="host1", display_name="alice",
    )
    bob = await guests.create_profile(
        host_user_id="host1", display_name="bob",
    )
    first = await devices.link_device(
        guest_profile_id=alice["id"], host_user_id="host1",
        device_uuid="dev-uuid-shared",
    )
    second = await devices.link_device(
        guest_profile_id=bob["id"], host_user_id="host1",
        device_uuid="dev-uuid-shared",
    )
    # Same row, new profile.
    assert first["id"] == second["id"]
    assert second["guest_profile_id"] == bob["id"]


# ── match ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_match_returns_profile_for_known_uuid():
    guests, devices, _ = await _mkstores()
    alice = await guests.create_profile(
        host_user_id="host1", display_name="alice",
        color="#4ade80",
    )
    await devices.link_device(
        guest_profile_id=alice["id"], host_user_id="host1",
        device_uuid="dev-abc",
    )
    found = await devices.match(
        host_user_id="host1", device_uuid="dev-abc",
    )
    assert found is not None
    assert found["id"] == alice["id"]
    assert found["display_name"] == "alice"
    assert found["color"] == "#4ade80"


@pytest.mark.asyncio
async def test_match_unknown_uuid_returns_none():
    _, devices, _ = await _mkstores()
    found = await devices.match(
        host_user_id="host1", device_uuid="dev-never-linked",
    )
    assert found is None


@pytest.mark.asyncio
async def test_match_scoped_per_host():
    """device_uuid is unique per host, not globally. Same uuid linked
    at host1 and host2 returns different profiles for different
    host_user_id arguments.
    """
    guests, devices, _ = await _mkstores()
    alice1 = await guests.create_profile(
        host_user_id="host1", display_name="alice",
    )
    alice2 = await guests.create_profile(
        host_user_id="host2", display_name="alice",
    )
    await devices.link_device(
        guest_profile_id=alice1["id"], host_user_id="host1",
        device_uuid="dev-shared",
    )
    await devices.link_device(
        guest_profile_id=alice2["id"], host_user_id="host2",
        device_uuid="dev-shared",
    )
    f1 = await devices.match(
        host_user_id="host1", device_uuid="dev-shared",
    )
    f2 = await devices.match(
        host_user_id="host2", device_uuid="dev-shared",
    )
    assert f1["id"] == alice1["id"]
    assert f2["id"] == alice2["id"]


@pytest.mark.asyncio
async def test_match_ua_hash_mismatch_returns_none():
    """Stored ua_hash != provided ua_hash → no match. Guest falls
    back to the name picker; they can re-link by picking themselves.
    """
    guests, devices, _ = await _mkstores()
    alice = await guests.create_profile(
        host_user_id="host1", display_name="alice",
    )
    await devices.link_device(
        guest_profile_id=alice["id"], host_user_id="host1",
        device_uuid="dev-abc", ua_hash="original-ua",
    )
    found = await devices.match(
        host_user_id="host1", device_uuid="dev-abc",
        ua_hash="different-ua",
    )
    assert found is None


@pytest.mark.asyncio
async def test_match_ua_hash_omitted_is_lenient():
    """When the caller doesn't provide ua_hash (Phase 2 path),
    UA isn't checked — the device_uuid alone is enough.
    """
    guests, devices, _ = await _mkstores()
    alice = await guests.create_profile(
        host_user_id="host1", display_name="alice",
    )
    await devices.link_device(
        guest_profile_id=alice["id"], host_user_id="host1",
        device_uuid="dev-abc", ua_hash="some-ua",
    )
    found = await devices.match(
        host_user_id="host1", device_uuid="dev-abc",
    )
    assert found is not None


# ── forget ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_forget_drops_row():
    guests, devices, _ = await _mkstores()
    alice = await guests.create_profile(
        host_user_id="host1", display_name="alice",
    )
    await devices.link_device(
        guest_profile_id=alice["id"], host_user_id="host1",
        device_uuid="dev-abc",
    )
    assert await devices.forget(
        host_user_id="host1", device_uuid="dev-abc",
    ) is True
    assert await devices.match(
        host_user_id="host1", device_uuid="dev-abc",
    ) is None


@pytest.mark.asyncio
async def test_forget_preserves_profile():
    """Forget-device drops the link, NOT the profile. Alice's host
    can still see her in their guest roster.
    """
    guests, devices, _ = await _mkstores()
    alice = await guests.create_profile(
        host_user_id="host1", display_name="alice",
    )
    await devices.link_device(
        guest_profile_id=alice["id"], host_user_id="host1",
        device_uuid="dev-abc",
    )
    await devices.forget(host_user_id="host1", device_uuid="dev-abc")
    fetched = await guests.get(alice["id"], host_user_id="host1")
    assert fetched is not None


@pytest.mark.asyncio
async def test_forget_unknown_returns_false():
    _, devices, _ = await _mkstores()
    assert await devices.forget(
        host_user_id="host1", device_uuid="never-linked",
    ) is False


# ── list_for_profile ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_for_profile_returns_all_devices():
    """One profile can have multiple devices (phone + tablet).
    Phase 4 host-side "Manage guests" UI will use this.
    """
    guests, devices, _ = await _mkstores()
    alice = await guests.create_profile(
        host_user_id="host1", display_name="alice",
    )
    await devices.link_device(
        guest_profile_id=alice["id"], host_user_id="host1",
        device_uuid="phone-uuid",
    )
    await devices.link_device(
        guest_profile_id=alice["id"], host_user_id="host1",
        device_uuid="tablet-uuid",
    )
    listed = await devices.list_for_profile(
        guest_profile_id=alice["id"], host_user_id="host1",
    )
    assert len(listed) == 2
    uuids = {d["device_uuid"] for d in listed}
    assert uuids == {"phone-uuid", "tablet-uuid"}


@pytest.mark.asyncio
async def test_schema_declares_cascade():
    """guest_profiles row delete must cascade to its devices —
    when host deletes alice's profile, her device links go too.
    """
    sql = _load_migration("230_cast_guest_devices.sql").lower()
    assert "references guest_profiles(id) on delete cascade" in sql
    assert "references users(id) on delete cascade" in sql
