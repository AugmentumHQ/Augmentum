"""Directory endpoint — auto-discovered mutual-consent peer list.

The endpoint implements the design spec's "mutual enablement as
consent" model: two users on the same instance see each other when
BOTH have ``connect_enabled = true`` AND
``connect_discoverable_same_instance = true``.

Tests cover:
* 503 when the substrate is disabled (parity with the rest of the
  routes — verified centrally in test_connect.py, just smoke here)
* Both opted-in → both visible
* One-sided opt-in → empty
* Inactive users excluded
* Self excluded
* The caller's own discoverability is echoed so the UI can render
  a "Be discoverable" affordance when they're invisible
* The ``connect_enabled`` default-on means absent rows still count
  as opted in (matches the post-2026-06-02 default flip)
"""

from __future__ import annotations

import asyncio

import pytest

from augmentum.config import settings


@pytest.fixture(autouse=True)
def _enable_connect():
    """Force connect_enabled=True for the duration of each test."""
    orig = settings.connect_enabled
    object.__setattr__(settings, "connect_enabled", True)
    yield
    object.__setattr__(settings, "connect_enabled", orig)


async def _seed_user(
    conn,
    *,
    user_id: str,
    username: str,
    display_name: str = "",
    is_active: bool = True,
) -> None:
    """Insert a user into the users table."""
    await conn.execute(
        """INSERT OR IGNORE INTO users
               (id, username, display_name, password_hash, role, is_active)
             VALUES (?, ?, ?, ?, 'user', ?)""",
        (user_id, username, display_name, "hash", 1 if is_active else 0),
    )
    await conn.commit()


async def _set_user_setting(
    conn, *, user_id: str, key: str, value: str,
) -> None:
    """Upsert a user_settings row."""
    await conn.execute(
        """INSERT INTO user_settings (user_id, key, value)
             VALUES (?, ?, ?)
             ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value""",
        (user_id, key, value),
    )
    await conn.commit()


def _conn_from_client(sqlite_client) -> object:
    """Pluck the aiosqlite conn off the app state."""
    sm = sqlite_client.app.state.state_manager
    return sm.backend.conn


# ── Happy path: mutual opt-in ──────────────────────────────────


class TestMutualOptIn:
    def test_both_optin_caller_sees_target(self, sqlite_client) -> None:
        conn = _conn_from_client(sqlite_client)

        async def setup():
            # Caller (the conftest test_user) opts into discoverability.
            await _set_user_setting(conn, user_id="usr_test",
                                    key="ui.connectEnabled", value="true")
            await _set_user_setting(
                conn, user_id="usr_test",
                key="ui.connectDiscoverableSameInstance", value="true",
            )
            # Target seed user.
            await _seed_user(conn, user_id="usr_alice", username="alice",
                             display_name="Alice")
            await _set_user_setting(conn, user_id="usr_alice",
                                    key="ui.connectEnabled", value="true")
            await _set_user_setting(
                conn, user_id="usr_alice",
                key="ui.connectDiscoverableSameInstance", value="true",
            )

        asyncio.get_event_loop().run_until_complete(setup())

        resp = sqlite_client.get("/api/connect/directory")
        assert resp.status_code == 200
        body = resp.json()
        people = body["people"]
        assert len(people) == 1
        p = people[0]
        assert p["user_id"] == "usr_alice"
        assert p["peer_did"] == "usr_alice@this-instance"
        assert p["display_name"] == "Alice"
        assert p["discovery_source"] == "same_instance"
        assert body["self_discoverable_same_instance"] is True

    def test_target_optin_caller_not_returns_empty(self, sqlite_client) -> None:
        """Caller hasn't opted into discoverability → mutual model
        excludes targets even if they're discoverable. Otherwise we'd
        leak presence: 'I can see you but you can't see me'."""
        conn = _conn_from_client(sqlite_client)

        async def setup():
            # Caller did NOT enable discoverability.
            await _seed_user(conn, user_id="usr_alice", username="alice")
            await _set_user_setting(conn, user_id="usr_alice",
                                    key="ui.connectEnabled", value="true")
            await _set_user_setting(
                conn, user_id="usr_alice",
                key="ui.connectDiscoverableSameInstance", value="true",
            )

        asyncio.get_event_loop().run_until_complete(setup())

        resp = sqlite_client.get("/api/connect/directory")
        body = resp.json()
        assert body["people"] == []
        assert body["self_discoverable_same_instance"] is False

    def test_caller_optin_target_not_target_excluded(self, sqlite_client) -> None:
        """Caller opts in, target doesn't → target excluded from list."""
        conn = _conn_from_client(sqlite_client)

        async def setup():
            await _set_user_setting(conn, user_id="usr_test",
                                    key="ui.connectEnabled", value="true")
            await _set_user_setting(
                conn, user_id="usr_test",
                key="ui.connectDiscoverableSameInstance", value="true",
            )
            # Target is "Connect on" by default but discoverability is off.
            await _seed_user(conn, user_id="usr_alice", username="alice")
            await _set_user_setting(conn, user_id="usr_alice",
                                    key="ui.connectEnabled", value="true")
            # Note: NO connect_discoverable_same_instance row → off.

        asyncio.get_event_loop().run_until_complete(setup())
        resp = sqlite_client.get("/api/connect/directory")
        body = resp.json()
        assert body["people"] == []


class TestExclusions:
    def test_self_excluded(self, sqlite_client) -> None:
        """The caller's own row never appears in the directory."""
        conn = _conn_from_client(sqlite_client)

        async def setup():
            # Caller seeds + opts into discoverability.
            await _seed_user(conn, user_id="usr_test", username="tester")
            await _set_user_setting(conn, user_id="usr_test",
                                    key="ui.connectEnabled", value="true")
            await _set_user_setting(
                conn, user_id="usr_test",
                key="ui.connectDiscoverableSameInstance", value="true",
            )

        asyncio.get_event_loop().run_until_complete(setup())
        resp = sqlite_client.get("/api/connect/directory")
        body = resp.json()
        assert all(p["user_id"] != "usr_test" for p in body["people"])

    def test_inactive_user_excluded(self, sqlite_client) -> None:
        """Inactive accounts (is_active=0) don't appear even if their
        settings are opted in — they can't be reached anyway."""
        conn = _conn_from_client(sqlite_client)

        async def setup():
            await _set_user_setting(conn, user_id="usr_test",
                                    key="ui.connectEnabled", value="true")
            await _set_user_setting(
                conn, user_id="usr_test",
                key="ui.connectDiscoverableSameInstance", value="true",
            )
            await _seed_user(conn, user_id="usr_ghost", username="ghost",
                             is_active=False)
            await _set_user_setting(conn, user_id="usr_ghost",
                                    key="ui.connectEnabled", value="true")
            await _set_user_setting(
                conn, user_id="usr_ghost",
                key="ui.connectDiscoverableSameInstance", value="true",
            )

        asyncio.get_event_loop().run_until_complete(setup())
        resp = sqlite_client.get("/api/connect/directory")
        body = resp.json()
        assert all(p["user_id"] != "usr_ghost" for p in body["people"])

    def test_connect_disabled_target_excluded(self, sqlite_client) -> None:
        """A target who explicitly set connect_enabled=false isn't
        visible even if they kept discoverability on (turning off the
        substrate is the master switch)."""
        conn = _conn_from_client(sqlite_client)

        async def setup():
            await _set_user_setting(conn, user_id="usr_test",
                                    key="ui.connectEnabled", value="true")
            await _set_user_setting(
                conn, user_id="usr_test",
                key="ui.connectDiscoverableSameInstance", value="true",
            )
            await _seed_user(conn, user_id="usr_bob", username="bob")
            await _set_user_setting(conn, user_id="usr_bob",
                                    key="ui.connectEnabled", value="false")
            await _set_user_setting(
                conn, user_id="usr_bob",
                key="ui.connectDiscoverableSameInstance", value="true",
            )

        asyncio.get_event_loop().run_until_complete(setup())
        resp = sqlite_client.get("/api/connect/directory")
        body = resp.json()
        assert all(p["user_id"] != "usr_bob" for p in body["people"])

    def test_fabric_machine_accounts_excluded(self, sqlite_client) -> None:
        """``fabric:<node-id>`` rows are MACHINES, not people.

        SessionManager provisions one per peer instance that dispatches to
        us so peer-owned data has an owner at the trust boundary. They were
        landing in the human directory: on the dogfood box 7 of 11 listed
        "people" were fabric rows, five sharing the identical display name
        "Fabric peer loopback".

        Both a role-carrying row and a legacy ``role='user'`` row are seeded,
        because the ``role='peer'`` convention post-dates the earliest peer
        rows -- a role-only filter still leaked those, so the id prefix is
        the load-bearing half of the check.
        """
        conn = _conn_from_client(sqlite_client)

        async def setup():
            await _set_user_setting(conn, user_id="usr_test",
                                    key="ui.connectEnabled", value="true")
            await _set_user_setting(
                conn, user_id="usr_test",
                key="ui.connectDiscoverableSameInstance", value="true",
            )
            # Modern peer row: role='peer'.
            await conn.execute(
                """INSERT OR IGNORE INTO users
                       (id, username, display_name, password_hash, role, is_active)
                     VALUES (?, ?, ?, ?, 'peer', 1)""",
                ("fabric:abc123", "fabric_peer_abc123", "Fabric peer abc123", "h"),
            )
            # Legacy peer row predating the role convention: role='user'.
            await _seed_user(conn, user_id="fabric:legacy1",
                             username="fabric_peer_legacy1",
                             display_name="Fabric peer legacy1")
            # A real person must still be listed -- the filter has to be
            # narrow, not a blanket "hide anything unfamiliar".
            await _seed_user(conn, user_id="usr_realperson",
                             username="realperson", display_name="Real Person")

        asyncio.get_event_loop().run_until_complete(setup())
        resp = sqlite_client.get("/api/connect/directory")
        ids = {p["user_id"] for p in resp.json()["people"]}
        assert "fabric:abc123" not in ids
        assert "fabric:legacy1" not in ids
        assert "usr_realperson" in ids

    def test_fabric_machine_accounts_excluded_from_search(
        self, sqlite_client,
    ) -> None:
        """Search must agree with browse on who is a person.

        Separate SQL, same rule -- without this the fabric rows filtered out
        of the list reappear the moment the user types "fabric".
        """
        conn = _conn_from_client(sqlite_client)

        async def setup():
            await conn.execute(
                """INSERT OR IGNORE INTO users
                       (id, username, display_name, password_hash, role, is_active)
                     VALUES (?, ?, ?, ?, 'peer', 1)""",
                ("fabric:srch01", "fabric_peer_srch01", "Fabric peer srch01", "h"),
            )
            await conn.commit()

        asyncio.get_event_loop().run_until_complete(setup())
        resp = sqlite_client.get("/api/connect/search?q=fabric")
        assert resp.status_code == 200
        people = resp.json().get("people", resp.json().get("results", []))
        assert all(
            not str(p.get("user_id", p.get("peer_did", ""))).startswith("fabric:")
            for p in people
        )


class TestConnectEnabledDefault:
    def test_absent_connect_enabled_treated_as_on(self, sqlite_client) -> None:
        """post-2026-06-02 default flip: connect_enabled defaults to true,
        so a user who never explicitly saved settings still counts as
        opted in. The discoverability flag must still be explicit
        (privacy default), but the absent connect_enabled row should
        not block them.
        """
        conn = _conn_from_client(sqlite_client)

        async def setup():
            await _set_user_setting(conn, user_id="usr_test",
                                    key="ui.connectEnabled", value="true")
            await _set_user_setting(
                conn, user_id="usr_test",
                key="ui.connectDiscoverableSameInstance", value="true",
            )
            await _seed_user(conn, user_id="usr_carl", username="carl",
                             display_name="Carl")
            # No connect_enabled row at all for Carl — relies on default.
            await _set_user_setting(
                conn, user_id="usr_carl",
                key="ui.connectDiscoverableSameInstance", value="true",
            )

        asyncio.get_event_loop().run_until_complete(setup())
        resp = sqlite_client.get("/api/connect/directory")
        body = resp.json()
        assert any(p["user_id"] == "usr_carl" for p in body["people"])


class TestSelfDiscoverabilityEcho:
    def test_returns_caller_discoverability_flags(self, sqlite_client) -> None:
        """The response carries the caller's own discoverability so
        the UI can render a 'Turn on to be visible' affordance."""
        conn = _conn_from_client(sqlite_client)

        async def setup():
            # Caller has neither flag set.
            pass

        asyncio.get_event_loop().run_until_complete(setup())
        resp = sqlite_client.get("/api/connect/directory")
        body = resp.json()
        assert body["self_discoverable_same_instance"] is False
        assert body["self_discoverable_fabric_peers"] is False

    def test_flips_true_when_user_saves(self, sqlite_client) -> None:
        conn = _conn_from_client(sqlite_client)

        async def setup():
            await _set_user_setting(
                conn, user_id="usr_test",
                key="ui.connectDiscoverableSameInstance", value="true",
            )

        asyncio.get_event_loop().run_until_complete(setup())
        resp = sqlite_client.get("/api/connect/directory")
        body = resp.json()
        assert body["self_discoverable_same_instance"] is True
