"""Tests — augmentum.auth.models reserved-username defense.

Covers:
  * Exact reserved names blocked (admin, system, internal-tool, etc.)
  * Case-insensitive matching
  * Reserved prefixes blocked (fabric_peer_, fabric:, usr_)
  * Normal usernames pass through unchanged
  * Edge cases (empty, whitespace, mixed case)
  * Route-level enforcement via centralised helper
  * CRUD-level enforcement raises ValueError
"""

from __future__ import annotations

import pytest

from augmentum.auth.models import (
    RESERVED_USERNAME_PREFIXES,
    RESERVED_USERNAMES,
    is_reserved_username,
)


class TestExactReservedNames:
    @pytest.mark.parametrize("name", [
        "system", "admin", "root", "superuser",
        "api", "service", "daemon", "bot",
        "internal", "internal-tool", "internal_tool", "internaltool",
        "augmentum", "becca",
        "anonymous", "guest", "nobody", "unknown",
        "demo", "test",
    ])
    def test_reserved_exact_blocks(self, name):
        assert is_reserved_username(name) is True

    @pytest.mark.parametrize("name", [
        "ADMIN", "Admin", "AdMiN",
        "SYSTEM", "System",
        "Internal-Tool", "INTERNAL_TOOL",
        "Augmentum", "BECCA",
    ])
    def test_case_insensitive_match(self, name):
        assert is_reserved_username(name) is True


class TestReservedPrefixes:
    @pytest.mark.parametrize("name", [
        "fabric_peer_abc123",
        "fabric_peer_",
        "FABRIC_PEER_xyz",
        "fabric:node1",
        "fabric:",
        "FABRIC:node",
        "usr_abc123",
        "USR_abc",
    ])
    def test_prefix_match_blocks(self, name):
        assert is_reserved_username(name) is True


class TestNormalNamesPass:
    @pytest.mark.parametrize("name", [
        "matt", "alice", "bob_smith",
        "user1", "developer_99",
        "MaTt",  # case doesn't matter for valid names either
        "a_normal_user",
        "person123",
        # Names that LOOK adjacent to reserved but aren't
        "admin_assistant",  # contains "admin" but isn't reserved exact
        "api_user",          # contains "api" but isn't reserved exact
        "system_designer",   # contains "system" but isn't exact
        "becca_fan",         # contains "becca" but isn't exact
        "augmentum_fan",     # contains "augmentum" but isn't exact
        # Prefix-adjacent
        "fabricator",        # starts with "fabric" but not the prefix
        "user_with_fabric",  # contains fabric but not at start
    ])
    def test_normal_names_allowed(self, name):
        assert is_reserved_username(name) is False


class TestEdgeCases:
    @pytest.mark.parametrize("name", ["", None, "   ", "\t\n"])
    def test_empty_or_whitespace(self, name):
        assert is_reserved_username(name) is False  # type: ignore[arg-type]

    def test_whitespace_around_reserved_name_still_blocked(self):
        # Canonical form strips whitespace, so leading/trailing space
        # around a reserved name still matches.
        assert is_reserved_username("  admin  ") is True


class TestRegistryShape:
    """Sanity checks on the constants themselves so accidental edits
    don't silently drop a guard."""

    def test_reserved_set_is_immutable(self):
        assert isinstance(RESERVED_USERNAMES, frozenset)

    def test_prefixes_is_tuple(self):
        assert isinstance(RESERVED_USERNAME_PREFIXES, tuple)

    def test_load_bearing_names_present(self):
        # If any of these get accidentally removed, future internal-tool
        # elevation footguns would silently land. Pin them.
        for required in ("admin", "system", "internal-tool", "internal_tool",
                         "api", "augmentum"):
            assert required in RESERVED_USERNAMES, f"missing {required!r}"

    def test_load_bearing_prefixes_present(self):
        # Fabric peer namespace MUST stay reserved — collisions break
        # peer auth (see session_manager.py:278).
        assert "fabric_peer_" in RESERVED_USERNAME_PREFIXES
        assert "fabric:" in RESERVED_USERNAME_PREFIXES


class TestCRUDEnforcement:
    """The session_manager CRUD methods must reject reserved names with
    ValueError. This is the backstop in case a future route handler
    forgets to call ``is_reserved_username`` up front."""

    @pytest.mark.asyncio
    async def test_create_user_rejects_reserved(self):
        from augmentum.auth.session_manager import SessionManager
        from augmentum.state.backends.sqlite import SQLiteBackend
        # SQLiteBackend(":memory:") gives us an in-memory aiosqlite
        # connection with all migrations applied. SessionManager wraps
        # the raw conn; same pattern as tests/test_auth_routes.py.
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        sm = SessionManager(backend._conn)

        with pytest.raises(ValueError, match="reserved"):
            await sm.create_user("admin", "longenoughpw")
        with pytest.raises(ValueError, match="reserved"):
            await sm.create_user("system", "longenoughpw")
        with pytest.raises(ValueError, match="reserved"):
            await sm.create_user("fabric_peer_squat", "longenoughpw")
        with pytest.raises(ValueError, match="reserved"):
            await sm.create_user("internal-tool", "longenoughpw")

        await backend.close()

    @pytest.mark.asyncio
    async def test_create_first_admin_rejects_reserved(self):
        from augmentum.auth.session_manager import SessionManager
        from augmentum.state.backends.sqlite import SQLiteBackend
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        sm = SessionManager(backend._conn)

        with pytest.raises(ValueError, match="reserved"):
            await sm.create_first_admin("admin", "longenoughpw")
        with pytest.raises(ValueError, match="reserved"):
            await sm.create_first_admin("internal-tool", "longenoughpw")

        await backend.close()

    @pytest.mark.asyncio
    async def test_normal_username_succeeds(self):
        from augmentum.auth.session_manager import SessionManager
        from augmentum.state.backends.sqlite import SQLiteBackend
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        sm = SessionManager(backend._conn)

        user = await sm.create_user("matt", "longenoughpw")
        assert user.username == "matt"

        await backend.close()
