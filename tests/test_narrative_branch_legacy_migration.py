"""Tests for scripts/migrate_narrative_branches.py — Stage 2 legacy unpack.

The script unpacks alternate-branch entries from ``narrative_memory.branch_states``
JSON blobs into properly-tagged rows in the new tables. Idempotent via
per-session marker rows in app_settings.

Critical scenarios:
  - Sessions with no alternate branches: no-op
  - Sessions with one alternate branch: branch row + snapshot + ledger entries
  - Sessions with multiple alternate branches: each gets its own rows
  - Idempotency: second run is a no-op (marker check)
  - Corrupt JSON: per-session error logged, other sessions still migrate
  - 'main' entries inside branch_states: skipped (already seeded by migration 119)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import aiosqlite
import pytest

from augmentum.state.backends.sqlite import SQLiteBackend


# Import the migration script module (scripts/ has no __init__.py)
_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "migrate_narrative_branches.py"
_spec = importlib.util.spec_from_file_location("migrate_narrative_branches", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["migrate_narrative_branches"] = _mod
_spec.loader.exec_module(_mod)


@pytest.fixture
async def backend():
    be = SQLiteBackend(":memory:")
    await be.connect()
    yield be
    await be.close()


async def _seed_session_with_branch_states(
    conn,
    session_id: str,
    user_id: str,
    branch_states_data: dict,
) -> None:
    """Seed narrative_memory with a branch_states JSON blob and required parents."""
    await conn.execute(
        "INSERT OR IGNORE INTO ui_sessions (id, user_id, title, mode, data) "
        "VALUES (?, ?, 't', 'narrative', '{}')",
        (session_id, user_id),
    )
    await conn.execute(
        "INSERT OR IGNORE INTO sessions (id, user_id) VALUES (?, ?)",
        (session_id, user_id),
    )
    await conn.execute(
        """INSERT INTO narrative_memory
           (session_id, card_type, memory_summary, last_summary_at,
            state_snapshot, memory_ledger, branch_states, message_count, user_id)
           VALUES (?, 'character', '', 0, '{}', '[]', ?, 0, ?)""",
        (session_id, json.dumps(branch_states_data), user_id),
    )
    # Also seed the 'main' branch row that migration 119 would have created
    await conn.execute(
        """INSERT OR IGNORE INTO narrative_branches
           (branch_id, session_id, parent_branch_id, branch_point, status, user_id)
           VALUES ('main', ?, NULL, 0, 'active', ?)""",
        (session_id, user_id),
    )
    await conn.commit()


# ===========================================================================
# Single-session migration
# ===========================================================================

class TestMigrateSession:
    @pytest.mark.asyncio
    async def test_unpacks_one_alternate_branch(self, backend):
        bs = {
            "branch_alpha": {
                "message_count": 12,
                "state_snapshot": {"card_type": "character",
                                   "fields": {"location": "alpha-place"}},
                "memory_ledger": [
                    {"round_num": 4, "category": "discovery", "content": "found alpha"},
                    {"round_num": 8, "category": "commitment", "content": "alpha pact"},
                ],
            },
        }
        await _seed_session_with_branch_states(
            backend.conn, "ses_a", "user_a", bs,
        )

        report = await _mod.migrate_session(
            backend.conn, "ses_a", "user_a", json.dumps(bs),
        )
        assert report["branches_inserted"] == 1
        assert report["snapshots_inserted"] == 1
        assert report["ledger_entries_inserted"] == 2
        assert report["errors"] == 0

        # Verify on disk
        cursor = await backend.conn.execute(
            "SELECT branch_id, parent_branch_id, branch_point FROM narrative_branches "
            "WHERE session_id = 'ses_a' AND branch_id = 'branch_alpha'"
        )
        row = await cursor.fetchone()
        assert tuple(row) == ("branch_alpha", "main", 0)

        cursor = await backend.conn.execute(
            "SELECT message_index FROM narrative_state_snapshots "
            "WHERE session_id = 'ses_a' AND branch_id = 'branch_alpha'"
        )
        rows = [tuple(r) for r in await cursor.fetchall()]
        assert rows == [(12,)]

        cursor = await backend.conn.execute(
            "SELECT round_num, category, content FROM narrative_ledger_entries "
            "WHERE session_id = 'ses_a' AND branch_id = 'branch_alpha' ORDER BY round_num"
        )
        rows = [tuple(r) for r in await cursor.fetchall()]
        assert rows == [
            (4, "discovery", "found alpha"),
            (8, "commitment", "alpha pact"),
        ]

    @pytest.mark.asyncio
    async def test_unpacks_multiple_branches(self, backend):
        bs = {
            "br_x": {
                "message_count": 10,
                "state_snapshot": {"fields": {"location": "x"}},
                "memory_ledger": [{"round_num": 3, "category": "x", "content": "x1"}],
            },
            "br_y": {
                "message_count": 18,
                "state_snapshot": {"fields": {"location": "y"}},
                "memory_ledger": [
                    {"round_num": 5, "category": "y", "content": "y1"},
                    {"round_num": 9, "category": "y", "content": "y2"},
                ],
            },
            "br_z": {
                "message_count": 4,
                "state_snapshot": None,  # no snapshot
                "memory_ledger": [],     # no ledger
            },
        }
        await _seed_session_with_branch_states(backend.conn, "ses_m", "user_m", bs)

        report = await _mod.migrate_session(
            backend.conn, "ses_m", "user_m", json.dumps(bs),
        )
        assert report["branches_inserted"] == 3
        assert report["snapshots_inserted"] == 2  # br_z had None
        assert report["ledger_entries_inserted"] == 3  # 1 + 2 + 0

    @pytest.mark.asyncio
    async def test_skips_main_inside_branch_states(self, backend):
        """Some legacy data may include a 'main' entry in branch_states (the
        engine snapshotted main when switching away). Migration 119 already
        seeded the main row — Stage 2 must not re-INSERT or duplicate."""
        bs = {
            "main": {
                "message_count": 50,
                "state_snapshot": {"fields": {"location": "main-snap"}},
                "memory_ledger": [{"round_num": 4, "category": "x", "content": "main-row"}],
            },
            "alt": {
                "message_count": 30,
                "state_snapshot": {"fields": {"location": "alt"}},
                "memory_ledger": [],
            },
        }
        await _seed_session_with_branch_states(backend.conn, "ses_mx", "user_mx", bs)

        report = await _mod.migrate_session(
            backend.conn, "ses_mx", "user_mx", json.dumps(bs),
        )
        assert report["branches_skipped_main"] == 1
        assert report["branches_inserted"] == 1  # only 'alt'

        # 'main' branch already exists from seed — should not have duplicate snapshot
        cursor = await backend.conn.execute(
            "SELECT COUNT(*) FROM narrative_state_snapshots "
            "WHERE session_id = 'ses_mx' AND branch_id = 'main'"
        )
        (n,) = await cursor.fetchone()
        assert n == 0  # main snapshot is migration 119's job, skipped here

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self, backend):
        await _seed_session_with_branch_states(
            backend.conn, "ses_bad", "user_bad", {"x": {}},
        )
        report = await _mod.migrate_session(
            backend.conn, "ses_bad", "user_bad", "{not json",
        )
        assert report["errors"] == 1
        assert report["branches_inserted"] == 0

    @pytest.mark.asyncio
    async def test_per_branch_invalid_payload_skipped(self, backend):
        """If one branch's saved data is corrupt (string instead of dict), it's
        skipped without aborting the whole session."""
        bs = {
            "good": {
                "message_count": 5,
                "state_snapshot": {"fields": {"location": "g"}},
                "memory_ledger": [],
            },
            "corrupt": "not a dict",  # legacy data malformed
        }
        await _seed_session_with_branch_states(backend.conn, "ses_p", "user_p", bs)
        report = await _mod.migrate_session(
            backend.conn, "ses_p", "user_p", json.dumps(bs),
        )
        assert report["branches_inserted"] == 1
        assert report["branches_skipped_invalid"] == 1


# ===========================================================================
# Top-level run_migration with markers
# ===========================================================================

class TestRunMigrationFlow:
    @pytest.mark.asyncio
    async def test_marker_set_after_success(self, backend):
        bs = {"alpha": {"message_count": 5,
                        "state_snapshot": {"fields": {"x": 1}},
                        "memory_ledger": []}}
        await _seed_session_with_branch_states(backend.conn, "ses_m1", "user_m1", bs)

        report = await _mod.migrate_session(
            backend.conn, "ses_m1", "user_m1", json.dumps(bs),
        )
        assert report["errors"] == 0
        await _mod.set_marker(backend.conn, "ses_m1", {"branches": 1})

        assert (await _mod.marker_exists(backend.conn, "ses_m1")) is True
        assert (await _mod.marker_exists(backend.conn, "ses_unrun")) is False

    @pytest.mark.asyncio
    async def test_iterate_skips_empty_blobs(self, backend):
        # Session with empty branch_states
        await backend.conn.execute(
            "INSERT INTO ui_sessions (id, user_id, title, mode, data) "
            "VALUES ('ses_empty', 'u', 't', 'narrative', '{}')",
        )
        await backend.conn.execute(
            "INSERT INTO sessions (id, user_id) VALUES ('ses_empty', 'u')",
        )
        await backend.conn.execute(
            """INSERT INTO narrative_memory
               (session_id, card_type, memory_summary, last_summary_at,
                state_snapshot, memory_ledger, branch_states, message_count, user_id)
               VALUES ('ses_empty', 'character', '', 0, '{}', '[]', '{}', 0, 'u')""",
        )
        # Session with non-empty branch_states
        await _seed_session_with_branch_states(
            backend.conn, "ses_full", "u",
            {"alt": {"message_count": 2,
                     "state_snapshot": None, "memory_ledger": []}},
        )

        seen = []
        async for s in _mod.iterate_sessions_with_branch_states(backend.conn):
            seen.append(s[0])
        assert seen == ["ses_full"]


# ===========================================================================
# Idempotency — the key safety property
# ===========================================================================

class TestIdempotency:
    @pytest.mark.asyncio
    async def test_re_running_migrate_session_does_not_duplicate(self, backend):
        bs = {"alpha": {
            "message_count": 5,
            "state_snapshot": {"fields": {"x": 1}},
            "memory_ledger": [{"round_num": 3, "category": "x", "content": "y"}],
        }}
        await _seed_session_with_branch_states(backend.conn, "ses_idem", "u", bs)

        # First run: inserts everything
        r1 = await _mod.migrate_session(
            backend.conn, "ses_idem", "u", json.dumps(bs),
        )
        assert r1["branches_inserted"] == 1

        # Second run: branch already exists, should skip
        r2 = await _mod.migrate_session(
            backend.conn, "ses_idem", "u", json.dumps(bs),
        )
        assert r2["branches_inserted"] == 0
        assert r2["branches_skipped_existing"] == 1

        # No duplicate ledger or snapshot rows
        cursor = await backend.conn.execute(
            "SELECT COUNT(*) FROM narrative_ledger_entries "
            "WHERE session_id = 'ses_idem' AND branch_id = 'alpha'"
        )
        (n,) = await cursor.fetchone()
        assert n == 1

        cursor = await backend.conn.execute(
            "SELECT COUNT(*) FROM narrative_state_snapshots "
            "WHERE session_id = 'ses_idem' AND branch_id = 'alpha'"
        )
        (n,) = await cursor.fetchone()
        assert n == 1
