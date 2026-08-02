"""Migration tests for the branch-tagged narrative tiers (115-119).

Verifies that:
  1. New tables (narrative_branches, narrative_state_snapshots,
     narrative_ledger_entries) exist with the expected schema after migrations.
  2. Migration 118 added branch_id to narrative_archive with default 'main'.
  3. Migration 119 backfill correctly seeds main-branch rows + copies
     state_snapshot to snapshot history + unpacks memory_ledger JSON to rows.
  4. The backfill is idempotent — re-running it does not create duplicates.
  5. Corrupt JSON in legacy columns does not abort the migration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from augmentum.state.backends.sqlite import SQLiteBackend

_MIGRATION_DIR = Path(__file__).parent.parent / "augmentum" / "state" / "migrations"


def _load_backfill_sql() -> str:
    """Load migration 119's backfill statements (excluding schema_version row).

    Tests against the actual migration content rather than a hand-copy so a
    drift in the migration is caught by these tests.
    """
    raw = (_MIGRATION_DIR / "119_narrative_branches_main_seed.sql").read_text(encoding="utf-8")
    # Drop the schema_version INSERT — it would fail if re-run via PRIMARY KEY
    out_lines: list[str] = []
    skip_block = False
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("INSERT OR IGNORE INTO schema_version"):
            skip_block = True
            continue
        if skip_block:
            if stripped.startswith("VALUES") or stripped == "" or stripped.startswith("--"):
                if stripped.endswith(";") or stripped.startswith("VALUES") and stripped.endswith(";"):
                    skip_block = False
                continue
            skip_block = False
        out_lines.append(line)
    return "\n".join(out_lines)


async def _exec_backfill(conn) -> None:
    sql = _load_backfill_sql()
    # SQLite executescript does multi-statement DDL/DML; commits implicitly.
    await conn.executescript(sql)


@pytest.fixture
async def sqlite_backend():
    """Fresh in-memory backend with all migrations (1-119) applied."""
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    yield backend
    await backend.close()


# ----------------------------------------------------------------------
# Schema presence
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_migration_115_creates_narrative_branches(sqlite_backend):
    cursor = await sqlite_backend.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='narrative_branches'"
    )
    row = await cursor.fetchone()
    assert row is not None, "narrative_branches table missing"

    cursor = await sqlite_backend.conn.execute("PRAGMA table_info(narrative_branches)")
    cols = {r[1]: r for r in await cursor.fetchall()}
    assert {"branch_id", "session_id", "parent_branch_id", "branch_point",
            "status", "user_id", "created_at", "last_visited_at"} <= set(cols)
    # status defaults to 'active'
    status_col = cols["status"]
    assert status_col[2] == "TEXT"
    assert status_col[3] == 1  # NOT NULL


@pytest.mark.asyncio
async def test_migration_116_creates_narrative_state_snapshots(sqlite_backend):
    cursor = await sqlite_backend.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='narrative_state_snapshots'"
    )
    assert (await cursor.fetchone()) is not None


@pytest.mark.asyncio
async def test_migration_117_creates_narrative_ledger_entries(sqlite_backend):
    cursor = await sqlite_backend.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='narrative_ledger_entries'"
    )
    assert (await cursor.fetchone()) is not None


@pytest.mark.asyncio
async def test_migration_118_adds_branch_id_to_archive(sqlite_backend):
    cursor = await sqlite_backend.conn.execute("PRAGMA table_info(narrative_archive)")
    cols = {r[1]: r for r in await cursor.fetchall()}
    assert "branch_id" in cols, "branch_id column missing from narrative_archive"
    branch_col = cols["branch_id"]
    assert branch_col[2] == "TEXT"
    assert branch_col[3] == 1  # NOT NULL
    # Default is 'main' — SQLite stores defaults with quotes
    assert "main" in str(branch_col[4])


@pytest.mark.asyncio
async def test_indexes_present(sqlite_backend):
    cursor = await sqlite_backend.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    )
    names = {r[0] for r in await cursor.fetchall()}
    expected = {
        "idx_narrative_branches_user",
        "idx_narrative_branches_parent",
        "idx_narrative_branches_status",
        "idx_state_snapshots_lookup",
        "idx_state_snapshots_user",
        "idx_ledger_entries_lookup",
        "idx_ledger_entries_user",
        "idx_narrative_archive_branch",
    }
    missing = expected - names
    assert not missing, f"Missing indexes: {missing}"


# ----------------------------------------------------------------------
# Backfill behavior
# ----------------------------------------------------------------------

async def _seed_session_with_state_and_ledger(
    conn,
    session_id: str = "ses_test_alpha",
    user_id: str = "user_test",
    *,
    state_snapshot: str = '{"card_type":"character","fields":{"location":"forest"}}',
    memory_ledger: str = '[{"round_num":4,"category":"discovery","content":"Found map"},'
                         '{"round_num":8,"category":"commitment","content":"Pledged help"}]',
    message_count: int = 12,
) -> None:
    # ui_sessions row first (FK target)
    await conn.execute(
        "INSERT OR IGNORE INTO ui_sessions (id, user_id, title, mode, data) "
        "VALUES (?, ?, 'test', 'narrative', '{}')",
        (session_id, user_id),
    )
    # sessions row (the one narrative FKs reference)
    await conn.execute(
        "INSERT OR IGNORE INTO sessions (id, user_id) VALUES (?, ?)",
        (session_id, user_id),
    )
    await conn.execute(
        """INSERT INTO narrative_memory
           (session_id, card_type, memory_summary, last_summary_at,
            state_snapshot, memory_ledger, branch_states, message_count, user_id)
           VALUES (?, 'character', '', 0, ?, ?, '{}', ?, ?)""",
        (session_id, state_snapshot, memory_ledger, message_count, user_id),
    )
    await conn.commit()


@pytest.mark.asyncio
async def test_backfill_seeds_main_branch(sqlite_backend):
    await _seed_session_with_state_and_ledger(sqlite_backend.conn)
    await _exec_backfill(sqlite_backend.conn)

    cursor = await sqlite_backend.conn.execute(
        "SELECT branch_id, parent_branch_id, branch_point, status FROM narrative_branches "
        "WHERE session_id = ?",
        ("ses_test_alpha",),
    )
    rows = await cursor.fetchall()
    assert len(rows) == 1
    branch_id, parent, branch_point, status = rows[0]
    assert branch_id == "main"
    assert parent is None
    assert branch_point == 0
    assert status == "active"


@pytest.mark.asyncio
async def test_backfill_copies_state_snapshot(sqlite_backend):
    await _seed_session_with_state_and_ledger(sqlite_backend.conn)
    await _exec_backfill(sqlite_backend.conn)

    cursor = await sqlite_backend.conn.execute(
        "SELECT branch_id, message_index, snapshot_data FROM narrative_state_snapshots "
        "WHERE session_id = ?",
        ("ses_test_alpha",),
    )
    rows = await cursor.fetchall()
    assert len(rows) == 1
    branch_id, msg_idx, data = rows[0]
    assert branch_id == "main"
    assert msg_idx == 12
    parsed = json.loads(data)
    assert parsed["card_type"] == "character"
    assert parsed["fields"]["location"] == "forest"


@pytest.mark.asyncio
async def test_backfill_unpacks_ledger_entries(sqlite_backend):
    await _seed_session_with_state_and_ledger(sqlite_backend.conn)
    await _exec_backfill(sqlite_backend.conn)

    cursor = await sqlite_backend.conn.execute(
        "SELECT branch_id, round_num, category, content FROM narrative_ledger_entries "
        "WHERE session_id = ? ORDER BY round_num",
        ("ses_test_alpha",),
    )
    rows = [tuple(r) for r in await cursor.fetchall()]
    assert len(rows) == 2
    assert rows[0] == ("main", 4, "discovery", "Found map")
    assert rows[1] == ("main", 8, "commitment", "Pledged help")


@pytest.mark.asyncio
async def test_backfill_idempotent_no_duplicates(sqlite_backend):
    await _seed_session_with_state_and_ledger(sqlite_backend.conn)
    await _exec_backfill(sqlite_backend.conn)
    await _exec_backfill(sqlite_backend.conn)  # second run
    await _exec_backfill(sqlite_backend.conn)  # third run

    # Branch row count: still 1
    cursor = await sqlite_backend.conn.execute(
        "SELECT COUNT(*) FROM narrative_branches WHERE session_id = ?",
        ("ses_test_alpha",),
    )
    (n_branches,) = await cursor.fetchone()
    assert n_branches == 1

    # Snapshot count: still 1
    cursor = await sqlite_backend.conn.execute(
        "SELECT COUNT(*) FROM narrative_state_snapshots WHERE session_id = ?",
        ("ses_test_alpha",),
    )
    (n_snapshots,) = await cursor.fetchone()
    assert n_snapshots == 1

    # Ledger count: still 2 (matches the 2 entries in the seed JSON)
    cursor = await sqlite_backend.conn.execute(
        "SELECT COUNT(*) FROM narrative_ledger_entries WHERE session_id = ?",
        ("ses_test_alpha",),
    )
    (n_ledger,) = await cursor.fetchone()
    assert n_ledger == 2


@pytest.mark.asyncio
async def test_backfill_skips_empty_state_snapshot(sqlite_backend):
    """Default narrative_memory.state_snapshot is '{}' — no snapshot row should be written."""
    await _seed_session_with_state_and_ledger(
        sqlite_backend.conn,
        session_id="ses_empty_state",
        state_snapshot="{}",
        memory_ledger="[]",
    )
    await _exec_backfill(sqlite_backend.conn)

    cursor = await sqlite_backend.conn.execute(
        "SELECT COUNT(*) FROM narrative_state_snapshots WHERE session_id = ?",
        ("ses_empty_state",),
    )
    (n,) = await cursor.fetchone()
    assert n == 0
    cursor = await sqlite_backend.conn.execute(
        "SELECT COUNT(*) FROM narrative_ledger_entries WHERE session_id = ?",
        ("ses_empty_state",),
    )
    (n,) = await cursor.fetchone()
    assert n == 0
    # But the main branch should still seed (every narrative_memory row gets one)
    cursor = await sqlite_backend.conn.execute(
        "SELECT COUNT(*) FROM narrative_branches WHERE session_id = ?",
        ("ses_empty_state",),
    )
    (n,) = await cursor.fetchone()
    assert n == 1


@pytest.mark.asyncio
async def test_backfill_skips_invalid_json(sqlite_backend):
    """Corrupt JSON in legacy columns must not abort the migration."""
    await _seed_session_with_state_and_ledger(
        sqlite_backend.conn,
        session_id="ses_corrupt",
        state_snapshot="{not-json",
        memory_ledger="[also-not-json",
    )
    await _seed_session_with_state_and_ledger(
        sqlite_backend.conn,
        session_id="ses_healthy",
        state_snapshot='{"card_type":"narrator","fields":{"active_quest":"rescue"}}',
        memory_ledger='[{"round_num":2,"category":"discovery","content":"x"}]',
    )

    # Backfill should succeed — corrupt session is skipped, healthy is processed
    await _exec_backfill(sqlite_backend.conn)

    cursor = await sqlite_backend.conn.execute(
        "SELECT session_id FROM narrative_state_snapshots ORDER BY session_id"
    )
    sessions_with_snapshots = [r[0] for r in await cursor.fetchall()]
    assert sessions_with_snapshots == ["ses_healthy"]

    cursor = await sqlite_backend.conn.execute(
        "SELECT session_id FROM narrative_ledger_entries ORDER BY session_id"
    )
    sessions_with_ledger = [r[0] for r in await cursor.fetchall()]
    assert sessions_with_ledger == ["ses_healthy"]

    # Both sessions still get a main branch — the seed step doesn't depend on JSON validity
    cursor = await sqlite_backend.conn.execute(
        "SELECT session_id FROM narrative_branches ORDER BY session_id"
    )
    sessions_with_branches = [r[0] for r in await cursor.fetchall()]
    assert sessions_with_branches == ["ses_corrupt", "ses_healthy"]


@pytest.mark.asyncio
async def test_backfill_skips_orphan_narrative_memory_rows(sqlite_backend):
    """REGRESSION (2026-05-06): legacy narrative_memory rows whose session_id
    has no row in the `sessions` table would cause migration 119 to abort with
    `IntegrityError: FOREIGN KEY constraint failed`, leaving schema_version at
    118 and the app unable to start. The fix added EXISTS-against-sessions
    guards. This test seeds an orphan row and verifies the migration skips it."""
    # Insert narrative_memory row WITHOUT corresponding sessions row
    await sqlite_backend.conn.execute(
        """INSERT INTO narrative_memory
           (session_id, card_type, memory_summary, last_summary_at,
            state_snapshot, memory_ledger, message_count, user_id)
           VALUES (?, 'character', '', 0, ?, ?, 5, ?)""",
        ("ses_orphan", '{"card_type":"character"}', '[]', "user_orphan"),
    )
    # Also insert a legitimate session for control
    await sqlite_backend.conn.execute(
        "INSERT INTO ui_sessions (id, user_id, title, mode, data) "
        "VALUES ('ses_real', 'user_real', 't', 'narrative', '{}')",
    )
    await sqlite_backend.conn.execute(
        "INSERT INTO sessions (id, user_id) VALUES ('ses_real', 'user_real')",
    )
    await sqlite_backend.conn.execute(
        """INSERT INTO narrative_memory
           (session_id, card_type, memory_summary, last_summary_at,
            state_snapshot, memory_ledger, message_count, user_id)
           VALUES (?, 'character', '', 0, ?, ?, 8, ?)""",
        ("ses_real", '{"card_type":"character"}',
         '[{"round_num":4,"category":"x","content":"y"}]', "user_real"),
    )
    await sqlite_backend.conn.commit()

    # Re-running the backfill must not raise (it would have prior to the fix)
    await _exec_backfill(sqlite_backend.conn)

    # Real session got seeded
    cursor = await sqlite_backend.conn.execute(
        "SELECT branch_id FROM narrative_branches WHERE session_id = 'ses_real'"
    )
    assert (await cursor.fetchone()) is not None

    # Orphan session was SKIPPED (no row created — it would have FK-violated)
    cursor = await sqlite_backend.conn.execute(
        "SELECT branch_id FROM narrative_branches WHERE session_id = 'ses_orphan'"
    )
    assert (await cursor.fetchone()) is None

    # Real session got its ledger entry too
    cursor = await sqlite_backend.conn.execute(
        "SELECT round_num FROM narrative_ledger_entries WHERE session_id = 'ses_real'"
    )
    rows = await cursor.fetchall()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_backfill_skips_ledger_entries_with_null_round_num(sqlite_backend):
    """REGRESSION: legacy memory_ledger entries missing round_num would cause
    `CAST(json_extract(...) AS INTEGER)` to return NULL, violating the
    `narrative_ledger_entries.round_num INTEGER NOT NULL` constraint."""
    # Real session
    await sqlite_backend.conn.execute(
        "INSERT INTO ui_sessions (id, user_id, title, mode, data) "
        "VALUES ('ses_x', 'user_x', 't', 'narrative', '{}')",
    )
    await sqlite_backend.conn.execute(
        "INSERT INTO sessions (id, user_id) VALUES ('ses_x', 'user_x')",
    )
    # Ledger with one valid + one missing-round-num entry
    bad_ledger = ('[{"category":"x","content":"missing-round-num"},'
                  '{"round_num":4,"category":"x","content":"valid"}]')
    await sqlite_backend.conn.execute(
        """INSERT INTO narrative_memory
           (session_id, card_type, memory_summary, last_summary_at,
            state_snapshot, memory_ledger, message_count, user_id)
           VALUES (?, 'character', '', 0, '{}', ?, 5, ?)""",
        ("ses_x", bad_ledger, "user_x"),
    )
    await sqlite_backend.conn.commit()

    await _exec_backfill(sqlite_backend.conn)

    cursor = await sqlite_backend.conn.execute(
        "SELECT round_num, content FROM narrative_ledger_entries "
        "WHERE session_id = 'ses_x'"
    )
    rows = [tuple(r) for r in await cursor.fetchall()]
    # Only the entry with a valid round_num is migrated
    assert rows == [(4, "valid")]


@pytest.mark.asyncio
async def test_archive_existing_rows_default_to_main(sqlite_backend):
    """Migration 118 added branch_id NOT NULL DEFAULT 'main'. Existing archive
    rows (none in fresh DB, but verify the default applies) plus rows inserted
    without branch_id explicitly should all read back as 'main'."""
    # Need ui_sessions and sessions row first (FK constraint)
    await sqlite_backend.conn.execute(
        "INSERT OR IGNORE INTO ui_sessions (id, user_id, title, mode, data) "
        "VALUES ('ses_legacy', 'user_x', 't', 'narrative', '{}')"
    )
    await sqlite_backend.conn.execute(
        "INSERT OR IGNORE INTO sessions (id, user_id) VALUES ('ses_legacy', 'user_x')"
    )
    await sqlite_backend.conn.execute(
        "INSERT INTO narrative_archive (id, session_id, user_content, assistant_content, "
        "summary, turn_number, user_id) "
        "VALUES ('arc_1', 'ses_legacy', 'u', 'a', 's', 5, 'user_x')"
    )
    await sqlite_backend.conn.commit()

    cursor = await sqlite_backend.conn.execute(
        "SELECT branch_id FROM narrative_archive WHERE id = 'arc_1'"
    )
    (branch_id,) = await cursor.fetchone()
    assert branch_id == "main"
