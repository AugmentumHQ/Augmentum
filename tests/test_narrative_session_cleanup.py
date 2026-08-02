"""Tests for the canonical narrative-session cleanup module.

Critical regressions covered:
  - test_purge_clears_vec_orphans  — closes the pre-existing leak where
    DELETE on narrative_archive left orphan rows in narrative_archive_vec
    accumulating ~3KB each forever per deleted chat.
  - test_purge_idempotent  — second call returns zeros, no errors.
  - test_purge_transactional_rolls_back_on_inject_fault  — inject a fault
    mid-purge; verify NO rows were deleted (atomicity).
  - test_residue_detection_logs_warning  — leave a forged row past the
    purge; verify the verification SELECT catches it.
  - test_purge_isolated_per_session  — session A's data is preserved when
    session B is purged.
  - test_purge_isolated_per_user  — user X's data is preserved when user
    Y purges the same session_id.
"""

from __future__ import annotations

import struct

import pytest

from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.narrative_cleanup import (
    NARRATIVE_TABLES,
    purge_narrative_session,
)
from augmentum.state.narrative_persistence import NarrativePersistence


def _stub_768d(seed: float) -> bytes:
    return struct.pack(f"<{768}f", *[seed * (i / 768.0) for i in range(768)])


@pytest.fixture
async def backend():
    be = SQLiteBackend(":memory:")
    await be.connect()
    yield be
    await be.close()


@pytest.fixture
async def persist(backend):
    return NarrativePersistence(backend.conn)


async def _seed_full_narrative_session(
    backend, persist,
    session_id: str = "ses_full",
    user_id: str = "user_full",
    *,
    branches: tuple[str, ...] = ("main", "B"),
    rows_per_branch: int = 2,
):
    """Seed a session with rows in every narrative tier (main + B branch)."""
    await backend.conn.execute(
        "INSERT OR IGNORE INTO ui_sessions (id, user_id, title, mode, data) "
        "VALUES (?, ?, 't', 'narrative', '{}')",
        (session_id, user_id),
    )
    await backend.conn.execute(
        "INSERT OR IGNORE INTO sessions (id, user_id) VALUES (?, ?)",
        (session_id, user_id),
    )
    # narrative_memory row
    await backend.conn.execute(
        """INSERT OR IGNORE INTO narrative_memory
           (session_id, card_type, memory_summary, last_summary_at, user_id)
           VALUES (?, 'character', '', 0, ?)""",
        (session_id, user_id),
    )
    await backend.conn.commit()
    # Create branches
    for i, b in enumerate(branches):
        parent = None if b == "main" else "main"
        bp = 0 if b == "main" else 5 * (i + 1)
        await persist.upsert_branch(session_id, b, parent, bp, user_id=user_id)
    # Add ledger / snapshot / archive rows on each branch
    for b in branches:
        await persist.store_ledger_entries(
            session_id, b,
            [{"round_num": j, "category": "x", "content": f"{b}-{j}"}
             for j in range(rows_per_branch)],
            user_id=user_id,
        )
        await persist.store_state_snapshot(
            session_id, b, rows_per_branch * 2,
            {"fields": {"location": f"{b}-place"}},
            user_id=user_id,
        )
        await persist.store_archive_exchanges_for_branch(
            session_id,
            [{"id": f"{session_id}_{b}_arc_{j}",
              "user_content": "u", "assistant_content": "a",
              "summary": f"{b}-arc-{j}", "turn_number": j,
              "embedding_blob": _stub_768d(0.1 * (j + 1))}
             for j in range(rows_per_branch)],
            user_id=user_id, branch_id=b,
        )
    # A migration marker too (simulates Phase 0 having run)
    await backend.conn.execute(
        "INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
        (f"narrative_branch_migration:{session_id}", '{"count": 1}'),
    )
    await backend.conn.commit()


async def _count_rows(conn, table: str, session_id: str, user_id: str) -> int:
    cursor = await conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE session_id = ? AND user_id = ?",
        (session_id, user_id),
    )
    row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def _count_vec_rows_for_session(conn, session_id: str) -> int:
    """Count vec rows whose joined archive row is for this session (any branch)."""
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM narrative_archive_vec v "
        "JOIN narrative_archive na ON na.id = v.id "
        "WHERE na.session_id = ?",
        (session_id,),
    )
    row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def _count_orphan_vec_rows(conn) -> int:
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM narrative_archive_vec v "
        "WHERE NOT EXISTS (SELECT 1 FROM narrative_archive na WHERE na.id = v.id)"
    )
    row = await cursor.fetchone()
    return int(row[0]) if row else 0


# ===========================================================================
# Basic cleanup
# ===========================================================================

class TestBasicCleanup:
    @pytest.mark.asyncio
    async def test_purge_clears_all_narrative_tables(self, backend, persist):
        await _seed_full_narrative_session(backend, persist)
        sid, uid = "ses_full", "user_full"

        report = await purge_narrative_session(backend.conn, sid, user_id=uid)

        assert report.ok is True
        assert report.archive_rows == 4   # 2 branches × 2 rows
        assert report.archive_vec_rows == 4
        assert report.ledger_entry_rows == 4
        assert report.state_snapshot_rows == 2
        assert report.branch_rows == 2
        assert report.memory_rows == 1
        assert report.migration_markers == 1

        # All tables zero for this session
        for table in NARRATIVE_TABLES:
            assert await _count_rows(backend.conn, table, sid, uid) == 0
        # No vec rows joined to deleted archive
        assert await _count_vec_rows_for_session(backend.conn, sid) == 0

    @pytest.mark.asyncio
    async def test_purge_idempotent(self, backend, persist):
        await _seed_full_narrative_session(backend, persist)
        sid, uid = "ses_full", "user_full"
        first = await purge_narrative_session(backend.conn, sid, user_id=uid)
        second = await purge_narrative_session(backend.conn, sid, user_id=uid)
        third = await purge_narrative_session(backend.conn, sid, user_id=uid)
        assert first.ok and second.ok and third.ok
        # Second and third return zeros
        assert second.archive_rows == 0
        assert second.ledger_entry_rows == 0
        assert second.branch_rows == 0
        assert second.memory_rows == 0
        assert third.archive_rows == 0

    @pytest.mark.asyncio
    async def test_purge_requires_user_id(self, backend):
        with pytest.raises(ValueError, match="user_id"):
            await purge_narrative_session(backend.conn, "ses_x", user_id="")


# ===========================================================================
# Vec orphan regression (the pre-existing leak this work fixes)
# ===========================================================================

class TestVecOrphanRegression:
    @pytest.mark.asyncio
    async def test_purge_clears_vec_orphans(self, backend, persist):
        """Regression: prior chat-delete only removed narrative_archive rows;
        the joined vec rows lingered as orphan embeddings forever."""
        await _seed_full_narrative_session(backend, persist)
        sid, uid = "ses_full", "user_full"
        # Confirm vec rows exist before purge
        assert await _count_vec_rows_for_session(backend.conn, sid) == 4

        report = await purge_narrative_session(backend.conn, sid, user_id=uid)
        assert report.archive_vec_rows == 4

        # No vec rows joined to this session's archive
        assert await _count_vec_rows_for_session(backend.conn, sid) == 0
        # And no global orphans were left behind
        assert await _count_orphan_vec_rows(backend.conn) == 0

    @pytest.mark.asyncio
    async def test_residue_detection_catches_pre_existing_orphans(
        self, backend, persist,
    ):
        """If a vec row exists with no archive parent before purge starts,
        the post-delete verification reports it via residue."""
        await _seed_full_narrative_session(backend, persist)
        sid, uid = "ses_full", "user_full"
        # Forge an orphan vec row (id has no archive parent)
        await backend.conn.execute(
            "INSERT INTO narrative_archive_vec (id, embedding) VALUES (?, ?)",
            ("orphan_id_xyz", _stub_768d(0.99)),
        )
        await backend.conn.commit()

        report = await purge_narrative_session(backend.conn, sid, user_id=uid)
        # Purge of this session succeeded but verification found a global orphan
        assert "narrative_archive_vec_global_orphans" in report.residue
        assert report.residue["narrative_archive_vec_global_orphans"] >= 1
        assert report.ok is False  # residue forces ok=False


# ===========================================================================
# Transactional atomicity
# ===========================================================================

class TestTransactionalAtomicity:
    @pytest.mark.asyncio
    async def test_rolls_back_on_mid_purge_failure(self, backend, persist, monkeypatch):
        """Inject a fault after vec+archive delete but before others.
        Verify ROLLBACK undoes the partial deletes."""
        await _seed_full_narrative_session(backend, persist)
        sid, uid = "ses_full", "user_full"

        original_execute = backend.conn.execute
        call_count = {"n": 0}

        async def boom_execute(sql, *args, **kwargs):
            call_count["n"] += 1
            # Let the BEGIN, vec delete, archive delete pass.
            # Fail the snapshot delete (4th statement).
            if call_count["n"] >= 4:
                raise RuntimeError("synthetic fault")
            return await original_execute(sql, *args, **kwargs)

        monkeypatch.setattr(backend.conn, "execute", boom_execute)
        report = await purge_narrative_session(backend.conn, sid, user_id=uid)
        # Restore
        monkeypatch.setattr(backend.conn, "execute", original_execute)

        assert report.ok is False

        # CRITICAL: no rows should have been deleted (rollback)
        # We expect 4 archive, 4 vec, 4 ledger, 2 snapshots, 2 branches, 1 memory
        assert await _count_rows(backend.conn, "narrative_archive", sid, uid) == 4
        assert await _count_rows(backend.conn, "narrative_ledger_entries", sid, uid) == 4
        assert await _count_rows(backend.conn, "narrative_state_snapshots", sid, uid) == 2
        assert await _count_rows(backend.conn, "narrative_branches", sid, uid) == 2
        assert await _count_rows(backend.conn, "narrative_memory", sid, uid) == 1
        # vec also untouched
        assert await _count_vec_rows_for_session(backend.conn, sid) == 4


# ===========================================================================
# Cross-session and cross-user isolation
# ===========================================================================

class TestIsolation:
    @pytest.mark.asyncio
    async def test_purge_isolated_per_session(self, backend, persist):
        await _seed_full_narrative_session(
            backend, persist, session_id="ses_keep", user_id="user_x")
        await _seed_full_narrative_session(
            backend, persist, session_id="ses_drop", user_id="user_x")

        await purge_narrative_session(backend.conn, "ses_drop", user_id="user_x")

        # ses_keep is untouched
        for table in NARRATIVE_TABLES:
            assert await _count_rows(backend.conn, table, "ses_keep", "user_x") > 0, \
                f"{table} lost ses_keep data when ses_drop was purged"
        # ses_drop is empty
        for table in NARRATIVE_TABLES:
            assert await _count_rows(backend.conn, table, "ses_drop", "user_x") == 0

    @pytest.mark.asyncio
    async def test_purge_with_wrong_user_id_deletes_nothing(self, backend, persist):
        """Defensive: if a caller invokes purge with a user_id that doesn't own
        the session, the WHERE clause filters out every row and nothing is
        deleted. Mirrors the auth-isolation pattern used everywhere else."""
        await _seed_full_narrative_session(
            backend, persist, session_id="ses_owned", user_id="user_owner")

        report = await purge_narrative_session(
            backend.conn, "ses_owned", user_id="user_attacker")
        # Purge succeeds (no exception), but every count is 0
        assert report.ok is True
        assert report.archive_rows == 0
        assert report.archive_vec_rows == 0
        assert report.ledger_entry_rows == 0
        assert report.branch_rows == 0
        assert report.memory_rows == 0

        # And the owner's data is fully intact
        for table in NARRATIVE_TABLES:
            assert await _count_rows(
                backend.conn, table, "ses_owned", "user_owner",
            ) > 0, f"{table} lost owner data when wrong user attempted purge"


# ===========================================================================
# CleanupReport API
# ===========================================================================

class TestCleanupReport:
    @pytest.mark.asyncio
    async def test_report_event_kwargs_serializable(self, backend, persist):
        await _seed_full_narrative_session(backend, persist)
        sid, uid = "ses_full", "user_full"
        report = await purge_narrative_session(backend.conn, sid, user_id=uid)
        kwargs = report.to_event_kwargs()
        # All values are loggable scalars / lists
        for k, v in kwargs.items():
            assert isinstance(v, (str, int, float, bool, list)), \
                f"{k}={v!r} is not log-friendly"
        # Required keys
        assert "session_id" in kwargs
        assert "duration_ms" in kwargs
        assert "ok" in kwargs

    @pytest.mark.asyncio
    async def test_report_duration_recorded(self, backend, persist):
        await _seed_full_narrative_session(backend, persist)
        sid, uid = "ses_full", "user_full"
        report = await purge_narrative_session(backend.conn, sid, user_id=uid)
        assert report.duration_ms > 0


# ===========================================================================
# Edge cases — what would silently break in extreme scenarios
# ===========================================================================

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_purge_with_only_legacy_data(self, backend):
        """Pre-migration session (only narrative_memory row, no branches/archives).
        Purge should still succeed and clear the memory row."""
        sid, uid = "ses_legacy_only", "user_legacy"
        await backend.conn.execute(
            "INSERT INTO ui_sessions (id, user_id, title, mode, data) "
            "VALUES (?, ?, 't', 'narrative', '{}')",
            (sid, uid),
        )
        await backend.conn.execute(
            "INSERT INTO sessions (id, user_id) VALUES (?, ?)", (sid, uid),
        )
        await backend.conn.execute(
            """INSERT INTO narrative_memory
               (session_id, card_type, memory_summary, last_summary_at, user_id)
               VALUES (?, 'character', '', 0, ?)""",
            (sid, uid),
        )
        await backend.conn.commit()
        report = await purge_narrative_session(backend.conn, sid, user_id=uid)
        assert report.ok is True
        assert report.memory_rows == 1
        assert report.branch_rows == 0
        assert report.archive_rows == 0

    @pytest.mark.asyncio
    async def test_purge_session_that_never_existed(self, backend):
        """Should return zeros, ok=True, no errors."""
        report = await purge_narrative_session(
            backend.conn, "ses_phantom", user_id="user_phantom")
        assert report.ok is True
        assert report.archive_rows == 0
        assert report.memory_rows == 0
        assert report.branch_rows == 0

    @pytest.mark.asyncio
    async def test_purge_handles_three_branches(self, backend, persist):
        """Many-branch session — confirm the explicit DELETE handles all rows."""
        await _seed_full_narrative_session(
            backend, persist,
            branches=("main", "B", "C", "D"),
            rows_per_branch=3,
        )
        sid, uid = "ses_full", "user_full"
        report = await purge_narrative_session(backend.conn, sid, user_id=uid)
        assert report.ok is True
        assert report.branch_rows == 4
        assert report.ledger_entry_rows == 12  # 4 branches × 3 rows
        assert report.archive_rows == 12
        assert report.archive_vec_rows == 12
        assert report.state_snapshot_rows == 4
