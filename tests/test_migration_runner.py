"""Tests for migration runner — apply all migrations on fresh :memory: DB."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from augmentum.state.backends.sqlite import (
    MigrationValidationError,
    SQLiteBackend,
    _list_existing_tables,
    _migration_required_tables,
    _split_sql_statements,
)

MIGRATIONS_DIR = Path(__file__).parent.parent / "augmentum" / "state" / "migrations"


class TestMigrationRunner:
    """Run all SQL migrations and verify no errors."""

    async def test_all_migrations_apply_cleanly(self):
        """Every migration file should execute without SQL errors."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        # connect() runs migrations automatically — if we get here, all passed
        assert backend._conn is not None
        await backend.close()

    async def test_migrations_idempotent(self):
        """Running migrations twice should not fail (IF NOT EXISTS guards)."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        # Run migrations again manually
        await backend._run_migrations()
        assert backend._conn is not None
        await backend.close()

    async def test_sessions_table_exists(self):
        """The sessions table should be created by migration 001."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        cursor = await backend.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        row = await cursor.fetchone()
        assert row is not None
        await backend.close()

    async def test_memories_table_exists(self):
        """The memories table should be created by an early migration."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        cursor = await backend.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        )
        row = await cursor.fetchone()
        assert row is not None
        await backend.close()

    async def test_documented_user_scoped_tables_have_user_id_column(self):
        """Every table CLAUDE.md lists as user-scoped must actually carry a
        user_id column in the migrated schema.

        Guards the isolation invariant: a new user-scoped table (or an
        ALTER that forgets user_id) would let cross-tenant rows through,
        and the doc-fact list alone can't detect a missing column. This is
        the broad backstop for the store-level isolation fixes (empty
        user_id no longer drops the WHERE clause in the per-user stores).
        """
        import re

        claude_md = Path(__file__).parent.parent / "CLAUDE.md"
        text = claude_md.read_text(encoding="utf-8")
        match = re.search(
            r"<!--fact:tables\.user_scoped\.list-->(.*?)<!--/-->",
            text,
            re.DOTALL,
        )
        assert match, "user-scoped table doc-fact not found in CLAUDE.md"
        documented = {t.strip() for t in match.group(1).split(",") if t.strip()}
        assert documented, "documented user-scoped table list is empty"

        # Tables the doc-fact lists as user-scoped but which scope by a
        # differently-named column by design. fabric_replay_watermarks is
        # SERVER-LEVEL federation infra keyed by ``owner_id`` (a user_id for
        # per-user E2E streams, '' for instance-level relay) — see migration
        # 292. It has no user_id column on purpose; the doc-fact generator
        # picked it up from a "user_id" mention in a column comment.
        scope_column_exceptions = {
            "fabric_replay_watermarks": "owner_id",
        }

        backend = SQLiteBackend(":memory:")
        await backend.connect()
        try:
            missing_column: list[str] = []
            for table in sorted(documented):
                cursor = await backend.conn.execute(
                    f"PRAGMA table_info({table})"
                )
                cols = {row[1] for row in await cursor.fetchall()}
                expected = scope_column_exceptions.get(table, "user_id")
                # Absent tables (created outside the migration runner) are
                # skipped; the guard is that PRESENT user-scoped tables
                # carry a tenant-scope column.
                if cols and expected not in cols:
                    missing_column.append(f"{table} (want {expected})")
            assert not missing_column, (
                "user-scoped tables missing their tenant-scope column: "
                + ", ".join(missing_column)
            )
        finally:
            await backend.close()

    async def test_app_settings_table_exists(self):
        """The app_settings table should be created by migration 007."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        cursor = await backend.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='app_settings'"
        )
        row = await cursor.fetchone()
        assert row is not None
        await backend.close()

    async def test_schema_version_tracked(self):
        """After migrations, schema_version should reflect applied count."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        cursor = await backend.conn.execute(
            "SELECT MAX(version) FROM schema_version"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] is not None
        assert row[0] > 0
        await backend.close()

    async def test_providers_table_exists(self):
        """The providers table should exist after migrations."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        cursor = await backend.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='providers'"
        )
        row = await cursor.fetchone()
        assert row is not None
        await backend.close()

    async def test_document_chunks_table_exists(self):
        """The document_chunks table should exist for document RAG."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        cursor = await backend.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='document_chunks'"
        )
        row = await cursor.fetchone()
        assert row is not None
        await backend.close()

    def test_split_sql_statements_basic(self):
        """Test SQL statement splitting handles semicolons."""
        sql = "CREATE TABLE a (id INT);\nCREATE TABLE b (id INT);"
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 2

    def test_split_sql_statements_skips_comments(self):
        """Comment-only lines should be skipped."""
        sql = "-- This is a comment\nCREATE TABLE x (id INT);"
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 1

    def test_split_sql_statements_begin_end(self):
        """BEGIN...END blocks should be kept as single statements."""
        sql = (
            "CREATE TRIGGER t AFTER INSERT ON x\n"
            "BEGIN\n"
            "  UPDATE y SET n = n + 1;\n"
            "END;"
        )
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 1
        assert "BEGIN" in stmts[0]

    async def test_migration_files_exist(self):
        """At least the initial migration file should exist."""
        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        assert len(migration_files) >= 1
        assert "001" in migration_files[0].stem

    async def test_repair_phantom_fts_noop_on_healthy_install(self):
        """Healthy install: _repair_phantom_fts_if_needed is a no-op."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        # memories_fts was created by migration 006 and is healthy.
        # Running the repair should do nothing observable.
        await backend._repair_phantom_fts_if_needed()

        cursor = await backend.conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='memories_fts'"
        )
        assert await cursor.fetchone() is not None
        # Basic FTS insert/match still works.
        await backend.conn.execute(
            "INSERT INTO memories (id, content, memory_type, importance) "
            "VALUES ('mem-1', 'alpha bravo', 'fact', 0.5)"
        )
        await backend.conn.commit()
        cursor = await backend.conn.execute(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'bravo'"
        )
        assert (await cursor.fetchone())[0] == 1
        await backend.close()

    async def test_repair_phantom_fts_noop_when_table_missing(self):
        """If memories_fts is entirely absent (pre-migration-006 state),
        the repair is a silent no-op rather than an error."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        # Remove memories_fts + shadow tables cleanly so there's no
        # phantom row at all.
        await backend.conn.execute("PRAGMA writable_schema = 1")
        await backend.conn.execute(
            "DELETE FROM sqlite_master "
            "WHERE name = 'memories_fts' "
            "   OR name LIKE 'memories_fts\\_%' ESCAPE '\\'"
        )
        await backend.conn.execute("PRAGMA writable_schema = RESET")
        await backend.conn.commit()
        # No phantom, no table — repair should detect and skip.
        await backend._repair_phantom_fts_if_needed()
        await backend.close()

    async def test_repair_phantom_fts_recovers_dangling_triggers(self):
        """Simple drift: memories_fts was dropped cleanly but the
        triggers survived. Repair detects the phantom via the probe and
        rebuilds the FTS table + triggers."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        await backend.conn.execute(
            "INSERT INTO memories (id, content, memory_type, importance) "
            "VALUES ('mem-1', 'alpha bravo charlie', 'fact', 0.5)"
        )
        await backend.conn.commit()

        # Clean drop. Triggers stay; shadow tables are gone.
        await backend.conn.execute("DROP TABLE memories_fts")
        await backend.conn.commit()

        # UPDATE fires memories_au → fails on missing memories_fts.
        err_raised = False
        try:
            await backend.conn.execute(
                "UPDATE memories SET access_count = access_count + 1 "
                "WHERE id = 'mem-1'"
            )
            await backend.conn.commit()
        except aiosqlite.OperationalError as exc:
            assert "memories_fts" in str(exc)
            err_raised = True
        assert err_raised

        await backend._repair_phantom_fts_if_needed()

        # UPDATE now works and FTS is re-indexed from memories.
        await backend.conn.execute(
            "UPDATE memories SET access_count = access_count + 1 "
            "WHERE id = 'mem-1'"
        )
        await backend.conn.commit()
        cursor = await backend.conn.execute(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'bravo'"
        )
        assert (await cursor.fetchone())[0] == 1
        await backend.close()

    async def test_repair_phantom_fts_recovers_phantom_master_entry(self):
        """Production incident: the sqlite_master row for memories_fts is
        still present (so CREATE VIRTUAL TABLE IF NOT EXISTS would
        short-circuit) but the shadow tables are gone. Only the
        writable_schema force-delete + RESET path recovers this state,
        which is exactly what _repair_phantom_fts_if_needed does.
        """
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        await backend.conn.execute(
            "INSERT INTO memories (id, content, memory_type, importance) "
            "VALUES ('mem-2', 'golf hotel india', 'fact', 0.5)"
        )
        await backend.conn.commit()

        # Fabricate the phantom: keep the memories_fts row in
        # sqlite_master, delete the shadow-table rows. This is the
        # minimum reproducer for the broken production state.
        await backend.conn.execute("PRAGMA writable_schema = 1")
        await backend.conn.execute(
            "DELETE FROM sqlite_master "
            "WHERE name LIKE 'memories_fts\\_%' ESCAPE '\\'"
        )
        await backend.conn.execute("PRAGMA writable_schema = RESET")
        await backend.conn.commit()

        # Probe confirms the phantom state: a SELECT raises "no such
        # table" even though the row still appears in sqlite_master.
        cursor = await backend.conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='memories_fts'"
        )
        assert await cursor.fetchone() is not None, (
            "Phantom setup: memories_fts row must remain in sqlite_master"
        )
        raised = False
        try:
            probe = await backend.conn.execute(
                "SELECT rowid FROM memories_fts LIMIT 0"
            )
            await probe.fetchall()
        except aiosqlite.DatabaseError as exc:
            # Phantom FTS can raise either OperationalError ("no such
            # table") or DatabaseError ("vtable constructor failed")
            # depending on which part of the FTS5 machinery fails first.
            assert "memories_fts" in str(exc)
            raised = True
        assert raised, "Phantom state must make SELECT fail"

        # Repair: clean phantom, reset schema cache, recreate.
        await backend._repair_phantom_fts_if_needed()

        # Queryable now, indexed from memories.
        cursor = await backend.conn.execute(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'hotel'"
        )
        assert (await cursor.fetchone())[0] == 1
        await backend.conn.execute(
            "UPDATE memories SET access_count = access_count + 1 "
            "WHERE id = 'mem-2'"
        )
        await backend.conn.commit()
        await backend.close()

    async def test_repair_phantom_file_index_fts_recovers_phantom_master_entry(self):
        """Same phantom pattern as memories_fts, but for file_index_fts.

        Reproduces the exact broken state observed in the dogfood DB on
        2026-04-20: the virtual-table row ``file_index_fts`` is missing
        from sqlite_master, but the five shadow tables (``_config``,
        ``_data``, ``_docsize``, ``_idx``) AND the three sync triggers
        all survived a prior corruption. Every INSERT into ``file_index``
        fired the ``file_index_fts_insert`` trigger, which raised
        ``no such table: main.file_index_fts`` — 453 audiobook rows
        silently rejected.

        Unlike the memories_fts equivalent (which mirrors the *opposite*
        phantom — row kept, shadows deleted), the production breakage
        here is row-missing, shadows-present. Both shapes are phantoms;
        the repair dispatcher must handle either.
        """
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        # Seed a file_index row so the rebuild has content to reindex.
        await backend.conn.execute(
            "INSERT INTO file_index (id, user_id, source, source_id, name) "
            "VALUES ('fi_t1', 'u', 'audiobookshelf', 'ext1', 'Alpha Bravo')"
        )
        await backend.conn.commit()

        # Fabricate the production phantom: DELETE the virtual-table row
        # from sqlite_master, keep shadow tables + triggers. RESET the
        # parsed-schema cache so the next SQL statement observes the
        # deletion (exactly the state after a corruption + restore).
        await backend.conn.execute("PRAGMA writable_schema = 1")
        await backend.conn.execute(
            "DELETE FROM sqlite_master "
            "WHERE type='table' AND name='file_index_fts'"
        )
        await backend.conn.execute("PRAGMA writable_schema = RESET")
        await backend.conn.commit()

        # The insert trigger is now broken: any write to file_index
        # fires file_index_fts_insert → "no such table".
        raised = False
        try:
            await backend.conn.execute(
                "INSERT INTO file_index (id, user_id, source, source_id, name) "
                "VALUES ('fi_t2', 'u', 'audiobookshelf', 'ext2', 'Charlie Delta')"
            )
            await backend.conn.commit()
        except aiosqlite.OperationalError as exc:
            assert "file_index_fts" in str(exc)
            raised = True
        assert raised, "Phantom state must make trigger-fired INSERT fail"

        # Repair.
        await backend._repair_phantom_fts_if_needed()

        # Writes work now and content is searchable. The rebuild also
        # reindexes the pre-phantom row ('Alpha Bravo') so we don't lose
        # existing data.
        await backend.conn.execute(
            "INSERT INTO file_index (id, user_id, source, source_id, name) "
            "VALUES ('fi_t2', 'u', 'audiobookshelf', 'ext2', 'Charlie Delta')"
        )
        await backend.conn.commit()
        cursor = await backend.conn.execute(
            "SELECT COUNT(*) FROM file_index_fts WHERE file_index_fts MATCH 'bravo'"
        )
        assert (await cursor.fetchone())[0] == 1
        cursor = await backend.conn.execute(
            "SELECT COUNT(*) FROM file_index_fts WHERE file_index_fts MATCH 'delta'"
        )
        assert (await cursor.fetchone())[0] == 1
        await backend.close()

    # ── File-tracking runner (added 2026-06-02) ──────────────────────
    #
    # Before this rewrite, the runner gated on ``version <= MAX(version)``
    # which silently skipped a new migration file with an out-of-order
    # version number. The fix tracks applied migrations by FILENAME in
    # a separate table; these tests pin that contract.

    async def test_migration_files_applied_table_populates(self):
        """Every applied migration ends up in migration_files_applied."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        cursor = await backend.conn.execute(
            "SELECT COUNT(*) FROM migration_files_applied"
        )
        recorded = (await cursor.fetchone())[0]
        # All real migration files for fresh install should be recorded.
        files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        assert recorded == len(files), (
            f"expected {len(files)} files tracked, got {recorded}"
        )
        await backend.close()

    async def test_out_of_order_migration_applies(self, tmp_path):
        """A new migration file with a version <= prior MAX must still
        apply — that's the whole point of the FILENAME-tracking rewrite.

        Mirrors the actual production case where a 230_observation_l0.sql
        was added after 232/233 had already shipped: with the old
        runner, the new file was silently skipped.
        """
        # Set up a real on-disk DB (so the connection survives the
        # post-fixture probe) with the production migrations baseline.
        db_path = tmp_path / "scratch.db"
        backend = SQLiteBackend(str(db_path))
        await backend.connect()

        # Confirm baseline: file-tracking table populated.
        cursor = await backend.conn.execute(
            "SELECT COUNT(*) FROM migration_files_applied"
        )
        baseline_count = (await cursor.fetchone())[0]
        assert baseline_count > 0
        await backend.close()

        # Drop a SYNTHETIC out-of-order migration into a SCRATCH dir
        # and point the runner at it. We swap _MIGRATIONS_DIR rather
        # than the real one to avoid polluting the project.
        from augmentum.state.backends import sqlite as sqlite_mod
        scratch_migrations = tmp_path / "fake_migrations"
        scratch_migrations.mkdir()
        # Copy real migrations into the scratch dir so the existing
        # state is preserved.
        for src in MIGRATIONS_DIR.glob("*.sql"):
            (scratch_migrations / src.name).write_bytes(src.read_bytes())
        # Now add an out-of-order one — version 50, behind everything.
        oop_path = scratch_migrations / "050_zzz_out_of_order_test.sql"
        oop_path.write_text(
            "CREATE TABLE IF NOT EXISTS oop_test_table (id INTEGER PRIMARY KEY);\n"
        )

        old_dir = sqlite_mod._MIGRATIONS_DIR
        sqlite_mod._MIGRATIONS_DIR = scratch_migrations
        try:
            backend2 = SQLiteBackend(str(db_path))
            await backend2.connect()
            # The out-of-order migration must have applied.
            cursor = await backend2.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='oop_test_table'"
            )
            assert await cursor.fetchone() is not None, (
                "out-of-order migration was not applied"
            )
            # And tracked.
            cursor = await backend2.conn.execute(
                "SELECT 1 FROM migration_files_applied WHERE filename = ?",
                (oop_path.name,),
            )
            assert await cursor.fetchone() is not None
            await backend2.close()
        finally:
            sqlite_mod._MIGRATIONS_DIR = old_dir

    async def test_backfill_on_first_run_skips_pre_applied(self, tmp_path):
        """Existing install (schema_version watermark > 0, no file-
        tracking table) should backfill the file-tracking table from
        the watermark — NOT re-run migrations 1..N."""
        db_path = tmp_path / "existing.db"
        backend = SQLiteBackend(str(db_path))
        await backend.connect()
        await backend.close()

        # Simulate the "pre-rewrite install" state: drop the file-
        # tracking table entirely; schema_version stays.
        import aiosqlite
        conn = await aiosqlite.connect(str(db_path))
        await conn.execute("DROP TABLE migration_files_applied")
        await conn.commit()
        await conn.close()

        # Re-connect → runner should rebuild the file-tracking table
        # AND backfill it without re-running migrations.
        backend2 = SQLiteBackend(str(db_path))
        await backend2.connect()
        cursor = await backend2.conn.execute(
            "SELECT COUNT(*) FROM migration_files_applied"
        )
        recorded = (await cursor.fetchone())[0]
        # All non-zero migration files should be backfilled as applied.
        files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        assert recorded == len(files)
        await backend2.close()

    async def test_runner_idempotent_with_file_tracking(self):
        """Running migrations twice doesn't double-apply or re-record."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        cursor = await backend.conn.execute(
            "SELECT COUNT(*) FROM migration_files_applied"
        )
        first = (await cursor.fetchone())[0]
        await backend._run_migrations()
        cursor = await backend.conn.execute(
            "SELECT COUNT(*) FROM migration_files_applied"
        )
        second = (await cursor.fetchone())[0]
        assert first == second
        await backend.close()

    async def test_repair_phantom_fts_runs_before_migrations(self):
        """End-to-end sanity: a fabricated phantom on a DB that stops at
        schema_version=95 should self-heal on the next connect() and
        then advance to 96."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        # Fabricate phantom + rewind schema_version to 95 so the next
        # fake "start" applies migration 096 fresh.
        await backend.conn.execute("PRAGMA writable_schema = 1")
        await backend.conn.execute(
            "DELETE FROM sqlite_master "
            "WHERE name LIKE 'memories_fts\\_%' ESCAPE '\\'"
        )
        await backend.conn.execute("PRAGMA writable_schema = RESET")
        await backend.conn.execute(
            "DELETE FROM schema_version WHERE version = 96"
        )
        await backend.conn.commit()

        # Simulate startup: repair first, then migrations.
        await backend._repair_phantom_fts_if_needed()
        await backend._run_migrations()

        cursor = await backend.conn.execute(
            "SELECT MAX(version) FROM schema_version"
        )
        assert (await cursor.fetchone())[0] >= 96
        cursor = await backend.conn.execute(
            "SELECT COUNT(*) FROM memories_fts"
        )
        await cursor.fetchone()  # succeeds == no phantom
        await backend.close()


# --------------------------------------------------------------------------
# Migration pre-flight validation
#
# Regression coverage for the migration-243 incident: a migration referenced
# `settings` when the actual table is `app_settings`, the runner had no
# pre-flight check, the SQL error fell out of bootstrap, and the connect
# path silently fell back to in-memory — every authed endpoint then 503'd
# with `auth_unavailable_denied`. The validator catches typo'd table names
# BEFORE any of the migration's statements run.
# --------------------------------------------------------------------------

class TestMigrationRequiredTables:
    """Pure-function tests for the table-reference parser."""

    def test_empty_sql(self):
        assert _migration_required_tables("") == set()

    def test_create_only_no_requirements(self):
        sql = "CREATE TABLE foo (id INTEGER PRIMARY KEY);"
        assert _migration_required_tables(sql) == set()

    def test_create_then_use_no_requirements(self):
        """A migration that CREATEs and then INSERTs into the same table is
        self-contained — INSERT target shouldn't appear in required set."""
        sql = (
            "CREATE TABLE foo (id INTEGER PRIMARY KEY);\n"
            "INSERT INTO foo (id) VALUES (1);"
        )
        assert _migration_required_tables(sql) == set()

    def test_delete_from_external_table(self):
        sql = "DELETE FROM existing_table WHERE k = 'x';"
        assert _migration_required_tables(sql) == {"existing_table"}

    def test_update_external_table(self):
        sql = "UPDATE other_tbl SET col = 1 WHERE id = 2;"
        assert _migration_required_tables(sql) == {"other_tbl"}

    def test_alter_external_table(self):
        sql = "ALTER TABLE old_thing ADD COLUMN x TEXT;"
        assert _migration_required_tables(sql) == {"old_thing"}

    def test_insert_into_external_table(self):
        sql = "INSERT OR IGNORE INTO ext (k, v) VALUES ('a', 'b');"
        assert _migration_required_tables(sql) == {"ext"}

    def test_create_index_on_external_table(self):
        sql = "CREATE UNIQUE INDEX idx_x ON existing_tbl(col);"
        assert _migration_required_tables(sql) == {"existing_tbl"}

    def test_create_trigger_on_external_table(self):
        sql = (
            "CREATE TRIGGER t1 AFTER INSERT ON foo BEGIN\n"
            "  UPDATE bar SET n = n + 1;\n"
            "END;"
        )
        # bar appears via UPDATE; foo is the trigger target.
        required = _migration_required_tables(sql)
        assert "foo" in required
        assert "bar" in required

    def test_migration_243_typo_regression(self):
        """The exact shape that broke production: DELETE FROM `settings`
        when the real table is `app_settings`. Validator should flag
        `settings` as required (and the bootstrap check will then catch
        that no such table exists)."""
        sql = (
            "DELETE FROM audio_providers WHERE id = 'kittentts-builtin';\n"
            "DELETE FROM settings WHERE key IN ('tts_kitten_builtin');\n"
            "DELETE FROM user_settings WHERE key IN ('tts_kitten_builtin');"
        )
        required = _migration_required_tables(sql)
        assert required == {"audio_providers", "settings", "user_settings"}

    def test_create_table_with_quoted_name(self):
        """Quoted/backtick-wrapped names should still be recognized."""
        for quoted in ('"foo"', "'foo'", "`foo`", "foo"):
            sql = f"CREATE TABLE {quoted} (id INT);\nINSERT INTO foo (id) VALUES (1);"
            assert _migration_required_tables(sql) == set(), \
                f"Failed for quote style: {quoted!r}"

    def test_rename_to_counts_as_create_for_followups(self):
        """Migration 200 case: ALTER TABLE old RENAME TO new, followed by
        ALTER TABLE new ADD COLUMN. The new name didn't exist before this
        migration, but it's effectively created by the rename, so it
        should NOT be flagged as missing for the subsequent statements."""
        sql = (
            "ALTER TABLE coder_workspaces RENAME TO project_checkouts;\n"
            "ALTER TABLE project_checkouts ADD COLUMN project_id TEXT;\n"
            "CREATE INDEX idx_pc_proj ON project_checkouts(project_id);\n"
            "UPDATE project_checkouts SET project_id = id;"
        )
        # Only the original table name (coder_workspaces) is required; the
        # new name is treated as created within the same migration.
        required = _migration_required_tables(sql)
        assert "project_checkouts" not in required
        assert "coder_workspaces" in required


class TestMigrationValidationIntegration:
    """End-to-end: simulating the runner's pre-flight against a real DB."""

    async def test_pre_flight_catches_missing_table_via_helper(self):
        """Validate by-helper: feed required tables vs an existing set, the
        difference is the missing list reported by the validator."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        try:
            existing = await _list_existing_tables(backend.conn)
            # Sanity: the real tables we know exist after a fresh bootstrap.
            assert "app_settings" in existing
            assert "user_settings" in existing
            assert "settings" not in existing  # the regression case

            # The bad migration's required set vs the truth:
            required = _migration_required_tables(
                "DELETE FROM settings WHERE key = 'x';"
            )
            missing = required - existing
            assert missing == {"settings"}
        finally:
            await backend.close()

    async def test_pre_flight_blocks_application(self, tmp_path, monkeypatch):
        """When a migration with a missing table reference runs through the
        full _run_migrations loop, it should raise MigrationValidationError
        BEFORE any of its statements execute (no partial state)."""
        import augmentum.state.backends.sqlite as sqlite_mod

        # Create a fake migration dir with one good baseline + one bad file.
        mig_dir = tmp_path / "migs"
        mig_dir.mkdir()
        (mig_dir / "001_baseline.sql").write_text(
            "CREATE TABLE good (id INTEGER PRIMARY KEY);\n"
        )
        # Bad migration references a table that doesn't exist anywhere.
        (mig_dir / "002_bad_ref.sql").write_text(
            "DELETE FROM nonexistent_tbl WHERE id = 1;\n"
        )
        monkeypatch.setattr(sqlite_mod, "_MIGRATIONS_DIR", mig_dir)

        backend = SQLiteBackend(":memory:")
        # connect() runs the migrations. Should raise on the second file.
        try:
            try:
                await backend.connect()
                raise AssertionError(
                    "expected MigrationValidationError, got clean connect"
                )
            except MigrationValidationError as exc:
                assert "002_bad_ref.sql" in str(exc)
                assert "nonexistent_tbl" in str(exc)
        finally:
            if backend._conn is not None:
                await backend.close()
