"""Migration 101 smoke test.

Applies ``101_comic_library_phase_a.sql`` against a minimal pre-101 schema and
verifies:
  - new columns land on file_index with correct defaults
  - comic_series and comic_scan_checkpoint tables exist
  - indexes are created
  - migration is idempotent (re-running doesn't explode)

Does not go through the real migration runner — that's covered by the
end-to-end SQLiteBackend tests. This test stays focused on the SQL itself.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite

_MIGRATION_PATH = (
    Path(__file__).parent.parent
    / "augmentum" / "state" / "migrations" / "101_comic_library_phase_a.sql"
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _seed_pre_101_schema(conn: aiosqlite.Connection) -> None:
    """The minimum schema the migration expects to exist."""
    await conn.executescript("""
        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE schema_version (
            version    INTEGER PRIMARY KEY,
            applied_at INTEGER NOT NULL
        );
        CREATE TABLE file_index (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
            source TEXT NOT NULL,
            source_id TEXT NOT NULL,
            name TEXT NOT NULL,
            mime_type TEXT NOT NULL DEFAULT '',
            size_bytes INTEGER NOT NULL DEFAULT 0,
            real_path TEXT,
            description TEXT NOT NULL DEFAULT '',
            tags TEXT NOT NULL DEFAULT '[]',
            thumbnail TEXT,
            embedding BLOB,
            is_directory INTEGER NOT NULL DEFAULT 0,
            parent_id TEXT,
            source_metadata TEXT NOT NULL DEFAULT '{}',
            kind TEXT NOT NULL DEFAULT '',
            is_favorite INTEGER NOT NULL DEFAULT 0,
            is_trashed INTEGER NOT NULL DEFAULT 0,
            trashed_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE UNIQUE INDEX idx_file_index_source_unique
            ON file_index(user_id, source, source_id);
    """)


async def _apply_migration(conn: aiosqlite.Connection) -> None:
    """Apply the migration with the same per-statement error tolerance the
    real runner uses (swallow ``already exists`` / ``duplicate column``)."""
    sql = _MIGRATION_PATH.read_text(encoding="utf-8")
    # Strip comments to simplify statement splitting
    lines = [
        ln for ln in sql.splitlines()
        if not ln.strip().startswith("--")
    ]
    cleaned = "\n".join(lines)
    # Split on semicolons; executescript can't handle our per-statement
    # error tolerance so we do it manually.
    for stmt in cleaned.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        try:
            await conn.execute(stmt)
        except Exception as e:
            msg = str(e).lower()
            if "already exists" in msg or "duplicate column" in msg:
                continue
            raise
    await conn.commit()


class TestMigration101:
    def test_file_index_columns_added(self):
        async def go():
            conn = await aiosqlite.connect(":memory:")
            conn.row_factory = aiosqlite.Row
            await _seed_pre_101_schema(conn)
            await _apply_migration(conn)

            cursor = await conn.execute("PRAGMA table_info(file_index)")
            cols = {r["name"]: r for r in await cursor.fetchall()}

            assert "scan_status" in cols
            assert cols["scan_status"]["dflt_value"] == "'pending'"
            assert cols["scan_status"]["notnull"] == 1

            assert "mtime" in cols
            assert cols["mtime"]["type"] == "INTEGER"

            assert "scan_error" in cols

            assert "metadata_confidence" in cols
            assert cols["metadata_confidence"]["type"] == "REAL"
            assert cols["metadata_confidence"]["dflt_value"] == "0.5"

            assert "series_id" in cols
            await conn.close()
        _run(go())

    def test_comic_series_table_created(self):
        async def go():
            conn = await aiosqlite.connect(":memory:")
            conn.row_factory = aiosqlite.Row
            await _seed_pre_101_schema(conn)
            await _apply_migration(conn)

            cursor = await conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='comic_series'"
            )
            assert await cursor.fetchone() is not None

            cursor = await conn.execute("PRAGMA table_info(comic_series)")
            cols = {r["name"] for r in await cursor.fetchall()}
            required = {
                "id", "user_id", "canonical_name", "sort_name", "alias_names",
                "publisher", "author", "description", "cover_file_id",
                "status", "year_started", "year_ended", "genres",
                "language_iso", "age_rating", "metadata_source",
                "metadata_confidence", "archive_count_reported", "accent_color",
                "created_at", "updated_at",
            }
            assert required.issubset(cols), (
                f"missing columns: {required - cols}"
            )
            await conn.close()
        _run(go())

    def test_comic_scan_checkpoint_table_created(self):
        async def go():
            conn = await aiosqlite.connect(":memory:")
            conn.row_factory = aiosqlite.Row
            await _seed_pre_101_schema(conn)
            await _apply_migration(conn)

            cursor = await conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='comic_scan_checkpoint'"
            )
            assert await cursor.fetchone() is not None

            cursor = await conn.execute(
                "PRAGMA table_info(comic_scan_checkpoint)"
            )
            cols = {r["name"] for r in await cursor.fetchall()}
            required = {
                "user_id", "library_root", "started_at", "total_found",
                "completed", "failed", "status", "last_path", "observed_rate",
                "updated_at",
            }
            assert required.issubset(cols), (
                f"missing columns: {required - cols}"
            )
            # Composite PK
            pk_cursor = await conn.execute(
                "PRAGMA table_info(comic_scan_checkpoint)"
            )
            pk_cols = [r["name"] for r in await pk_cursor.fetchall() if r["pk"] > 0]
            assert set(pk_cols) == {"user_id", "library_root"}
            await conn.close()
        _run(go())

    def test_indexes_created(self):
        async def go():
            conn = await aiosqlite.connect(":memory:")
            conn.row_factory = aiosqlite.Row
            await _seed_pre_101_schema(conn)
            await _apply_migration(conn)

            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
            indexes = {r["name"] for r in await cursor.fetchall()}
            expected = {
                "idx_file_index_scan_status",
                "idx_file_index_series",
                "idx_file_index_mtime",
                "idx_comic_series_user_sort",
                "idx_comic_series_user_updated",
                "idx_comic_scan_checkpoint_status",
            }
            assert expected.issubset(indexes), (
                f"missing indexes: {expected - indexes}"
            )
            await conn.close()
        _run(go())

    def test_schema_version_bumped(self):
        async def go():
            conn = await aiosqlite.connect(":memory:")
            conn.row_factory = aiosqlite.Row
            await _seed_pre_101_schema(conn)
            await _apply_migration(conn)

            cursor = await conn.execute(
                "SELECT MAX(version) as v FROM schema_version"
            )
            row = await cursor.fetchone()
            assert row["v"] == 101
            await conn.close()
        _run(go())

    def test_idempotent_rerun(self):
        """Running the migration twice should succeed (runner swallows
        'duplicate column' / 'already exists' errors)."""
        async def go():
            conn = await aiosqlite.connect(":memory:")
            conn.row_factory = aiosqlite.Row
            await _seed_pre_101_schema(conn)
            await _apply_migration(conn)
            await _apply_migration(conn)

            cursor = await conn.execute("PRAGMA table_info(file_index)")
            cols = {r["name"] for r in await cursor.fetchall()}
            assert "scan_status" in cols
            assert "series_id" in cols
            await conn.close()
        _run(go())

    def test_defaults_applied_to_existing_rows(self):
        """Existing file_index rows should get default scan_status='pending'
        and metadata_confidence=0.5 on ALTER TABLE."""
        async def go():
            conn = await aiosqlite.connect(":memory:")
            conn.row_factory = aiosqlite.Row
            await _seed_pre_101_schema(conn)
            # Seed an existing row before applying the migration
            await conn.execute(
                "INSERT INTO users (id) VALUES ('u_a')"
            )
            await conn.execute(
                "INSERT INTO file_index (id, user_id, source, source_id, name) "
                "VALUES ('fi_legacy', 'u_a', 'audiobookshelf', 'abs_1', 'Old Book')"
            )
            await conn.commit()

            await _apply_migration(conn)

            cursor = await conn.execute(
                "SELECT scan_status, metadata_confidence, series_id, mtime "
                "FROM file_index WHERE id = 'fi_legacy'"
            )
            row = await cursor.fetchone()
            assert row["scan_status"] == "pending"
            assert row["metadata_confidence"] == 0.5
            assert row["series_id"] is None
            assert row["mtime"] is None
            await conn.close()
        _run(go())
