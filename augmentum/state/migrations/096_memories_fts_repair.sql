-- 096_memories_fts_repair.sql: Version marker for the memories_fts
-- phantom-repair work. The actual repair runs in Python at connect
-- time (see ``SQLiteBackend._repair_phantom_fts_if_needed`` in
-- ``augmentum/state/backends/sqlite.py``), BEFORE migrations execute.
--
-- Why not pure SQL? Two interacting sqlite quirks make the repair
-- unreliable in a migration:
--
-- 1) ``CREATE VIRTUAL TABLE IF NOT EXISTS`` reads from the connection's
--    cached parsed schema. If a phantom sqlite_master row was just
--    DELETEd via writable_schema, the cache still shows the table as
--    existing and the CREATE short-circuits silently. The subsequent
--    rebuild INSERT then fails with "no such table: memories_fts".
--
-- 2) The migration runner (``_run_migrations``) swallows errors
--    containing "already exists" for ALTER TABLE idempotency. A
--    cached-schema CREATE raises exactly that shape, so the migration
--    can't even observe that it failed.
--
-- The Python path sidesteps both issues: it uses
-- ``PRAGMA writable_schema = RESET`` to invalidate the schema cache
-- between the DELETE and the CREATE, and it handles errors directly
-- rather than going through the migration runner. This SQL file just
-- records that the repair step belongs to schema version 96 so
-- accounting stays consistent across environments.

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (96, 'memories_fts phantom repair (handled in Python: _repair_phantom_fts_if_needed)');
