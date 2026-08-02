-- 135_file_index_user_recent.sql
-- Composite index covering the Files-tab default chip ("All / newest").
--
-- Pre-fix EXPLAIN PLAN:
--   SEARCH file_index USING INDEX idx_file_index_mtime (user_id=?)
--   USE TEMP B-TREE FOR ORDER BY                       ← in-memory sort
--
-- The existing ``idx_file_index_mtime(user_id, mtime)`` could satisfy
-- ``WHERE user_id=?`` but not ``ORDER BY created_at DESC``, so SQLite
-- read every user-scoped row, sorted in memory, then took LIMIT 60.
-- Cost on a 64k-row table: 126ms median. Filtered chips (single
-- source / favorites / trash) hit purpose-built indexes in <1ms.
-- The default chip was the slowest one.
--
-- After this index lands the optimizer walks rows in ``created_at DESC``
-- order (within ``user_id, is_trashed=0``) and stops at LIMIT 60. No
-- temp B-tree, no memory sort, no full materialize of every row.
--
-- Index column order:
--   1. user_id        — equality predicate
--   2. is_trashed     — equality predicate (always 0 on this path)
--   3. created_at DESC — order-by, walked in index order
--
-- Combined with the SELECT * → explicit-columns change in
-- vfs/index.py (drops the 3KB embedding BLOB read per row), the
-- All/newest path drops from ~126ms to <2ms.

CREATE INDEX IF NOT EXISTS idx_file_index_user_recent
    ON file_index(user_id, is_trashed, created_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (135, 'file_index user/trashed/created composite index');
