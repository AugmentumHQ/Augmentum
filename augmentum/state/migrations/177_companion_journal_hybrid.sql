-- 177_companion_journal_hybrid.sql
-- Hybrid retrieval substrate for companion_journal — enables the
-- Reference Resolver (Piece 6) to do fast KNN + keyword search over
-- the inner stream.
--
-- Two virtual tables:
--   companion_journal_fts  — FTS5 over the content column
--   companion_journal_vec  — vec0 mirror of the existing embedding BLOB
--
-- Design notes — these matter for production stability:
--
-- 1. FTS5 is wired as a CONTENT TABLE (no duplicate storage). Triggers
--    keep it in sync on INSERT/UPDATE/DELETE so application code never
--    has to think about it. Standard SQLite recipe.
--
-- 2. vec0 is NOT triggered. Triggers can't bind BLOBs cleanly to the
--    virtual table, and we want the mirror write to be best-effort and
--    swallow-on-failure (extension might not be loaded). Python side
--    handles it in memory.py::journal() with try/except, same pattern
--    as file_index._upsert_file_vec.
--
-- 3. Backfill is NOT in this migration. A `journal_vec_backfill` job
--    runs paged with sleeps between batches so the tick loop never
--    stalls. Migration only creates schema; no data walk.
--
-- 4. IF NOT EXISTS everywhere — runner re-runs are safe.

-- FTS5 over content. Porter+unicode61 tokenizer matches what
-- file_index_fts uses (097_file_index_fts_repair.sql) so query parsing
-- behaves consistently across the resolver's two main legs.
CREATE VIRTUAL TABLE IF NOT EXISTS companion_journal_fts USING fts5(
    content,
    content='companion_journal',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Sync triggers — keep FTS5 mirror coherent with the base table.
-- AFTER INSERT: copy new content into FTS.
CREATE TRIGGER IF NOT EXISTS companion_journal_fts_ai
AFTER INSERT ON companion_journal BEGIN
    INSERT INTO companion_journal_fts(rowid, content)
    VALUES (new.id, new.content);
END;

-- AFTER DELETE: tombstone the FTS row (FTS5 'delete' command).
CREATE TRIGGER IF NOT EXISTS companion_journal_fts_ad
AFTER DELETE ON companion_journal BEGIN
    INSERT INTO companion_journal_fts(companion_journal_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;

-- AFTER UPDATE: delete-then-insert (FTS5 doesn't have a true update).
CREATE TRIGGER IF NOT EXISTS companion_journal_fts_au
AFTER UPDATE ON companion_journal BEGIN
    INSERT INTO companion_journal_fts(companion_journal_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO companion_journal_fts(rowid, content)
    VALUES (new.id, new.content);
END;

-- vec0 mirror — 768-dim float32, keyed by journal id.
-- nomic-embed-text-v1.5 dim matches what file_index_vec uses (768),
-- so embeddings are interoperable between the two sources.
CREATE VIRTUAL TABLE IF NOT EXISTS companion_journal_vec USING vec0(
    journal_id INTEGER PRIMARY KEY,
    embedding  float[768]
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (177, 'companion_journal hybrid retrieval: FTS5 + vec0 mirror');
