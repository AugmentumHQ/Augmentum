-- 309_library_publications_first_class.sql
-- Library publications (coder "Save to Library" rows, id prefix pub_) reach
-- full parity with artifacts in the item-action layer:
--   * pinnable + taggable (new columns; the /api/library/items UNION now reads
--     these instead of faking pinned=0 / tags='[]').
--   * able to join the activity timeline + manual collections, which were
--     locked to the artifacts table by a FOREIGN KEY on artifact_id.
--
-- SQLite can't DROP a FOREIGN KEY in place, so library_activity and
-- library_collection_items are rebuilt WITHOUT the artifacts(id) reference.
-- Nothing else REFERENCES these two tables, so no foreign_keys pragma dance is
-- required (the copy satisfies the surviving users/collections FKs).
--
-- CONSEQUENCE — the artifacts(id) FK carried ON DELETE CASCADE. Removing it
-- means deleting an item no longer auto-purges its activity/membership rows.
-- Application code now owns that cleanup: ArtifactStore.delete AND
-- PublicationStore.delete both sweep library_activity + library_collection_items
-- by (artifact_id, user_id). See the delete methods in
-- augmentum/tools/artifact_storage.py and augmentum/library/publications.py.

-- 1. Parity columns on publications. The UNION reads these directly.
ALTER TABLE library_publications ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;
ALTER TABLE library_publications ADD COLUMN tags   TEXT    NOT NULL DEFAULT '[]';

CREATE INDEX IF NOT EXISTS idx_library_publications_user_pinned
    ON library_publications(user_id, pinned);

-- 2. Rebuild library_activity without the artifacts(id) FK. artifact_id now
--    holds a UNION id (a real artifact id OR a pub_ publication id).
CREATE TABLE library_activity_new (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    artifact_id     TEXT NOT NULL,                      -- union id (artifact OR pub_)
    action          TEXT NOT NULL,
    surface         TEXT NOT NULL DEFAULT '',
    payload         TEXT NOT NULL DEFAULT '{}',
    occurred_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
INSERT INTO library_activity_new
    SELECT id, user_id, artifact_id, action, surface, payload, occurred_at
    FROM library_activity;
DROP TABLE library_activity;
ALTER TABLE library_activity_new RENAME TO library_activity;

CREATE INDEX IF NOT EXISTS idx_library_activity_user_time
    ON library_activity(user_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_library_activity_artifact_time
    ON library_activity(artifact_id, occurred_at DESC);

-- 3. Rebuild library_collection_items without the artifacts(id) FK. Keeps the
--    collections + users FKs (both satisfied by the copied rows).
CREATE TABLE library_collection_items_new (
    collection_id   TEXT NOT NULL REFERENCES library_collections(id) ON DELETE CASCADE,
    artifact_id     TEXT NOT NULL,                      -- union id (artifact OR pub_)
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    sort_order      INTEGER NOT NULL DEFAULT 0,
    added_at        TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (collection_id, artifact_id)
);
INSERT INTO library_collection_items_new
    SELECT collection_id, artifact_id, user_id, sort_order, added_at
    FROM library_collection_items;
DROP TABLE library_collection_items;
ALTER TABLE library_collection_items_new RENAME TO library_collection_items;

CREATE INDEX IF NOT EXISTS idx_library_collection_items_artifact
    ON library_collection_items(artifact_id);
CREATE INDEX IF NOT EXISTS idx_library_collection_items_user
    ON library_collection_items(user_id);

-- NOTE: keep this description free of semicolons — the migration runner
-- splits statements on ';' and would truncate the string literal.
INSERT OR IGNORE INTO schema_version (version, description)
VALUES (309, 'Library publications first-class - pinned+tags columns, drop artifacts(id) FK on activity + collection_items so union ids participate');
