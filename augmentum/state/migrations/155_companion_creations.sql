-- 155_companion_creations.sql
-- Things she makes — commitment 5 (autonomous creative output).
--
-- During reflective time the companion produces artifacts: short prose,
-- sketches in text, fragments, tiny poems, observational paragraphs.
-- Most stay private (`shared_at IS NULL`). The activity selector in
-- Sprint 4a may choose to share one (~5-10% of creations), at which
-- point `shared_at` is populated.
--
-- The temptation to surface every creation as engagement is real and
-- must be refused at the dispatch layer. This table is the data side;
-- the discipline lives in the activity selector and the user-facing
-- surfaces.

CREATE TABLE IF NOT EXISTS companion_creations (
    id                  INTEGER PRIMARY KEY,
    companion_id        TEXT NOT NULL,
    kind                TEXT NOT NULL,              -- sketch|note|fragment|poem|patch|...
    title               TEXT,
    content             TEXT,
    artifact_uri        TEXT,                       -- when content is binary/external
    origin_journal_id   INTEGER,                    -- the journal entry that prompted this
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    shared_at           TEXT,                       -- null until she chooses to share
    FOREIGN KEY(origin_journal_id) REFERENCES companion_journal(id)
);

CREATE INDEX IF NOT EXISTS idx_creations_companion_time
    ON companion_creations(companion_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_creations_shared
    ON companion_creations(companion_id, shared_at DESC)
    WHERE shared_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_creations_kind
    ON companion_creations(companion_id, kind, created_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (155, 'companion_creations: things she makes (commitment 5)');
