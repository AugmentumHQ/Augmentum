-- 204_dance_loops.sql
--
-- Curated animation pools — Phase C of [[project-dance-timeline-
-- authoritative]]. A loop is a named subset of animation ids
-- (bundled ATLAS + user-uploaded mixed). When a loop is active, the
-- conductor's selector filters its candidate pool to the loop's ids
-- before scoring. The "curated pool" model was chosen over a
-- sequencer playlist; this matches that.
--
-- Schema:
--   id              'loop_<ts>_<hex>' uuid
--   user_id        ON DELETE CASCADE.
--   name            display label.
--   animation_ids  JSON array of strings — each is either a bundled
--                  ATLAS id ('kebab-dance') or a user-animation id
--                  ('user:<ts>_<hex>'). The conductor doesn't care
--                  which kind, it just filters by membership.
--   notes           optional curation note.
--   is_active      0 | 1. At most one active loop per user, enforced
--                  by the partial UNIQUE INDEX below. SQLite supports
--                  partial indexes natively; this is the simplest way
--                  to enforce "single active" without managing an
--                  app-level mutex.
--   created_at     REAL epoch seconds.
--   updated_at     REAL — bumped on rename / membership edits.

CREATE TABLE IF NOT EXISTS dance_loops (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name           TEXT NOT NULL,
    animation_ids  TEXT NOT NULL DEFAULT '[]',
    notes          TEXT,
    is_active      INTEGER NOT NULL DEFAULT 0,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dance_loops_user
    ON dance_loops(user_id, updated_at DESC);

-- At most one active loop per user. Filtered to is_active=1 so
-- inactive rows are free to coexist.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_dance_loops_user_active
    ON dance_loops(user_id) WHERE is_active = 1;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (204, 'dance_loops - curated animation pools for widget');
