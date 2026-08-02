-- 154_companion_journal.sql
-- Private inner stream — commitment 1 (an inside that exists when nobody is watching).
--
-- The companion writes here during reflective time. Entries are queryable
-- but never surfaced unprompted: the inspectability primitive lets the
-- user read on demand, but no notification, no UI badge, no auto-surface
-- in dispatch. This is the architectural protection that prevents
-- inwardness from being optimized away into a "feature."
--
-- entry_type is open-ended but the runtime ships with four canonical
-- kinds: observation (passing notice), wondering (open question),
-- noticing (something about the user — these may seed companion_observations),
-- unfinished (a thought she hasn't resolved yet).
--
-- affect_tag is the runtime's affect state at write time. unsure and
-- not_okay are first-class per commitment 6.

CREATE TABLE IF NOT EXISTS companion_journal (
    id                  INTEGER PRIMARY KEY,
    companion_id        TEXT NOT NULL,
    user_id             TEXT,                       -- nullable: some entries aren't about a user
    entry_type          TEXT NOT NULL DEFAULT 'observation',
                          -- observation|wondering|noticing|unfinished|...
    content             TEXT NOT NULL,
    embedding           BLOB,
    affect_tag          TEXT,                       -- her affect at write time
    related_memory_ids  TEXT,                       -- JSON array of memories.id
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cj_companion_time
    ON companion_journal(companion_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cj_user_time
    ON companion_journal(user_id, companion_id, created_at DESC)
    WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cj_entry_type
    ON companion_journal(companion_id, entry_type, created_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (154, 'companion_journal: private inner stream (commitment 1)');
