-- 117_narrative_ledger_entries.sql
-- Per-branch ledger with row-level branch tagging. Replaces the JSON-array
-- storage in narrative_memory.memory_ledger so all three memory tiers (STATE,
-- LEDGER, ARCHIVE) share the same row-tagged + ancestry-filtered architecture.
-- Reads are SQL with index hits, not Python-side filtering on a deserialized blob.

CREATE TABLE IF NOT EXISTS narrative_ledger_entries (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    branch_id   TEXT NOT NULL,
    round_num   INTEGER NOT NULL,
    category    TEXT NOT NULL,
    content     TEXT NOT NULL,
    user_id     TEXT REFERENCES users(id),
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ledger_entries_lookup
    ON narrative_ledger_entries(session_id, branch_id, round_num);

CREATE INDEX IF NOT EXISTS idx_ledger_entries_user
    ON narrative_ledger_entries(user_id, session_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (117, 'narrative_ledger_entries table');
