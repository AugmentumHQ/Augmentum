-- 183_companion_journal_archive.sql
-- Aletheia × Augmentum arc, Sprint 1 (R1).
--
-- Stores consolidated paragraphs that summarize 7-day journal windows.
-- The weekly heal job (Sprint 4) writes here: groups entries older
-- than 30d into 7-day windows, summarizes each via the utility-tier
-- LLM, writes one row per window. Original entries get
-- ``archived_at`` stamped and remain in companion_journal for 60d
-- post-archive (soft-delete window), then are hard-deleted at 90d.
--
-- This is the forgetting-curve made explicit: detail decays into
-- summary, then summary persists. The summary stays addressable for
-- retrieval; the originals become recoverable-only from backup.
--
-- Per-user scoped (user_id + companion_id) so consolidation is
-- isolated. Migration 179's per-user pivot established the
-- invariant; this table inherits it directly via its own user_id
-- column.

CREATE TABLE IF NOT EXISTS companion_journal_archive (
    id                    INTEGER PRIMARY KEY,
    user_id               TEXT NOT NULL,
    companion_id          TEXT NOT NULL,
    window_start          TEXT NOT NULL,        -- ISO; inclusive
    window_end            TEXT NOT NULL,        -- ISO; exclusive
    entry_ids             TEXT NOT NULL,        -- JSON array of source journal ids
    summary               TEXT NOT NULL,        -- the consolidated paragraph
    source_count          INTEGER NOT NULL,     -- len(entry_ids); cached for cheap UI
    avg_confidence        REAL,                 -- mean confidence_numeric over window
    affect_signature_json TEXT,                 -- {valence_mean, arousal_mean, top_facets}
    created_at            TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cja_user_time
    ON companion_journal_archive(user_id, companion_id, window_start DESC);

CREATE INDEX IF NOT EXISTS idx_cja_user_recent
    ON companion_journal_archive(user_id, companion_id, created_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (183, 'companion_journal_archive: consolidated 7-day window summaries');
