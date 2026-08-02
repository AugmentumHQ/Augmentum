-- 186_companion_today_reflections.sql
-- Aletheia × Augmentum arc — Today entry surface.
--
-- One reflection row per (user, companion, local-date). Becca writes a
-- short daily journal-to-the-user summarizing what she did since you
-- last talked. Lives at the top of the Notes drawer.
--
-- ``content_text`` is the prose she wrote (validated through
-- safe_journal: structural / injection / refs / quality). Inline
-- citations like [note:N] / [wondering:N] reference source rows; the
-- list of those refs is mirrored in ``source_refs_json`` for forget /
-- redact gestures.
--
-- ``settled_at`` is null until end-of-day; while null, opportunistic
-- regen can rewrite ``content_text`` (debounced ≤ 1/hour). After
-- settle, the row is immutable except for ``quarantined`` flips.
--
-- Per-user isolation is load-bearing — strict (user_id, companion_id)
-- PK + ON DELETE CASCADE so user-delete sweeps these rows along with
-- the rest of the companion strand.

CREATE TABLE IF NOT EXISTS companion_today_reflections (
    user_id            TEXT NOT NULL,
    companion_id       TEXT NOT NULL,
    date_local         TEXT NOT NULL,                       -- YYYY-MM-DD in user's local tz
    generated_at       TEXT NOT NULL DEFAULT (datetime('now')),
    last_updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    settled_at         TEXT,                                -- null until day rolls / explicit settle
    content_text       TEXT NOT NULL DEFAULT '',
    source_refs_json   TEXT NOT NULL DEFAULT '[]',          -- [{kind, id}, ...]
    validation_score   REAL NOT NULL DEFAULT 1.0,
    quarantined        INTEGER NOT NULL DEFAULT 0,
    quarantine_reason  TEXT,
    PRIMARY KEY (user_id, companion_id, date_local)
);

CREATE INDEX IF NOT EXISTS idx_companion_today_user_date
    ON companion_today_reflections(user_id, companion_id, date_local DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (186, 'companion_today_reflections: daily in-her-voice journal-to-user surface');
