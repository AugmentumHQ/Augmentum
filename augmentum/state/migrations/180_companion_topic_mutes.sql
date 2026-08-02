-- 180_companion_topic_mutes.sql
-- Aletheia × Augmentum arc, Sprint 3 Piece 12.
--
-- Topic mutes — the "stop researching this thread" gesture. Wondering
-- generator (Sprint 2) checks this table at every potential write to
-- skip threads matching an active mute scope.
--
-- ── Scope semantics ──────────────────────────────────────────────────
--
-- scope_json is a structured JSON document:
--
--     {
--       "domains": ["example.com", "other.com"],
--       "keywords": ["politics", "crypto"]
--     }
--
-- A thread matches an active mute when:
--   * Any of its domains overlaps muted_domains, OR
--   * At least 2 of its keywords overlap muted_keywords
--
-- The "2 keyword overlap" threshold prevents single-keyword false
-- positives from runaway-muting everything that mentions a common word.
-- Domain overlap is single-hit because domain is already specific.
--
-- ── Expiry ───────────────────────────────────────────────────────────
--
-- Default 90 days. Tuning surface: companion_topic_mute_default_days.
-- After expiry, the row stays for audit but is excluded from active
-- checks (WHERE expires_at > datetime('now')).
--
-- ── note_id provenance ──────────────────────────────────────────────
--
-- When a mute is created by user-clicking "mute this topic" on a note,
-- note_id captures which note triggered it. Lets the Observatory show
-- "you muted this because of [note]" for transparency.

CREATE TABLE IF NOT EXISTS companion_topic_mutes (
    id           INTEGER PRIMARY KEY,
    user_id      TEXT NOT NULL,
    companion_id TEXT NOT NULL,
    scope_json   TEXT NOT NULL,                                  -- {domains:[], keywords:[]}
    note_id      INTEGER,                                         -- the note that triggered (when user-initiated)
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at   TEXT NOT NULL
);

-- Hot-path index — the wondering generator's check on every write.
CREATE INDEX IF NOT EXISTS idx_topic_mutes_user_active
    ON companion_topic_mutes(user_id, companion_id, expires_at);

-- Audit index — Observatory's "active mutes" panel.
CREATE INDEX IF NOT EXISTS idx_topic_mutes_user_time
    ON companion_topic_mutes(user_id, companion_id, created_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (180, 'companion_topic_mutes: scope-based topic suppression (Sprint 3 Piece 12)');
