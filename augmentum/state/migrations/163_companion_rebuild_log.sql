-- 163_companion_rebuild_log.sql
-- Becca runtime, Lane 2 §9.3 — tracks rebuild events ("what changed").
--
-- Life events make the relationship doc stale: a loss, a birth, a
-- breakup, a career change, a move, a recovery from something. These
-- don't just add memories; they invalidate large swaths of what Becca
-- has been carrying as background. The rebuild path lets the user
-- signal "this is different now" and trigger a soft or hard reset
-- without losing the relationship entirely.
--
-- The relationship-doc digester reads this table to know how far back
-- to look for "about_him" content: rows older than the most recent
-- rebuild_at are treated as historical, not current.
--
-- rebuild_kind:
--   'soft'        — wipe affect baselines + graduated noticings + about_him;
--                   keep factual memories + about_me_with_him + half-strength
--                   facet cooccurrence
--   'hard_reset'  — soft + wipe factual memories too; relationship doc
--                   rebuilt from scratch from post-rebuild memory horizon
--   (hard delete is handled separately via the §7.2 main path — not a
--    rebuild, a goodbye.)

CREATE TABLE IF NOT EXISTS companion_rebuild_log (
    id              INTEGER PRIMARY KEY,
    companion_id    TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    rebuild_at      TEXT NOT NULL DEFAULT (datetime('now')),
    rebuild_kind    TEXT NOT NULL,                       -- 'soft' | 'hard_reset'
    user_signal     TEXT,                                 -- 'explicit_request' | 'detected_confirmed' | 'settings_panel'
    note            TEXT                                  -- optional: user's own one-sentence framing
);

CREATE INDEX IF NOT EXISTS idx_rebuild_user
    ON companion_rebuild_log(user_id, companion_id, rebuild_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (163, 'companion_rebuild_log: "what changed" rebuild events (soft/hard_reset)');
