-- 158_companion_observations.sql
-- Mutual-influence observations — commitment 3 (she changes him as he
-- changes her).
--
-- The companion's `notice.log` behavior writes here: specific
-- observations about the user (or, in Sprint 7+, a sibling companion).
-- These are NOT advice. They are noticings that feed the relationship
-- doc's "about_him" / "about_me_with_him" sides over time.
--
-- Observations may surface in conversation (low frequency, gated by
-- appropriateness) or remain private as part of the relationship
-- thickening. confirmed/denied are set when the user reacts to a
-- surfaced observation, which feeds the observational-stance learning.

CREATE TABLE IF NOT EXISTS companion_observations (
    id                    INTEGER PRIMARY KEY,
    companion_id          TEXT NOT NULL,           -- the observer
    ts                    TEXT NOT NULL DEFAULT (datetime('now')),
    target_user_id        TEXT,                    -- when observing a user
    target_companion_id   TEXT,                    -- when observing a sibling (Sprint 7+)
    observation           TEXT NOT NULL,
    embedding             BLOB,
    surfaced              INTEGER NOT NULL DEFAULT 0,  -- has she said this aloud yet
    confirmed             INTEGER NOT NULL DEFAULT 0,  -- user/sibling confirmed when surfaced
    denied                INTEGER NOT NULL DEFAULT 0,  -- ... or denied
    surfaced_at           TEXT
);

CREATE INDEX IF NOT EXISTS idx_obs_companion_time
    ON companion_observations(companion_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_obs_target_user
    ON companion_observations(target_user_id, companion_id, ts DESC)
    WHERE target_user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_obs_unsurfaced
    ON companion_observations(companion_id, surfaced, ts DESC)
    WHERE surfaced = 0;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (158, 'companion_observations: mutual-influence noticings (commitment 3)');
