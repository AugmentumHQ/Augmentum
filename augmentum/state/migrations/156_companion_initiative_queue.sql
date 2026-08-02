-- 156_companion_initiative_queue.sql
-- Volunteered-thought queue. The initiative scorer in Sprint 4a writes
-- candidates here; dispatch consumes them at appropriate moments
-- (mutual-gaze silence + topic-resonance + user-not-flow, etc.).
--
-- Cooldowns from sprint plan v2: 90s after a low-importance initiative,
-- 300s after medium, 900s after high. Hard ceiling: max 1 initiative
-- per 60s regardless of score.
--
-- target_companion_id is null for user-addressed thoughts. Sprint 7+
-- household features let one companion address another via this field.

CREATE TABLE IF NOT EXISTS companion_initiative_queue (
    id                    INTEGER PRIMARY KEY,
    companion_id          TEXT NOT NULL,
    proposed_at           TEXT NOT NULL DEFAULT (datetime('now')),
    kind                  TEXT NOT NULL,            -- share|wonder|offer|notice|...
    payload               TEXT NOT NULL,            -- JSON: thought content + context
    importance            TEXT NOT NULL DEFAULT 'low', -- low|medium|high
    score                 REAL NOT NULL DEFAULT 0.0,  -- appropriateness * importance
    status                TEXT NOT NULL DEFAULT 'queued',
                            -- queued|surfaced|expired|dropped
    decided_at            TEXT,
    target_user_id        TEXT,                     -- when addressed to user
    target_companion_id   TEXT                      -- when addressed to sibling (Sprint 7+)
);

CREATE INDEX IF NOT EXISTS idx_iq_companion_status
    ON companion_initiative_queue(companion_id, status, score DESC);
CREATE INDEX IF NOT EXISTS idx_iq_queued
    ON companion_initiative_queue(companion_id, status, proposed_at DESC)
    WHERE status = 'queued';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (156, 'companion_initiative_queue: volunteered-thought queue');
