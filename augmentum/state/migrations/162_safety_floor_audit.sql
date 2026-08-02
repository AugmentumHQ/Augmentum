-- 162_safety_floor_audit.sql
-- Becca runtime, Lane 2 §6.7 — anonymized aggregate audit for the
-- regression-floor classifier (acute explicit-language detection).
--
-- This table holds NO user content and NO user_id. The per-turn
-- fingerprint is an HMAC over a per-install salt + a turn-id hash, so
-- the same turn doesn't double-count across review but no two users'
-- traffic is correlatable. The safety team reviews this weekly for
-- false-positive / false-negative drift and per-surface skew; the
-- quarterly threshold tune draws from this table.
--
-- The companion table is companion_safety_floor_rolling_user_view —
-- per-user rolling counters for the user's own awareness panel
-- (opt-in; defaults off). Never includes scores or content.

CREATE TABLE IF NOT EXISTS companion_safety_floor_audit (
    id                 INTEGER PRIMARY KEY,
    fingerprint        TEXT NOT NULL,                   -- HMAC(install_salt, turn_id_hash)
    fired              INTEGER NOT NULL,                 -- 0 or 1
    score              REAL NOT NULL,                    -- classifier output [0, 1]
    surface            TEXT NOT NULL,                    -- 'free_chat' | 'voice' | 'narrative_boundary' | 'coder'
    threshold_used     REAL NOT NULL,
    locale             TEXT NOT NULL,
    classifier_version TEXT NOT NULL,
    user_outcome       TEXT,                             -- 'engaged_resource' | 'dismissed' | 'continued_conversation' | NULL
    outcome_at         TEXT,
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_safety_floor_audit_review
    ON companion_safety_floor_audit(created_at DESC, fired, surface);

-- Per-install rolling summary for the user's own awareness panel.
-- Populated nightly by the consolidation pipeline; never includes scores
-- or content. The user can see "in the last 30 days, the resource panel
-- surfaced N times, you engaged with it M times." That's it.
CREATE TABLE IF NOT EXISTS companion_safety_floor_rolling_user_view (
    user_id            TEXT PRIMARY KEY,
    last_30d_fires     INTEGER NOT NULL DEFAULT 0,
    last_30d_engaged   INTEGER NOT NULL DEFAULT 0,
    last_updated       TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (162, 'companion_safety_floor_audit: anonymized regression-floor audit + per-user rolling view');
