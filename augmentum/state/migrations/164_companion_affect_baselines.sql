-- 164_companion_affect_baselines.sql
-- Becca runtime, Lane 2 §1.3 — per-user affect baseline distributions.
--
-- The perception loop builds a per-(user, companion) baseline against
-- which short-window affect patterns are evaluated. Depression doesn't
-- look the same on two people; one person's "settled" is another
-- person's "low." The baseline encodes "what is normal for this person
-- with this companion" so anomaly detection compares against the right
-- reference frame.
--
-- Three windows are maintained: 7d (current), 30d (this stretch),
-- 180d (long arc). Each is a row in this table keyed by window_days.
-- Nightly consolidation rebuilds them from personality_facet_activations
-- (migration 160) via a single GROUP BY query per window.
--
-- The trust gate: when turn_count < 60 for the 30d window, the
-- perception loop does NOT voice any care-noticings. She doesn't act
-- like she knows him on day three. Below that threshold, observations
-- still get written to the journal with confidence='early', they just
-- never surface.
--
-- facet_mean / facet_stddev are JSON-encoded {facet_name: float} maps.
-- Storing as JSON because the facet vocabulary is open-ended at the
-- application layer (vocabulary lives in personality_facets); a
-- columnar schema would couple the table to the vocabulary version.

CREATE TABLE IF NOT EXISTS companion_affect_baselines (
    user_id              TEXT NOT NULL,
    companion_id         TEXT NOT NULL,
    window_days          INTEGER NOT NULL,                  -- 7 | 30 | 180
    facet_mean_json      TEXT NOT NULL DEFAULT '{}',        -- {facet: mean_intensity}
    facet_stddev_json    TEXT NOT NULL DEFAULT '{}',        -- {facet: stddev}
    activation_density   REAL NOT NULL DEFAULT 0.0,         -- facets per turn, averaged
    turn_count           INTEGER NOT NULL DEFAULT 0,        -- sample size — gates trust
    last_updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, companion_id, window_days)
);

CREATE INDEX IF NOT EXISTS idx_affect_baseline_user
    ON companion_affect_baselines(user_id, companion_id, last_updated_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (164, 'companion_affect_baselines: per-user affect baselines (7d/30d/180d) for perception loop');
